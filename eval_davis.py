import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore
from matplotlib import pyplot as plt
from progressbar import progressbar
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
import sys

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument("--model", default="/isilon/Datasets/Aharon/stcn.pth")
# parser.add_argument('--davis_path', default='/isilon/Datasets/Aharon/stcn_training_datasets/DAVIS/2017')
parser.add_argument("--davis_path", default="../DAVIS/2017")
parser.add_argument("--output")
parser.add_argument("--split", help="val/testdev", default="val")
parser.add_argument("--top", type=int, default=100)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--mem_every", default=5, type=int)
parser.add_argument("--tta", default=True, type=bool)
parser.add_argument(
    "--include_last",
    help="include last frame as temporary memory?",
    action="store_true",
)
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(
    path.expanduser(davis_path + "/trainval/Annotations/480p/blackswan/00000.png")
).getpalette()

torch.autograd.set_grad_enabled(False)


def evaluate_semisupervised(all_gt_masks, all_res_masks):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write(
            "\nIn your PNG files there is an index higher than the number of objects in the sequence!"
        )
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros(
            (all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:])
        )
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = (
        np.zeros(all_gt_masks.shape[:2]),
        np.zeros(all_gt_masks.shape[:2]),
    )
    for ii in range(all_gt_masks.shape[0]):
        j_metrics_res[ii, :] = db_eval_iou(
            all_gt_masks[ii, ...], all_res_masks[ii, ...], None
        )
        f_metrics_res[ii, :] = db_eval_boundary(
            all_gt_masks[ii, ...], all_res_masks[ii, ...], None
        )
    return j_metrics_res, f_metrics_res


# Setup Dataset
if args.split == "val":
    test_dataset = DAVISTestDataset(
        davis_path + "/trainval", imset="2017/val.txt", tta=args.tta
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
elif args.split == "testdev":
    test_dataset = DAVISTestDataset(davis_path + "/test-dev", imset="2017/test-dev.txt")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == "value_encoder.conv1.weight":
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

scores = []
metrics_res = {}
metrics_res["J"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
metrics_res["F"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
orig_scores = np.load("/isilon/Datasets/Aharon/stcn_scores.npy", allow_pickle=True)
# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data["rgb"].cuda()
        msk = data["gt"][0].cuda()
        info = data["info"]
        name = info["name"][0]
        k = len(info["labels"][0])
        n_frames = rgb.shape[1]
        size = info["size_480p"]

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(
            prop_model,
            rgb,
            k,
            top_k=top_k,
            mem_every=args.mem_every,
            include_last=args.include_last,
        )
        if args.tta:
            rgb_aug = data["rgb_aug"].cuda()
            msk_aug = data["gt_aug"][0].cuda()
            n_aug_frames = rgb_aug.shape[1]
            for i in range(n_aug_frames):
                processor.memorize_with_frames(msk_aug[:, i], rgb_aug[:, i])
        processor.interact(msk[:, 0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size
        out_masks = torch.zeros(
            (processor.t, 1, *size), dtype=torch.uint8, device="cuda"
        )
        for ti in range(processor.t):
            prob = unpad(processor.prob[:, ti], processor.pad)
            prob = F.interpolate(prob, size, mode="bilinear", align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        # msk = torch.cat([1-torch.sum(msk, 0)[None, ], msk])
        # int_masks = torch.argmax(msk.int().cuda(), dim=0)
        int_masks = msk[:, :, 0]
        # score = jaccard(out_masks[:, 0].cuda(), int_masks[:, 0]).item()
        all_pred_masks = np.zeros(
            (k, out_masks.shape[0], out_masks.shape[2], out_masks.shape[3])
        )
        for i in range(1, k + 1):
            all_pred_masks[i - 1, :, :, :] = (
                out_masks[:, 0].cpu().numpy() == i
            ).astype(np.uint8)
        # score = batched_jaccard(out_masks[len(out_masks)-info['num_frames']+1:, 0].detach().cpu().numpy(), int_masks[len(out_masks)-info['num_frames']+1:, 0].detach().cpu().numpy())
        all_gt_masks = int_masks.detach().cpu().numpy()
        # all_pred_masks = out_masks[len(out_masks)-info['num_frames']+1:, 0].detach().cpu().numpy()
        real_start_ind = len(out_masks) - info["num_frames"].item() + 1
        j_metrics_res, f_metrics_res = evaluate_semisupervised(
            all_gt_masks[:, real_start_ind:], all_pred_masks[:, real_start_ind:]
        )
        JF = 0
        num = 0
        for ii in range(all_gt_masks.shape[0]):
            [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
            JF += JM
            metrics_res["J"]["M"].append(JM)
            metrics_res["J"]["R"].append(JR)
            metrics_res["J"]["D"].append(JD)
            [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
            JF += FM
            metrics_res["F"]["M"].append(FM)
            metrics_res["F"]["R"].append(FR)
            metrics_res["F"]["D"].append(FD)
            num += 1
        # print(JF/(2*num))

        # plt.subplot(121)
        # plt.imshow(out_masks[10, 0].detach().cpu().numpy())
        # plt.title(score)
        # plt.subplot(122)
        # plt.imshow(int_masks[10, 0].detach().cpu().numpy())
        # plt.show()

        scores.append(j_metrics_res)
        iougain = (
            np.mean(np.mean(scores[-1], 0) - np.mean(orig_scores[len(scores) - 1], 0))
        ) * 100
        print(f"Class: {name}, IoU gain: {iougain}")
        plt.plot(np.mean(scores[-1], 0), "r"), plt.plot(
            np.mean(orig_scores[len(scores) - 1], 0), "k"
        ), plt.title(iougain), plt.show()
        # print(f"Class: {name}, IoU gain: {(np.mean(scores[-1])) * 100}")
        out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        # Save the results
        # this_out_path = path.join(out_path, name)
        # os.makedirs(this_out_path, exist_ok=True)
        # for f in range(out_masks.shape[0]):
        #     img_E = Image.fromarray(out_masks[f])
        #     img_E.putpalette(palette)
        #     img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

        del out_masks
        del all_gt_masks
        del all_pred_masks
        del rgb
        del msk
        del processor
        if args.tta:
            del msk_aug
            del rgb_aug

J, F = metrics_res["J"], metrics_res["F"]
g_measures = [
    "J&F-Mean",
    "J-Mean",
    "J-Recall",
    "J-Decay",
    "F-Mean",
    "F-Recall",
    "F-Decay",
]
final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.0
g_res = np.array(
    [
        final_mean,
        np.mean(J["M"]),
        np.mean(J["R"]),
        np.mean(J["D"]),
        np.mean(F["M"]),
        np.mean(F["R"]),
        np.mean(F["D"]),
    ]
)

print("J&F:" + str(g_res[0]), "J:" + str(g_res[1] * 100), "F:" + str(g_res[4] * 100))
print("Total processing time: ", total_process_time)

# todo remove memory after first 5 framea.
# np.save("/isilon/Datasets/Aharon/stcn_scores.npy", scores)
