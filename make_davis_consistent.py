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
from dataset.range_transform import inv_im_trans

import imageio
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='/isilon/Datasets/Aharon/stcn.pth')
parser.add_argument('--davis_path', default='/isilon/Datasets/Aharon/stcn_training_datasets/DAVIS/2017')
parser.add_argument('--output')
parser.add_argument('--split', help='val/testdev', default='train')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--tta', default=False, type=bool)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)
def get_start_ind(ti, length, n_memorized):
    context = 5
    inds = np.concatenate([np.arange(max(ti-context, 0), max(ti-n_memorized+1, 0)), np.arange(min(ti+n_memorized,length-n_memorized), min(ti+context,length-n_memorized))])
    return inds

def get_indecies(ti, n_frames, length):
    Inds = np.random.choice(np.concatenate([np.arange(0, ti), np.arange(ti+1, length)]), n_frames, False)
    Inds.sort()
    return Inds

def arr2vid(arr, video_filename="out", fps=10):
    w = imageio.get_writer(video_filename + ".mp4", fps=fps)
    for im in arr:
        w.append_data(im)
    w.close()

def overlay(x,m):
    overlay1 = x / 255.0
    overlay1[..., 0:1] += m[..., None]
    overlay1[overlay1 > 1] = 1
    return (overlay1 * 255.0).astype("uint8")

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt', tta=args.tta)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'train':
    test_dataset = DAVISTestDataset(davis_path + '/trainval', imset='2017/train.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

scores = []
# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=False):
    with torch.cuda.amp.autocast(enabled=args.amp):
        with torch.no_grad():
            rgb = data['rgb'].cuda()
            msk = data['gt'][0].cuda()
            info = data['info']
            name = info['name'][0]
            k = len(info['labels'][0])
            size = info['size_480p']
            n_frames = rgb.shape[1]
            print(f"n_frames:{n_frames}")
            torch.cuda.synchronize()
            out_masks = torch.zeros((n_frames, k+1, *size))
            # #hacky but efficient way to do both forward and backward
            # out_masks = out_masks[::-1]
            # counter = counter[::-1]
            # rgb = rgb[:, :-1]
            # msk = msk[:, :-1]

            processor = InferenceCore(prop_model, rgb, k, top_k=top_k,
                                      mem_every=1, include_last=args.include_last)
            for ti in range(n_frames):
                processor.memorize(msk[:, ti], ti)

            prob = processor.propagate()

            for ti in range(n_frames):
                print(f"postprocessing to {ti}")
                prob = processor.prob[:, ti]
                prob = unpad(prob, processor.pad)
                prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                out_masks[ti] = prob[:, 0]


        rgb_orig = inv_im_trans(rgb[0])
        rgb_orig*=255
        rgb_orig = rgb_orig.cpu().numpy().astype("uint8")
        rgb_orig = rgb_orig.transpose((0,2,3,1))
        out_masks = (out_masks.detach().cpu().numpy())
        out_masks *= 255
        out_masks = out_masks.astype("uint8")
        # plt.imshow(out_masks[10][1:].max(0)), plt.show()

        # Save the results
        np.save(out_path+"/"+name, out_masks)
        msk = msk.cpu().numpy()
        msk = np.concatenate([
            1 - msk.sum(0)[None,],
            msk
        ], 0)
        origmasks = msk.argmax(0)[:, 0]
        multiple = np.floor(255/origmasks.max())
        # origmasks = origmasks * multiple
        origmasks = origmasks.astype("uint8")
        predicted_masks = out_masks.argmax(1)
        predicted_masks = predicted_masks*multiple
        origmasks = origmasks*multiple
        predicted_masks = predicted_masks.astype("uint8")
        origmasks = origmasks.astype("uint8")
        print("outputing videos")
        arr2vid(np.concatenate([overlay(rgb_orig, origmasks), overlay(rgb_orig, predicted_masks), overlay(rgb_orig, np.abs(origmasks-predicted_masks))], 1), out_path+"/"+name+"_comparison")
        # arr2vid(np.concatenate([origmasks, out_masks[:,1], np.abs(origmasks-predicted_masks)], 1), out_path+"/"+name+"_comparison_masks")
        #todo avoid frame jumps and propagate to every frame.
        del rgb
        del msk
        del processor
        del out_masks
        del predicted_masks
        del origmasks

