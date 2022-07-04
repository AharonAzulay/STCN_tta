import torch

from inference_memory_bank import MemoryBank
from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by
import copy

class InferenceCore:
    def __init__(self, prop_net:STCN, images, num_objects, top_k=20, mem_every=5, include_last=False):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cpu'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.mem_bank = MemoryBank(k=self.k, top_k=top_k)
        self.mem = {}

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:, idx].cuda())
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        self.mem_bank.add_memory(key_k, key_v)
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1
        for ti in this_range:
            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:, ti] = out_mask
            if ti != end:
                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:
                    # if ti == this_range[0]:
                    #     single_key_length = k16.flatten(start_dim=2).shape[-1]
                    #     self.mem_bank.mem_k = self.mem_bank.mem_k[...,]
                    prev_value = self.prop_net.encode_value(self.images[:, ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, is_temp=not is_mem_frame)

        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:, frame_idx].cuda(), qf16, self.prob[1:, frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)

    def memorize(self, mask, frame_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        k16, qv16, qf16, qf8, qf4 = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:, frame_idx].cuda(), qf16, self.prob[1:, frame_idx].cuda())
        key_k = k16.unsqueeze(2)
        self.mem_bank.add_memory(key_k, key_v)
        self.mem[frame_idx] = (k16, qv16, qf16, qf8, qf4)

    def memorize_with_frames(self, mask, frame):
        mask, _ = pad_divide_by(mask.cuda(), 16)
        frame, _ = pad_divide_by(frame, 16)
        # self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        k16, qv16, qf16, qf8, qf4 = self.prop_net.encode_key(frame.cuda())
        key_v = self.prop_net.encode_value(frame.cuda(), qf16, mask.cuda())
        key_k = k16.unsqueeze(2)
        self.mem_bank.add_memory(key_k, key_v)

    def propagate(self):
        for i in range(self.images.shape[1]):
            print(f"Propagating to {i}")
            k16, qv16, qf16, qf8, qf4 = self.mem[i]

            this_memory_bank = copy.deepcopy(self.mem_bank)
            single_key_length = k16.flatten(start_dim=2).shape[-1]
            if (i+1==self.images.shape[1]):
                this_memory_bank.mem_k = self.mem_bank.mem_k[..., :i*single_key_length]
                this_memory_bank.mem_v = self.mem_bank.mem_v[..., :i*single_key_length]
            else:
                this_memory_bank.mem_k = torch.cat([self.mem_bank.mem_k[..., :i*single_key_length], self.mem_bank.mem_k[..., (i+1)*single_key_length:]], 2)
                this_memory_bank.mem_v = torch.cat([self.mem_bank.mem_v[..., :i*single_key_length], self.mem_bank.mem_v[..., (i+1)*single_key_length:]], 2)

            out_mask = self.prop_net.segment_with_query(this_memory_bank, qf8, qf4, k16, qv16)
            out_mask = aggregate(out_mask, keep_bg=True)
            del this_memory_bank
            self.prob[:, i] = out_mask
        print("done propagating")

