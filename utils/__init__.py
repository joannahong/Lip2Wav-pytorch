import os
import random
import torch
import numpy as np

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o775)

def setSeed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def mode(obj, hp, model = False):
    if model and hp.use_cuda:
        obj = obj.cuda()
    elif hp.use_cuda:
        obj = obj.cuda(non_blocking = hp.pin_mem)
    return obj

def to_var(tensor, hp):
    var = torch.autograd.Variable(tensor)
    return mode(var, hp)

def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)

def get_mask_from_lengths(lengths, hp, pad = False):
    max_len = torch.max(lengths).int().item()
    if pad and max_len % hp.n_frames_per_step != 0:
        max_len += hp.n_frames_per_step - max_len % hp.n_frames_per_step
        assert max_len % hp.n_frames_per_step == 0
    ids = torch.arange(0, max_len, out = torch.LongTensor(max_len))
    ids = mode(ids, hp)
    mask = (ids < lengths.unsqueeze(1))
    return mask
