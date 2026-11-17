"""RGB input conversion and deterministic spatial augmentation."""
import hashlib
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def stretch(a):
 x=a.astype(np.float32);out=np.empty_like(x)
 for c in range(3):
  low=np.percentile(x[:,:,c],2.);high=np.percentile(x[:,:,c],98.);span=high-low
  out[:,:,c]=0. if span<1e-6 else (np.clip(x[:,:,c],low,high)-low)/(span+1e-6)
 return out.transpose(2,0,1)

def recipe(seed,epoch,sid):
 r=random.Random(int.from_bytes(hashlib.sha256(('%d|%d|%s'%(seed,epoch,sid)).encode()).digest()[:8],'big'))
 return r.randint(0,8),r.randint(0,8),r.random()<.5,r.random()<.5,r.randint(0,3)

def load_rgb(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if image.size != (256, 256):
            raise ValueError('Expected a 256 x 256 image; use the dataset-specific crop rules.')
        array = np.array(image)
    return torch.from_numpy(stretch(array)).contiguous()


def augment(x, seed, epoch, sample_id):
    top, left, horizontal, vertical, turns = recipe(seed, epoch, sample_id)
    x = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze(0)[:, top:top+256, left:left+256]
    if horizontal:
        x = torch.flip(x, (2,))
    if vertical:
        x = torch.flip(x, (1,))
    if turns:
        x = torch.rot90(x, turns, (1, 2))
    return x.contiguous()
