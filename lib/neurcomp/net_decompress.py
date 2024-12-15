from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import random
import time
import json
import re
import math

from sklearn.cluster import KMeans

import torch as th
from torch.utils.data import DataLoader

from utils import tiled_net_out, field_from_net

from data import VolumeDataset

from func_eval import trilinear_f_interpolation,finite_difference_trilinear_grad

from siren import FieldNet, compute_num_neurons
from net_coder import SirenDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--volume', default=None, help='path to volumetric dataset (optional)')
parser.add_argument('--compressed', required=True, help='path to compressed file')
parser.add_argument('--recon', default='recon', help='path to reconstructed file output')
parser.add_argument('--resolution', type=str, default=None, help='resolution for reconstruction (format: NxNxN)')
parser.add_argument('--format', choices=['npy', 'vtk'], default='npy', help='output format')

parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
parser.set_defaults(cuda=False)

opt = parser.parse_args()
print(opt)

# Load decoder and network
decoder = SirenDecoder()
net = decoder.decode(opt.compressed)
if opt.cuda:
    net = net.cuda()
net.eval()

# Handle volume and dataset creation
if opt.volume:
    np_volume = np.load(opt.volume).astype(np.float32)
    volume = th.from_numpy(np_volume)
    vol_res = volume.shape
    dataset = VolumeDataset(volume)
elif opt.resolution:
    res = [int(x) for x in opt.resolution.split('x')]
    vol_res = res
    dataset = VolumeDataset(th.zeros(res))  # Create dummy dataset with desired resolution
else:
    raise ValueError("Either --volume or --resolution must be specified")

# Reconstruct
reconstructed = field_from_net(dataset, net, opt.cuda, verbose=True)

# Save output
if opt.format == 'npy':
    np.save(f"{opt.recon}.npy", reconstructed.cpu().numpy())
else:  # vtk
    from pyevtk.hl import imageToVTK
    imageToVTK(opt.recon, pointData={'sf': reconstructed.cpu().numpy()})

# Print compression stats if original volume exists
if opt.volume:
    vol_size = np.prod(vol_res) * 4  # 4 bytes per float32
    compressed_size = os.path.getsize(opt.compressed)
    cr = vol_size/compressed_size
    print(f"Compression ratio: {cr:.2f}x")
