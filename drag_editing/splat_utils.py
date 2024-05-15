import os
import numpy as np
import torch
import struct
from tqdm import tqdm
from gaussiansplatting.scene import GaussianModel

def load_splat(p):
    input = open(p, "rb")
    filesize = os.path.getsize(p)
    num_splats = filesize // 32
    pos = np.zeros((num_splats, 3))
    svec = np.zeros((num_splats, 3))
    rgb = np.zeros((num_splats, 3), dtype=np.uint8)
    opacity = np.zeros(num_splats, dtype=np.uint8)
    qvec = np.zeros((num_splats, 4), dtype=np.uint8)
    for i in tqdm(range(0, num_splats)):
        cur = input.read(32)
        pos[i] = struct.unpack("fff", cur[:12])
        svec[i] = struct.unpack("fff", cur[12:24])
        *rgb[i], opacity[i] = struct.unpack("BBBB", cur[24:28])
        qvec[i] = struct.unpack("BBBB", cur[28:32])
    
    scales = np.log(svec)
    opacities = -np.log(1 / (opacity / 255).clip(min=1e-3, max=1-1e-3) - 1)
    rots = (qvec.astype(np.float32) - 128) / 128
    C0 = 0.28209479177387814
    features_dc = (rgb / 255 - 0.5) / C0

    gaussian = GaussianModel(
        sh_degree=0,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )
    gaussian._features_dc = torch.from_numpy(features_dc).to(gaussian._features_dc.data)[:, None]
    gaussian._features_rest = torch.zeros(features_dc.shape[0], 15, 3)
    gaussian._rotation = torch.from_numpy(rots).to(gaussian._rotation.data)
    gaussian._scaling = torch.from_numpy(scales).to(gaussian._scaling.data)
    gaussian._opacity = torch.from_numpy(opacities).to(gaussian._opacity.data)[..., None]
    gaussian._xyz = torch.from_numpy(pos).to(gaussian._xyz.data)

    return gaussian