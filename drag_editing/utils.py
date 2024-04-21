import torch.nn.functional as F
import torch
import numpy as np
import cv2
import math
from gaussiansplatting.scene.cameras import Simple_Camera

def simple_camera_to_c2w_k(cam):
    cam_pos = - cam.R @ cam.T
    cam_rot = cam.R
    width, height = cam.image_width, cam.image_height
    fy = fov2focal(cam.FoVy, width)
    fx = fov2focal(cam.FoVx, height)
    c2w = torch.eye(4, device='cuda')
    c2w[:3, :3] = torch.from_numpy(cam_rot).to(c2w)
    c2w[:3, 3] = torch.from_numpy(cam_pos).to(c2w)

    K = torch.tensor([
        [fx, 0, width / 2],
        [0, fy, height / 2],
        [0, 0, 1],
    ], device='cuda')
    return c2w, K

def c2w_k_to_simple_camera(c2w, K):
    width, height = K[:2, 2] * 2
    cam_pos = c2w[:3, 3]
    cam_rot = c2w[:3, :3]
    FoVx = focal2fov(K[0, 0], width)
    FoVy = focal2fov(K[1, 1], height)
    cam_R = cam_rot
    cam_T = - cam_R.T @ cam_pos
    
    return Simple_Camera(0, cam_R, cam_T, FoVx, FoVy, height, width, "", 0)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
def unproject(c2w, K, points2d, depth):
    """
    c2w: (4, 4)
    K: (3, 3)
    points2d: (N, 2) - (x, y) in the range [0; (W,H)]
    depth: (H, W)
    """

    point_depth = depth[points2d[:, 1].long(), points2d[:, 0].long()] # (N,)
    d = torch.cat((points2d, torch.ones((points2d.shape[0], 1)).to(points2d.device)), dim=-1) # (N, 3)
    d = d @ K.inverse().T # (N, 3)
    # d = d / d.norm(1, keepdim=True) # (N, 3)
    d = d @ c2w[:3, :3].T # (N, 3)
    camera_pos = c2w[:3, 3] # (3,)

    points3d = d * point_depth[:, None] + camera_pos[None, :] # (N, 3)

    return points3d

def to_homogeneous(v):
    if type(v) is np.ndarray:
        v = np.concatenate((v, np.ones((*v.shape[:-1], 1))), axis=-1)
    elif type(v) is torch.Tensor:
        v = torch.cat((v, torch.ones((*v.shape[:-1], 1)).to(v.device)), dim=-1)
    return v

def get_points(img,
               sel_pix):
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)