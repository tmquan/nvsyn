import torch
import math
import numpy as np
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer import FoVOrthographicCameras
from kaolin.render.camera import Camera
from kaolin.render.camera import CameraExtrinsics
from kaolin.render.camera import PinholeIntrinsics
from kaolin.render.camera import OrthographicIntrinsics
from kaolin.render.camera import ExtrinsicsRep

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)

def make_cameras_dea(
    dist: torch.Tensor, # ndc
    elev: torch.Tensor, # degree
    azim: torch.Tensor, # degree
    fov: int = 40,      # degree
    znear: int = 7.0,   # ndc
    zfar: int = 9.0,    # ndc
    width: int = 800,
    height: int = 600,
    is_orthogonal: bool = False,
    return_pytorch3d: bool = True
):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    if return_pytorch3d:
        if is_orthogonal:
            return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
        else:
            return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)
    else:
        extrinsics = ExtrinsicsRep._from_world_in_cam_coords(rotation=R, translation=T, device=_device)
        if is_orthogonal:
            intrinsics = OrthographicIntrinsics.from_frustum(width=width, height=height, near=znear, far=zfar, fov=np.deg2rad(fov), device='cuda')
            return Camera(extrinsics=extrinsics, intrinsics=intrinsics)
        else:
            intrinsics = PinholeIntrinsics.from_fov(width=width, height=height, near=znear, far=zfar, fov_distance=1.0, device='cuda')
            return Camera(extrinsics=extrinsics, intrinsics=intrinsics) 


# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 nn.init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 nn.init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 nn.init.orthogonal_(m.weight.data, gain=init_gain)
#             elif init_type == 'zero':
#                 nn.init.constant_(m.weight.data, 0)         
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             nn.init.normal_(m.weight.data, 1.0, init_gain)
#             nn.init.constant_(m.bias.data, 0.0)
#     # print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>
    