import torch
import torch.nn as nn
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
)
from dvr.raymarcher import AbsorptionEmissionRaymarcher

def rescaled(x, val=64, eps=1e-8):
    return (x + eps) / (val + eps)

def minimized(x, eps=1e-8):
    return (x + eps) / (x.max() + eps)

def normalized(x, eps=1e-8):
    return (x - x.min() + eps) / (x.max() - x.min() + eps)

def standardized(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)


class BaseXRayVolumeRenderer(nn.Module):
    def __init__(
        self, 
        image_width: int = 256, 
        image_height: int = 256, 
        n_pts_per_ray: int = 320, 
        min_depth: float = 3.0, 
        max_depth: float = 9.0, 
        ndc_extent: float = 1.0,
    ):
        super().__init__()
        self.n_pts_per_ray = n_pts_per_ray
        self.image_width = image_width
        self.image_height = image_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ndc_extent = ndc_extent

    def create_volumes(self, features, densities):
        shape = max(features.shape[2], features.shape[3])
        return Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0 * float(self.ndc_extent) / shape,
        )

    def forward(self,
                image3d,
                cameras,
                opacity=None,
                norm_type="standardized",
                scaling_factor=1.0,
                is_grayscale=True,
                return_bundle=False,
                stratified_sampling=False
    ) -> torch.Tensor:    
        features = image3d.repeat(1, 3, 1, 1, 1) if image3d.shape[1] == 1 else image3d
        densities = opacity * scaling_factor if opacity is not None else torch.ones_like(image3d[:, [0]]) * scaling_factor
        shape = max(features.shape[2], features.shape[3])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0 * float(self.ndc_extent) / shape,
        )
        
        screen_RGBA, bundle = self.renderer(cameras=cameras, volumes=volumes)
        
        screen_RGBA = screen_RGBA.permute(0, 3, 1, 2)
        screen_RGB = screen_RGBA.mean(dim=1, keepdim=True) if is_grayscale else screen_RGBA
        
        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))

        if return_bundle:
            return screen_RGB, bundle
        return screen_RGB


class ScreenCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize raymarcher and renderer specific to Screen Centric
        raymarcher = EmissionAbsorptionRaymarcher()
        
        raysampler = NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False
        )
        
        self.renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)


class ObjectCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize raymarcher and renderer specific to Object Centric
        raymarcher = AbsorptionEmissionRaymarcher()
        
        raysampler = NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False
        )
        
        self.renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)
