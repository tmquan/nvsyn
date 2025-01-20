from contextlib import contextmanager, nullcontext

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import diffusers

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch

from monai.utils import optional_import
from monai.networks.nets import PatchDiscriminator, SwinUNETR
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.networks.layers.factories import Norm
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import PSNRMetric, SSIMMetric

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple, List
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf

from dvr.renderer import ObjectCentricXRayVolumeRenderer
from util.camera import make_cameras_dea

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'zero':
                nn.init.constant_(m.weight.data, 0)         
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class SynLightningModule(LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        
        self.fwd_renderer = ObjectCentricXRayVolumeRenderer(
            image_width=model_cfg.img_shape,
            image_height=model_cfg.img_shape,
            n_pts_per_ray=model_cfg.n_pts_per_ray,
            min_depth=model_cfg.min_depth,
            max_depth=model_cfg.max_depth,
            ndc_extent=model_cfg.ndc_extent,
        )
        
        # self.unet2d_model = SwinUNETR(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=2*model_cfg.vol_shape+model_cfg.fov_depth,
        #     img_size=self.model_cfg.img_shape, 
        #     feature_size=48,
        #     depths=(2, 2, 2, 2),
        #     num_heads=(3, 6, 12, 24),
        #     norm_name="instance",
        #     drop_rate=0.5,
        #     attn_drop_rate=0.5,
        #     dropout_path_rate=0.5,
        #     normalize=True,
        #     use_checkpoint=False,
        #     downsample="mergingv2",
        #     use_v2=True,
        # )
        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=model_cfg.fov_depth + 2*model_cfg.vol_shape,
            channels=[128, 256, 256],
            attention_levels=[False, True, True],
            num_head_channels=[0, 256, 256],
            num_res_blocks=2,
            with_conditioning=True, 
            cross_attention_dim=4, # Condition with straight/hidden view  # flatR | flatT
            upcast_attention=True,
            use_flash_attention=True,
            use_combined_linear=True,
            dropout_cattn=0.5
        )
        init_weights(self.unet2d_model, "normal")

        self.unet3d_model = None
        # self.unet3d_model = DiffusionModelUNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=1,
        #     channels=[128, 128, 256],
        #     attention_levels=[False, False, True],
        #     num_head_channels=[0, 0, 256],
        #     num_res_blocks=2,
        #     with_conditioning=True, 
        #     cross_attention_dim=4, # Condition with straight/hidden view  # flatR | flatT
        #     upcast_attention=True,
        #     use_flash_attention=True,
        #     use_combined_linear=True,
        #     dropout_cattn=0.5
        # )
        # self.unet3d_model = SwinUNETR(
        #     spatial_dims=3,
        #     img_size=self.model_cfg.vol_shape,
        #     in_channels=1,
        #     out_channels=1,
        #     feature_size=36,
        #     # use_checkpoint=True,
        # )
        # init_weights(self.unet3d_model, "zero")

        # self.p_disc_model = None
        self.p_disc_model = PatchDiscriminator(
            spatial_dims=3, 
            num_layers_d=4, 
            channels=64, 
            in_channels=1, 
            out_channels=1
        )
        init_weights(self.p_disc_model, init_type="normal", init_gain=0.1)
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        self.perc25d_loss = None
        # self.perc25d_loss = PerceptualLoss(
        #     spatial_dims=3, 
        #     network_type="radimagenet_resnet50", 
        #     is_fake_3d=True, 
        #     fake_3d_ratio=8/256.
        # ).eval()

        self.perc30d_loss = None
        # self.perc30d_loss = PerceptualLoss(
        #     spatial_dims=3, 
        #     network_type="medicalnet_resnet50_23datasets", 
        #     is_fake_3d=False, 
        #     # fake_3d_ratio=10/256.
        # ).eval()

        if model_cfg.phase=="finetune":
            pass
        
        if self.train_cfg.ckpt:
            print("Loading.. ", self.train_cfg.ckpt)
            checkpoint = torch.load(self.train_cfg.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)
            # self.load_from_checkpoint(self.train_cfg.ckpt)

        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []

        self.scheduler = DDPMScheduler(num_train_timesteps=model_cfg.timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
        self.psnr = PSNRMetric(max_val=1.0)
        self.ssim = SSIMMetric(spatial_dims=3, data_range=1.0)
        self.psnr_outputs = []
        self.ssim_outputs = []
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def correct_window(self,
        T_old, 
        a_min=-1024, 
        a_max=3071, 
        b_min=-512, #-100, #-512, 
        b_max=3071, #+900, #3071, #1536, #2560, 
        factor=1.0    
    ):
        # Calculate the range for the old and new scales
        range_old = a_max - a_min
        range_new = b_max - b_min

        # Reverse the incorrect scaling
        T_raw = (T_old * range_old) + a_min

        # Apply the correct scaling
        T_new = (T_raw - b_min) / range_new 
        return T_new.clamp(0, 1)
    
    def forward_screen(self, image3d, cameras, learnable_windowing=False):
        image3d = self.correct_window(image3d, a_min=-1024, a_max=3071, b_min=-512, b_max=3071)
        image2d = self.fwd_renderer(image3d, cameras)
        return image2d
    
    def flatten_cameras(self, cameras, zero_translation=False):
        camera_ = cameras.clone()
        R = camera_.R
        if zero_translation:
            T = torch.zeros_like(camera_.T.unsqueeze_(-1))
        else:
            T = camera_.T.unsqueeze_(-1)
        return torch.cat([R.reshape(-1, 1, 9), T.reshape(-1, 1, 3)], dim=-1).contiguous().view(-1, 1, 12)

    def forward_volume(self, image2d, camfeat, cameras, is_training=False):
        _device = image2d.device
        B = image2d.shape[0]

        image2d = torch.flip(image2d, dims=(-1,))
        # image2d = torch.rot90(image2d, 2, [2, 3])
        if is_training:
            timesteps = 1*torch.randint(0, 1000, (B,), device=_device).long() 
        else:
            timesteps = 0*torch.randint(0, 1000, (B,), device=_device).long() 
        noise = torch.randn_like(image2d)
        vol = self.inferer(
            inputs=image2d, 
            diffusion_model=self.unet2d_model, 
            noise=noise, 
            timesteps=timesteps, 
            condition=camfeat
        )
        # vol = self.unet2d_model.forward(image2d, timesteps, camfeat)

        detcams = cameras.clone()
        R = detcams.R
        # T = detcams.T.unsqueeze_(-1)
        T = torch.zeros_like(detcams.T.unsqueeze_(-1))
        inv = torch.cat([torch.inverse(R), -T], dim=-1)
        fwd = torch.cat([R, -T], dim=-1)

        org = vol[:, -2*self.model_cfg.vol_shape:-1*self.model_cfg.vol_shape,...].view(-1,1,self.model_cfg.vol_shape,self.model_cfg.vol_shape,self.model_cfg.vol_shape)
        grd = F.affine_grid(inv, org.size()).type(image2d.dtype)
        rot = F.grid_sample(vol[:, -1*self.model_cfg.vol_shape:,...].view(-1,1,self.model_cfg.vol_shape,self.model_cfg.vol_shape,self.model_cfg.vol_shape), grd)
        
        # Resample the frustum out
        z = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        y = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        x = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        grd = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(image2d.shape[0], 1, 1)  # 1 DHW 3 to B DHW 3
        
        # Process (resample) the volumes from ray views to ndc
        pts = cameras.transform_points_ndc(grd)  # world to ndc, 1 DHW 3
        fov = F.grid_sample(
            vol[:, :self.model_cfg.fov_depth,...].float().view(-1,1,self.model_cfg.fov_depth,self.model_cfg.vol_shape,self.model_cfg.vol_shape), 
            pts.view(-1, self.model_cfg.vol_shape, self.model_cfg.vol_shape, self.model_cfg.vol_shape, 3).float(), 
            mode="bilinear", 
            padding_mode="zeros", 
            align_corners=True,
        ) 

        out = (rot + org + 2*fov) / 4.0
        # out = torch.permute(out, [0, 1, 4, 3, 2])
        # out = torch.flip(out, dims=(-2,))

        # # # Resample the frustum out
        # z = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        # y = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        # x = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        # grd = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(image2d.shape[0], 1, 1)  # 1 DHW 3 to B DHW 3
        
        # # Process (resample) the volumes from ray views to ndc
        # pts = cameras.transform_points_ndc(grd)  # world to ndc, 1 DHW 3
        # fov = F.grid_sample(
        #     out[:, :self.model_cfg.fov_depth,...].float().view(-1,1,self.model_cfg.fov_depth,self.model_cfg.vol_shape,self.model_cfg.vol_shape), 
        #     pts.view(-1, self.model_cfg.vol_shape, self.model_cfg.vol_shape, self.model_cfg.vol_shape, 3).float(), 
        #     mode="bilinear", 
        #     padding_mode="zeros", 
        #     align_corners=True,
        # ) 
        # out = fov 

        out = torch.permute(out, [0, 1, 4, 3, 2])
        out = torch.flip(out, dims=(-2,))
        if self.unet3d_model is not None:
            # out = self.unet3d_model.forward(out, timesteps, camfeat) + out
            out = self.unet3d_model.forward(out) + out
            return out
        else:
            return out
    
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_optim, d_optim = self.optimizers()

        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        like_number = torch.ones(B, device=_device)
        
        dist_hidden = 8 * torch.ones(B, device=_device) # ndc
        elev_hidden = torch.zeros(B, device=_device)    # deg
        azim_hidden = torch.zeros(B, device=_device)    # deg
        degs_hidden = torch.full(like_number.shape, self.model_cfg.fov, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=degs_hidden, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)
        conf_hidden = torch.cat([dist_hidden.unsqueeze(-1).unsqueeze(-1), 
                                 elev_hidden.unsqueeze(-1).unsqueeze(-1), 
                                 azim_hidden.unsqueeze(-1).unsqueeze(-1), 
                                 degs_hidden.unsqueeze(-1).unsqueeze(-1)], dim=2)

        dist_random = 8 * torch.ones(B, device=_device) 
        if stage=="train":
            dist_random += torch.rand_like(dist_random) / 4.0 - 0.125      # ndc
        elev_random = torch.zeros(B, device=_device)                  # deg
        azim_random = torch.randint_like(like_number, -179, 180)      # deg 
        degs_random = torch.full(like_number.shape, self.model_cfg.fov, device=_device) 
        if stage=="train":
            degs_random += torch.randint_like(degs_random, -2, 3)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=degs_random, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)
        conf_random = torch.cat([dist_random.unsqueeze(-1).unsqueeze(-1), 
                                 elev_random.unsqueeze(-1).unsqueeze(-1), 
                                 azim_random.unsqueeze(-1).unsqueeze(-1), 
                                 degs_random.unsqueeze(-1).unsqueeze(-1)], dim=2)
        
        # Construct the samples in 2D
        figure_xr_source_hidden = image2d
        figure_ct_source_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        figure_ct_source_random = self.forward_screen(image3d=image3d, cameras=view_random)
            
        # Run the forward pass
        figure_dx_source_concat = torch.cat([figure_xr_source_hidden, figure_ct_source_hidden, figure_ct_source_random])
        camera_dx_render_concat = join_cameras_as_batch([view_hidden, view_hidden, view_random])
        config_dx_render_concat = torch.cat([conf_hidden, conf_hidden, conf_random], dim=0)

        # For 3D
        volume_dx_reproj_concat = self.forward_volume(
            image2d=figure_dx_source_concat, 
            camfeat=config_dx_render_concat, 
            cameras=camera_dx_render_concat,
            is_training=(stage=="train")
        )
        volume_xr_reproj_hidden, \
        volume_ct_reproj_hidden, \
        volume_ct_reproj_random = torch.split(volume_dx_reproj_concat, B, dim=0)

        figure_xr_reproj_hidden_hidden = self.forward_screen(image3d=volume_xr_reproj_hidden[:,[0],...], cameras=view_hidden)
        figure_xr_reproj_hidden_random = self.forward_screen(image3d=volume_xr_reproj_hidden[:,[0],...], cameras=view_random)
        
        figure_ct_reproj_hidden_hidden = self.forward_screen(image3d=volume_ct_reproj_hidden[:,[0],...], cameras=view_hidden)
        figure_ct_reproj_hidden_random = self.forward_screen(image3d=volume_ct_reproj_hidden[:,[0],...], cameras=view_random)
        
        figure_ct_reproj_random_hidden = self.forward_screen(image3d=volume_ct_reproj_random[:,[0],...], cameras=view_hidden)
        figure_ct_reproj_random_random = self.forward_screen(image3d=volume_ct_reproj_random[:,[0],...], cameras=view_random)
        
        im3d_loss_inv = F.l1_loss(volume_ct_reproj_hidden, image3d) * self.train_cfg.alpha \
                      + F.l1_loss(volume_ct_reproj_random, image3d) * self.train_cfg.gamma \

        im2d_loss_inv = F.l1_loss(figure_ct_reproj_hidden_hidden, figure_ct_source_hidden) * self.train_cfg.alpha \
                      + F.l1_loss(figure_ct_reproj_hidden_random, figure_ct_source_random) * self.train_cfg.gamma \
                      + F.l1_loss(figure_ct_reproj_random_hidden, figure_ct_source_hidden) * self.train_cfg.gamma \
                      + F.l1_loss(figure_ct_reproj_random_random, figure_ct_source_random) * self.train_cfg.alpha \

        r_loss = im2d_loss_inv + im3d_loss_inv  
        
        if self.perc25d_loss is not None:
            perc25d_loss = self.perc25d_loss(volume_ct_reproj_hidden, image3d) \
                         + self.perc25d_loss(volume_ct_reproj_random, image3d) 
            r_loss += self.train_cfg.lamda * perc25d_loss

        if self.perc30d_loss is not None:
            perc30d_loss = self.perc30d_loss(volume_ct_reproj_hidden, image3d) \
                         + self.perc30d_loss(volume_ct_reproj_random, image3d) 
            r_loss += self.train_cfg.lamda * perc30d_loss

        # self.log(f"{stage}_loss", loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
        # loss = torch.nan_to_num(loss, nan=1.0) 
        
        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                zeros = torch.zeros_like(image2d)
                
                viz2d = torch.cat([
                    torch.cat([
                        image3d[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_source_hidden, 
                        figure_ct_source_random, 
                        volume_ct_reproj_hidden[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_reproj_hidden_hidden, 
                        figure_ct_reproj_hidden_random, 
                    ], dim=-1),
                    torch.cat([
                        image3d[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_source_hidden, 
                        figure_ct_source_random, 
                        volume_ct_reproj_random[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_reproj_random_hidden, 
                        figure_ct_reproj_random_random, 
                    ], dim=-1),
                    torch.cat([
                        zeros,
                        figure_xr_source_hidden, 
                        zeros,
                        volume_xr_reproj_hidden[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_xr_reproj_hidden_hidden, 
                        figure_xr_reproj_hidden_random,
                    ], dim=-1)
                ], dim=-2)

                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(0, 1)
                tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx)    

        if stage=="train":
            # Generator
            fake_logits = self.p_disc_model.forward(torch.cat([volume_ct_reproj_hidden, volume_ct_reproj_hidden]).contiguous().float())[-1] 
            g_fake_loss = self.adv_loss(fake_logits, target_is_real=True, for_discriminator=False)
            g_loss = self.train_cfg.delta * g_fake_loss + r_loss
            self.log(f"{stage}_g_fake_loss", g_fake_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
            g_optim.zero_grad()
            self.manual_backward(g_loss)
            g_optim.step()

            # Discriminator
            fake_logits = self.p_disc_model.forward(torch.cat([volume_ct_reproj_hidden, volume_ct_reproj_hidden]).contiguous().detach())[-1] 
            real_logits = self.p_disc_model.forward(torch.cat([image3d, image3d]).contiguous().detach())[-1] 
            d_fake_loss = self.adv_loss(fake_logits, target_is_real=False, for_discriminator=True)
            d_real_loss = self.adv_loss(real_logits, target_is_real=True, for_discriminator=True)
            d_loss = self.train_cfg.delta * (d_real_loss + d_fake_loss) / 2
            self.log(f"{stage}_d_fake_loss", d_fake_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
            self.log(f"{stage}_d_real_loss", d_real_loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
            
            d_optim.zero_grad()
            self.manual_backward(d_loss)
            d_optim.step()
            loss = r_loss
        else:
            loss = r_loss       
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B) 
        return loss
                        
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_hidden = 8 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        # Construct the samples in 2D
        figure_xr_source_hidden = image2d
        figure_ct_source_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)

        # Reconstruct the Encoder-Decoder
        volume_ct_reproj_hidden = self.forward_volume(image2d=figure_ct_source_hidden, cameras=view_hidden)
        psnr = self.psnr(volume_ct_reproj_hidden, image3d)
        ssim = self.ssim(volume_ct_reproj_hidden, image3d)
        self.psnr_outputs.append(psnr)
        self.ssim_outputs.append(ssim)
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(
            f"train_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        print(f"PSNR :{torch.stack(self.psnr_outputs).mean()}")
        print(f"SSIM :{torch.stack(self.ssim_outputs).mean()}")
        self.psnr_outputs.clear()
        self.ssim_outputs.clear()

    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            [
                {'params': self.unet2d_model.parameters()},
            ], lr=1*self.train_cfg.lr, betas=(0.5, 0.999)
        )

        optimizer_d = torch.optim.AdamW(
            [
                {'params': self.p_disc_model.parameters()},
            ], lr=1*self.train_cfg.lr, betas=(0.5, 0.999)
        )
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_g, #
            milestones=[100, 200, 300, 400], 
            gamma=0.5
        )
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_d, #
            milestones=[100, 200, 300, 400], 
            gamma=0.5
        )
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
