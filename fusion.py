import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
import plotly.graph_objects as go
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import mcubes
import trimesh
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
from scipy.spatial.transform import Rotation as R

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from XMem.model.network import XMem
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.inference.inference_core import InferenceCore
from XMem.dataset.range_transform import im_normalization
from utils.grounded_sam import grounded_instance_sam_bacth_queries_np
from utils.draw_utils import draw_keypoints, aggr_point_cloud_from_data
from utils.corr_utils import compute_similarity_tensor, compute_similarity_tensor_multi
from utils.my_utils import  depth2fgpcd, fps_np, find_indices, depth2normal
from network.dense_correspondence_network import DenseCorrespondenceNetwork
torch.cuda.set_device(0)

class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0] #Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps

def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
        depth:          [rfn,pn,1]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=torch.float32)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=torch.float32)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return: feats_inter: b,n,f
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return feats_inter

def create_init_grid(boundaries, step_size):
    x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
    y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
    z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']
    x = torch.arange(x_lower, x_upper, step_size, dtype=torch.float32) + step_size / 2
    y = torch.arange(y_lower, y_upper, step_size, dtype=torch.float32) + step_size / 2
    z = torch.arange(z_lower, z_upper, step_size, dtype=torch.float32) + step_size / 2
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords, xx.shape


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def vis_tracking_pts(database, match_pts, sel_time):
    img_id = database.get_img_ids()[2]
    
    img = database.get_image(img_id, sel_time)[..., ::-1]
    
    color_cands = [(31,119,180), # in BGR
                (255,127,14),
                (44,160,44),
                (214,39,40),
                (148,103,189),]
    
    N = match_pts.shape[0]
    colors = color_cands[:N]
    
    Ks = database.get_K(img_id)
    pose = database.get_pose(img_id)
    
    fx = Ks[0, 0]
    fy = Ks[1, 1]
    cx = Ks[0, 2]
    cy = Ks[1, 2]
    
    match_pts = np.concatenate([match_pts, np.ones([N, 1])], axis=-1) # [N, 4]
    match_pts = np.matmul(pose, match_pts.T)[:3].T # [N, 3]
    
    match_pts_2d = match_pts[:, :2] / match_pts[:, 2:] # [N, 2]
    match_pts_2d[:, 0] = match_pts_2d[:, 0] * fx + cx
    match_pts_2d[:, 1] = match_pts_2d[:, 1] * fy + cy
    
    match_pts_2d = match_pts_2d.astype(np.int32)
    img = draw_keypoints(img, match_pts_2d, colors, radius=5)
    
    return img

def vis_tracking_multimodal_pts(database, match_pts_list, conf_list, sel_time, mask, view_idx = 0):
    # :param match_pts_list: list of [num_pts, 3]
    # :mask: [num_view, H, W, NQ] numpy array
    img_id = database.get_img_ids()[view_idx]
    
    img = database.get_image(img_id, sel_time)[..., ::-1]
    
    color_cands = [(31,119,180), # in BGR
                (255,127,14),
                (44,160,44),
                (214,39,40),
                (148,103,189),]
    
    Ks = database.get_K(img_id)
    pose = database.get_pose(img_id)
    
    fx = Ks[0, 0]
    fy = Ks[1, 1]
    cx = Ks[0, 2]
    cy = Ks[1, 2]
    
    for i, match_pts in enumerate(match_pts_list):
        # topk = min(5,match_pts.shape[0])
        # conf = conf_list[i]
        # topk_conf_idx = np.argpartition(conf, -topk)[-topk:]
        num_pts = match_pts.shape[0]
        # colors = color_cands[:num_pts]
        cmap = cm.get_cmap('viridis')
        colors = (cmap(np.linspace(0, 1, num_pts))[:, :3] * 255).astype(np.int32)[::-1, ::-1]
        match_pts = np.concatenate([match_pts, np.ones([num_pts, 1])], axis=-1) # [num_pts, 4]
        match_pts = np.matmul(pose, match_pts.T)[:3].T # [num_pts, 3]
        
        match_pts_2d = match_pts[:, :2] / match_pts[:, 2:] # [num_pts, 2]
        match_pts_2d[:, 0] = match_pts_2d[:, 0] * fx + cx
        match_pts_2d[:, 1] = match_pts_2d[:, 1] * fy + cy
        
        match_pts_2d = match_pts_2d.astype(np.int32)
        match_pts_2d = match_pts_2d.reshape(num_pts, 2)
        # img = draw_keypoints(img, match_pts_2d[topk_conf_idx], colors[topk_conf_idx], radius=5)
        img = draw_keypoints(img, match_pts_2d, colors, radius=5)
    
    # visualize the mask
    # mask = onehot2instance(mask) # [num_view, H, W]
    # mask = mask / mask.max() # [num_view, H, W]
    # num_view, H, W = mask.shape
    # cmap = cm.get_cmap('jet')
    # mask_vis = cmap(mask.reshape(-1)).reshape(num_view, H, W, 4)[..., :3] # [num_view, H, W, 3]
    # mask_vis = (mask_vis * 255).astype(np.uint8)
    # mask_vis = mask_vis[view_idx]
    
    # img = cv2.addWeighted(img, 0.5, mask_vis, 0.5, 0)
    
    return img

def octree_subsample(sim_vol, que_pts, last_res, topK):
    # :param sim_vol: [nd, N] # larger means more similar
    # :param que_pts: [nd, 3]
    # :param last_res: float
    # :param topK: float
    # :return child_que_pts = que_pts[alpha_mask] # [n_pts * 8, 3]
    assert sim_vol.shape[0] == que_pts.shape[0]
    sim_vol_topk, sim_vol_topk_idx = torch.topk(sim_vol, topK, dim=0, largest=True, sorted=False) # [topK, N]
    sim_vol_topk_idx = sim_vol_topk_idx.reshape(-1) # [topK*N]
    sim_vol_topk_idx = torch.unique(sim_vol_topk_idx) # [n_pts], n_pts <= topK*N
    sel_que_pts = que_pts[sim_vol_topk_idx] # [n_pts, 3]
    curr_res = last_res / 2
    
    child_offsets = torch.tensor([[curr_res, curr_res, curr_res],
                                  [curr_res, curr_res, -curr_res],
                                  [curr_res, -curr_res, curr_res],
                                  [curr_res, -curr_res, -curr_res],
                                  [-curr_res, curr_res, curr_res],
                                  [-curr_res, curr_res, -curr_res],
                                  [-curr_res, -curr_res, curr_res],
                                  [-curr_res, -curr_res, -curr_res]], dtype=torch.float32, device=que_pts.device) # [8, 3]
    child_que_pts = [sel_que_pts + child_offsets[i] for i in range(8)] # [n_pts * 8, 3]
    child_que_pts = torch.cat(child_que_pts, dim=0) # [n_pts * 8, 3]
    return child_que_pts, curr_res

def extract_kypts_gpu(sim_vol, que_pts, match_metric='sum'):
    # :param sim_vol: [n_pts, N] numpy array
    # :param que_pts: [n_pts, 3] numpy array
    # :return: [N, 3] numpy array
    N = sim_vol.shape[1]
    if type(sim_vol) is not torch.Tensor:
        sim_vol_tensor = torch.from_numpy(sim_vol).to("cuda:0") # [n_pts, N]
        que_pts_tensor = torch.from_numpy(que_pts).to("cuda:0") # [n_pts, 3]
    else:
        sim_vol_tensor = sim_vol
        que_pts_tensor = que_pts
    if match_metric == 'max':
        raise NotImplementedError
    elif match_metric == 'sum':
        # scale = 0.05
        # sim_vol_tensor = torch.exp(-sim_vol_tensor*scale)
        # match_pts_tensor = torch.zeros([N, 3]).cuda() # [N, 3]
        # for j in range(N):
        #     match_pts_tensor[j] = torch.sum(que_pts_tensor * sim_vol_tensor[:, j].unsqueeze(-1), dim=0) / torch.sum(sim_vol_tensor[:, j])
        
        # vectorized version
        match_pts_tensor = torch.sum(que_pts_tensor.unsqueeze(1) * sim_vol_tensor.unsqueeze(-1), dim=0) / torch.sum(sim_vol_tensor, dim=0).unsqueeze(-1) # [N, 3]
        # conf = sim_vol_tensor / torch.sum(sim_vol_tensor, dim=0).unsqueeze(0) # [n_pts, N]
        # conf = conf.max(dim=0)[0] # [N]
    return match_pts_tensor # , conf

def instance2onehot(instance, N = None):
    # :param instance: [**dim] numpy array uint8, val from 0 to N-1
    # :return: [**dim, N] numpy array bool
    if N is None:
        N = instance.max() + 1
    if type(instance) is np.ndarray:
        assert instance.dtype == np.uint8
        # assert instance.min() == 0
        H, W = instance.shape
        out = np.zeros(instance.shape + (N,), dtype=bool)
        for i in range(N):
            out[:, :, i] = (instance == i)
    elif type(instance) is torch.Tensor:
        assert instance.dtype == torch.uint8
        # assert instance.min() == 0
        out = torch.zeros(instance.shape + (N,), dtype=torch.bool, device=instance.device)
        for i in range(N):
            out[..., i] = (instance == i)
    return out

def onehot2instance(one_hot_mask):
    # :param one_hot_mask: [**dim, N] numpy array float32 or bool (probalistic or not)
    # :return: [**dim] numpy array uint8, val from 0 to N-1
    if type(one_hot_mask) == np.ndarray:
        return np.argmax(one_hot_mask, axis=-1).astype(np.uint8)
    elif type(one_hot_mask) == torch.Tensor:
        return torch.argmax(one_hot_mask, dim=-1).to(dtype=torch.uint8)
    else:
        raise NotImplementedError

class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2'):
        self.device = 'cuda:0'
        
        # hyper-parameters
        self.mu = 0.02
        
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        
        # dino feature extractor
        self.feat_backbone = feat_backbone
        if self.feat_backbone == 'dinov2':
            self.dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
        elif self.feat_backbone == 'dino':
            self.dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
            self.feat_extractor = ViTExtractor(model_type='dino_vitb8', stride=8, device=self.device)
        elif self.feat_backbone == 'don':
            model_folder = os.path.join(os.getcwd(),'ckpts','don')
            self.dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
            self.feat_extractor = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=os.path.join(model_folder,'003501.pth'))
            self.feat_extractor.eval()
        
        # load GroundedSAM model
        # config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
        config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py'
        # grounded_checkpoint = 'ckpts/groundingdino_swint_ogc.pth'  # change the path of the model
        grounded_checkpoint = 'ckpts/groundingdino_swinb_cogcoor.pth'
        sam_checkpoint = 'ckpts/sam_vit_h_4b8939.pth'
        self.ground_dino_model = load_model(config_file, grounded_checkpoint, device=self.device)

        self.sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        self.sam_model.model = self.sam_model.model.to(self.device)
        
        # load XMem model]
        xmem_config = {
            'model': 'XMem/saves/XMem.pth',
            'disable_long_term': False,
            'enable_long_term': True,
            'max_mid_term_frames': 10,
            'min_mid_term_frames': 5,
            'max_long_term_elements': 10000,
            'num_prototypes': 128,
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'save_scores': False,
            'size': 480,
            'key_dim': 64,
            'value_dim': 512,
            'hidden_dim': 64,
            'enable_long_term_count_usage': True,
        }
        
        network = XMem(xmem_config, xmem_config['model']).to(self.device).eval()
        model_weights = torch.load(xmem_config['model'])
        network.load_weights(model_weights, init_as_zero_if_needed=True)
        self.xmem_mapper = MaskMapper()
        self.xmem_processors = [InferenceCore(network, config=xmem_config) for _ in range(self.num_cam)]
        if xmem_config['size'] < 0:
            self.xmem_im_transform = T.Compose([
                T.ToTensor(),
                im_normalization,
            ])
            self.xmem_mask_transform = None
        else:
            self.xmem_im_transform = T.Compose([
                T.ToTensor(),
                im_normalization,
                T.Resize(xmem_config['size'], interpolation=T.InterpolationMode.BILINEAR),
            ])
            self.xmem_mask_transform = T.Compose([
                T.Resize(xmem_config['size'], interpolation=T.InterpolationMode.NEAREST),
            ])
        self.xmem_first_mask_loaded = False
        self.track_ids = [0]
        
    def eval(self, pts, return_names=['dino_feats', 'mask'], return_inter=False):
        # :param pts: (N, 3) torch tensor in world frame
        # :param return_names: a set of {'dino_feats', 'mask'}
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        #          - 'dino_feats': (N, f) torch tensor, the features of the points
        #          - 'mask': (N, NQ) torch tensor, the query masks of the points
        #          - 'valid_mask': (N) torch tensor, whether the point is valid
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        inter_normal = interpolate_feats(self.curr_obs_torch['normals'].permute(0,3,1,2),
                                         pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='bilinear') # [rfn,pn,3]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu) # [rfn,pn]
        
        # distance-based weight
        dist_weight = torch.exp(torch.clamp(self.mu-torch.abs(dist), max=0) / self.mu) # [rfn,pn]
        
        # # normal-based weight
        # fxfy = [torch.Tensor([self.curr_obs_torch['K'][i,0,0].item(), self.curr_obs_torch['K'][i,1,1].item()]) for i in range(self.num_cam)] # [rfn, 2]
        # fxfy = torch.stack(fxfy, dim=0).to(self.device) # [rfn, 2]
        # view_dir = pts_2d / fxfy[:, None, :] # [rfn,pn,2]
        # view_dir = torch.cat([view_dir, torch.ones_like(view_dir[...,0:1])], dim=-1) # [rfn,pn,3]
        # view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True) # [rfn,pn,3]
        # dist_weight = torch.abs(torch.sum(view_dir * inter_normal, dim=-1)) # [rfn,pn]
        # dist_weight = dist_weight * dist_valid.float() # [rfn,pn]
        
        dist = torch.clamp(dist, min=-self.mu, max=self.mu) # [rfn,pn]
        
        # # weighted distance
        # dist = (dist * dist_weight).sum(0) / (dist_weight.sum(0) + 1e-6) # [pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        dist[dist_all_invalid] = 1e3
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        for k in return_names:
            inter_k = interpolate_feats(self.curr_obs_torch[k].permute(0,3,1,2),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='bilinear') # [rfn,pn,k_dim]
            
            # weighted sum
            # val = (inter_k * dist_weight.unsqueeze(-1)).sum(0) / (dist_weight.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            
            # # valid-weighted sum
            val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            val[dist_all_invalid] = 0.0
            
            outputs[k] = val
            if return_inter:
                outputs[k+'_inter'] = inter_k
            else:
                del inter_k
        
        return outputs

    def eval_dist(self, pts):
        # this version does not clamp the distance or change the invalid points to 1e3
        # this is for grasper planner to find the grasping pose that does not penalize the depth
        # :param pts: (N, 3) torch tensor in world frame
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask # [rfn,pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        return outputs
        
        # if 'dino_feats' in return_names:
        #     inter_feats = interpolate_feats(self.curr_obs_torch['dino_feats'].permute(0,3,1,2),
        #                                     pts_2d,
        #                                     h = self.H,
        #                                     w = self.W,
        #                                     padding_mode='zeros',
        #                                     align_corners=True,
        #                                     inter_mode='bilinear') # [rfn,pn,feat_dim]
        # else:
        #     inter_feats = None
        
        # if 'mask_shoe' in self.curr_obs_torch and 'mask_shoe' in return_names:
        #     inter_masks = interpolate_feats(self.curr_obs_torch['mask_shoe'].permute(0,3,1,2),
        #                                     pts_2d,
        #                                     h = self.H,
        #                                     w = self.W,
        #                                     padding_mode='zeros',
        #                                     align_corners=True,
        #                                     inter_mode='nearest') # [rfn,pn,nq]
        # else:
        #     inter_masks = None
        
        # # compute the features of the points
        # if 'dino_feats' in return_names:
        #     features = (inter_feats * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,feat_dim]
        #     # features = (inter_feats * dist_valid.float().unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,feat_dim]
        #     features[dist_all_invalid] = 0.0
        # else:
        #     features = None
        
        # # compute the query masks of the points
        # if inter_masks is not None and 'mask_shoe' in return_names:
        #     query_masks = (inter_masks * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,nq]
        #     # query_masks = (inter_masks * dist_valid.float().unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,nq]
        #     query_masks[dist_all_invalid] = 0.0
        # else:
        #     query_masks = None
        
        # return {'dist': dist,
        #         'dino_feats': features,
        #         'mask_shoe': query_masks,
        #         'valid_mask': ~dist_all_invalid}
    
    def batch_eval(self, pts, return_names=['dino_feats', 'mask']):
        batch_pts = 60000
        outputs = {}
        for i in tqdm(range(0, pts.shape[0], batch_pts)):
            st_idx = i
            ed_idx = min(i + batch_pts, pts.shape[0])
            out = self.eval(pts[st_idx:ed_idx], return_names=return_names)
            for k in out:
                if k not in outputs:
                    outputs[k] = [out[k]]
                else:
                    outputs[k].append(out[k])
        
        # concat the outputs
        for k in outputs:
            if outputs[k][0] is not None:
                outputs[k] = torch.cat(outputs[k], dim=0)
            else:
                outputs[k] = None
        return outputs
    
    # def extract_dist_vol(self, boundaries):
    #     step = 0.002
    #     init_grid, grid_shape = create_init_grid(boundaries, step)
    #     init_grid = init_grid.to(self.device, dtype=torch.float32)
        
    #     batch_pts = 10000
        
    #     dist_vol = torch.zeros(init_grid.shape[0], dtype=torch.float32, device=self.device)
    #     valid_mask = torch.zeros(init_grid.shape[0], dtype=torch.bool, device=self.device)
        
    #     for i in range(0, init_grid.shape[0], batch_pts):
    #         st_idx = i
    #         ed_idx = min(i + batch_pts, init_grid.shape[0])
    #         out = self.eval(init_grid[st_idx:ed_idx], return_names={})
            
    #         dist_vol[st_idx:ed_idx] = out['dist']
    #         valid_mask[st_idx:ed_idx] = out['valid_mask']
    #     return {'init_grid': init_grid,
    #             'grid_shape': grid_shape,
    #             'dist': dist_vol,
    #             'valid_mask': valid_mask,}

    def extract_dinov2_features(self, imgs, params):
        K, H, W, _ = imgs.shape
        
        patch_h = params['patch_h']
        patch_w = params['patch_w']
        # feat_dim = 384 # vits14
        # feat_dim = 768 # vitb14
        feat_dim = 1024 # vitl14
        # feat_dim = 1536 # vitg14
        
        transform = T.Compose([
            # T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=self.device)
        for j in range(K):
            img = Image.fromarray(imgs[j])
            imgs_tensor[j] = transform(img)[:3]
        with torch.no_grad():
            features_dict = self.dinov2_feat_extractor.forward_features(imgs_tensor)
            features = features_dict['x_norm_patchtokens']
            features = features.reshape((K, patch_h, patch_w, feat_dim))
        return features

    def extract_features(self, imgs, params):
        # :param imgs (K, H, W, 3) np array, color images
        # :param params: dict contains:
        #                - 'patch_h', 'patch_w': int, the size of the patch
        # :return features: (K, patch_h, patch_w, feat_dim) np array, features of the images
        if self.feat_backbone == 'dinov2':
            return self.extract_dinov2_features(imgs, params)
        elif self.feat_backbone == 'dino':
            K, H, W, _ = imgs.shape
            
            transform = T.Compose([
                # T.GaussianBlur(9, sigma=(0.1, 2.0)),
                T.Resize((H, W)),
                T.CenterCrop((H, W)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            imgs_tensor = torch.zeros((K, 3, H, W), device=self.device)
            for j in range(K):
                img = Image.fromarray(imgs[j])
                imgs_tensor[j] = transform(img)[:3]
            with torch.no_grad():
                features = self.feat_extractor.extract_descriptors(imgs_tensor)
            return features
        elif self.feat_backbone == 'don':
            K, H, W, _ = imgs.shape
            
            transform = T.Compose([
                # T.GaussianBlur(9, sigma=(0.1, 2.0)),
                # T.Resize((H, W)),
                # T.CenterCrop((H, W)),
                T.ToTensor(),
                # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.Normalize(mean=(0.5573105812072754, 0.37420374155044556, 0.37020164728164673), std=(0.24336038529872894, 0.2987397611141205, 0.31875079870224)),
            ])
            
            imgs_tensor = torch.zeros((K, 3, H, W), device=self.device)
            for j in range(K):
                img = Image.fromarray(imgs[j])
                imgs_tensor[j] = transform(img)[:3]
            with torch.no_grad():
                # features = self.feat_extractor(torch.flip(imgs_tensor, dims=[1]))
                features = []
                for i in range(K):
                    # features.append(self.feat_extractor.forward_on_img(img))
                    features.append(self.feat_extractor.forward_single_image_tensor(imgs_tensor[i]))
                features = torch.stack(features, dim=0)
                # norm = torch.norm(features, 2, 1, keepdim=True) # [N,1,H,W]
                # features = features/norm
            # feature = features.detach().cpu().numpy()
            # feature_map = feature[0, 2, :, :]
            # plt.imshow(feature_map)
            # plt.show()
            # reshape to (K, patch_h, patch_w, feat_dim)
            # for i in range(K):
            #     for j in range(3):
            #         features[i, j] = (features[i, j] - features[i, j].min()) / (features[i, j].max() - features[i, j].min())
            # features = features.permute(0, 2, 3, 1)
            return features
        else:
            raise NotImplementedError
    
    def xmem_process(self, rgb, mask):
        # track the mask using XMem
        # :param: rgb: (K, H, W, 3) np array, color image
        # :param: mask: None or (K, H, W) torch tensor, mask
        # return: out_masks: (K, H, W) torch tensor, mask
        # rgb_tensor = torch.zeros((self.num_cam, 3, self.H, self.W), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            rgb_tensor = []
            for i in range(self.num_cam):
                rgb_tensor.append(self.xmem_im_transform(rgb[i]).to(self.device, dtype=torch.float32))
            rgb_tensor = torch.stack(rgb_tensor, dim=0)
            if self.xmem_mask_transform is not None and mask is not None:
                mask = self.xmem_mask_transform(mask).to(self.device, dtype=torch.float32)
            
            if mask is not None and not self.xmem_first_mask_loaded:
                # converted_masks = []
                for i in range(self.num_cam):
                    _, labels = self.xmem_mapper.convert_mask(mask[i].cpu().numpy(), exhaustive=True)
                    # converted_masks.append(converted_mask)
                converted_masks = [self.xmem_mapper.convert_mask(mask[i].cpu().numpy(), exhaustive=True)[0] for i in range(self.num_cam)]
                # # assume that labels for all views are the same
                # for labels in labels_list:
                #     assert labels == labels_list[0]
                converted_masks = torch.from_numpy(np.stack(converted_masks, axis=0)).to(self.device, dtype=torch.float32)
                for processor in self.xmem_processors:
                    processor.set_all_labels(list(self.xmem_mapper.remappings.values()))
                self.track_ids = [0,] + list(self.xmem_mapper.remappings.values())
            elif mask is not None and self.xmem_first_mask_loaded:
                converted_masks = instance2onehot(mask.to(torch.uint8), len(self.track_ids))
                converted_masks = converted_masks.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
                converted_masks = converted_masks[:, 1:] # remove the background
            
            if not self.xmem_first_mask_loaded:
                if mask is not None:
                    self.xmem_first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    raise ValueError('No mask provided for the first frame')
            
            out_masks = torch.zeros((self.num_cam, self.H, self.W)).to(self.device, dtype=torch.uint8)
            for view_i, processor in enumerate(self.xmem_processors):
                prob = processor.step(rgb_tensor[view_i],
                                      converted_masks[view_i] if mask is not None else None,
                                      self.track_ids[1:] if mask is not None else None,
                                      end=False)
                prob = F.interpolate(prob.unsqueeze(1), (self.H, self.W), mode='bilinear', align_corners=False)[:,0]
                
                out_mask = torch.argmax(prob, dim=0)
                out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
                out_mask = self.xmem_mapper.remap_index_mask(out_mask)
                # out_mask = instance2onehot(out_mask)
                out_masks[view_i] = torch.from_numpy(out_mask).to(self.device, dtype=torch.uint8)
            out_masks = instance2onehot(out_masks, len(self.track_ids))
        return out_masks.to(self.device, dtype=torch.float32)
    
    def update(self, obs):
        # :param obs: dict contains:
        #             - 'color': (K, H, W, 3) np array, color image
        #             - 'depth': (K, H, W) np array, depth image
        #             - 'pose': (K, 4, 4) np array, camera pose
        #             - 'K': (K, 3, 3) np array, camera intrinsics
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {
            'patch_h': color.shape[1] // 14,
            'patch_w': color.shape[2] // 14,
        }
        features = self.extract_features(color, params)
        
        # self.curr_obs_torch = {
        #     'dino_feats': features,
        #     'color': color,
        #     'depth': torch.from_numpy(obs['depth']).to(self.device, dtype=torch.float32),
        #     'pose': torch.from_numpy(obs['pose']).to(self.device, dtype=torch.float32),
        #     'K': torch.from_numpy(obs['K']).to(self.device, dtype=torch.float32),
        # }
        self.curr_obs_torch['dino_feats'] = features
        if self.feat_backbone == 'dinov2':
            self.curr_obs_torch['dino_v2_feats'] = features
        else:
            self.curr_obs_torch['dino_v2_feats'] = self.extract_dinov2_features(color, params)
        self.curr_obs_torch['color'] = color
        normals_np = [depth2normal(obs['depth'][i], obs['K'][i]) for i in range(self.num_cam)]
        normals_np = np.stack(normals_np, axis=0)
        self.curr_obs_torch['normals'] = torch.from_numpy(normals_np).to(self.device, dtype=torch.float32)
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=torch.float32) / 255.0
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).to(self.device, dtype=torch.float32)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=torch.float32)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=torch.float32)
        
        _, self.H, self.W = obs['depth'].shape
    
    # DEPRECATED
    # def text_query(self, queries, thresholds):
    #     assert len(queries) == len(thresholds)
    #     if 'color' not in self.curr_obs_torch:
    #         print('Please call update() first!')
    #         exit()
        
    #     K = self.curr_obs_torch['color'].shape[0]
    #     NQ = len(queries)
    #     query_masks = torch.zeros((K, self.H, self.W, NQ), device=self.device)
    #     for i in range(K):
    #         masks = grounded_sam_batch_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds)
    #         query_masks[i] = masks.permute(1,2,0)
    #     self.curr_obs_torch['mask'] = query_masks # [K, H, W, NQ]
    
    # DEPRECATED: it only works for one type of object
    # def align_instance_mask(self, query):
    #     pose_0 = self.curr_obs_torch['pose'][0]
    #     K_0 = self.curr_obs_torch['K'][0]
    #     depth_0 = self.curr_obs_torch['depth'][0]
    #     num_view = self.curr_obs_torch['color'].shape[0]
    #     mask_0 = self.curr_obs_torch[f'mask_{query}'][0]
    #     for i in range(1, num_view):
    #         mask_i = self.curr_obs_torch[f'mask_{query}'][i]
    #         num_instance = int(mask_i.max().item())
    #         pose_i = self.curr_obs_torch['pose'][i]
    #         pose_i = torch.cat([pose_i, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
    #         pose_i = torch.inverse(pose_i)
    #         K_i = self.curr_obs_torch['K'][i]
    #         depth_i = self.curr_obs_torch['depth'][i]
    #         camera_param_i = [K_i[0, 0].item(),
    #                           K_i[1, 1].item(),
    #                           K_i[0, 2].item(),
    #                           K_i[1, 2].item(),]
    #         new_mask_i = torch.zeros_like(mask_i).to(self.device, dtype=torch.uint8)
    #         for j in range(1, num_instance+1):
    #             mask_i_inst_j = (mask_i == j)
    #             pcd_i_np = depth2fgpcd(depth_i.cpu().numpy(), mask_i_inst_j.cpu().numpy(), camera_param_i)
    #             pcd_i = torch.from_numpy(pcd_i_np).to(self.device, dtype=torch.float32)
                
    #             # # numpy version for reference
    #             # pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    #             # pose = np.linalg.inv(pose)
    #             # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
                
    #             # torch version
    #             pcd_i = torch.cat([pcd_i.T, torch.ones((1, pcd_i.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
    #             pcd_i = (pose_i @ pcd_i)[:3].T
                
    #             pts_2d, valid_mask, pts_depth = project_points_coords(pcd_i, pose_0[None], K_0[None]) # [1,N,2], [1,N], [1,N,1]
    #             pts_depth = pts_depth[...,0] # [1,N]
                
    #             # get interpolated depth and features
    #             inter_depth = interpolate_feats(depth_0[None,None],
    #                                             pts_2d,
    #                                             h = self.H,
    #                                             w = self.W,
    #                                             padding_mode='zeros',
    #                                             align_corners=True,
    #                                             inter_mode='nearest')[...,0] # [1,N]
    #             num_vis_list = [] # count the visibility of each point on mask_0
    #             for k in range(1, num_instance+1):
    #                 mask_0_inst_k = (mask_0 == k)
    #                 inter_mask = interpolate_feats(mask_0_inst_k[None,None].to(torch.float32),
    #                                                pts_2d,
    #                                                h=self.H,
    #                                                w=self.W,
    #                                                padding_mode='zeros',
    #                                                align_corners=True,
    #                                                inter_mode='nearest')[...,0] # [1,N]
    #                 inter_mask = inter_mask > 0.5
                    
    #                 num_vis = ((torch.abs(pts_depth - inter_depth) < 0.03 ) & inter_mask).sum().item()
    #                 num_vis_list.append(num_vis)
    #             max_vis_arg = np.argmax(num_vis_list)
    #             new_mask_i[mask_i_inst_j] = max_vis_arg + 1
    #         self.curr_obs_torch[f'mask_{query}'][i] = new_mask_i
    
    # def align_instance_mask(self):
    #     # NOTE: assume all objects are visible in the first frame. In another word, labels in different views are identical.
    #     num_view = self.curr_obs_torch['color'].shape[0]
    #     for i in range(num_view):
    #         assert self.curr_obs_torch['mask_label'][0] == self.curr_obs_torch['mask_label'][i]
    #     pose_0 = self.curr_obs_torch['pose'][0]
    #     K_0 = self.curr_obs_torch['K'][0]
    #     depth_0 = self.curr_obs_torch['depth'][0]
    #     mask_0 = self.curr_obs_torch[f'mask'][0]
    #     label_0 = self.curr_obs_torch['mask_label'][0]
    #     for i in range(1, num_view):
    #         mask_i = self.curr_obs_torch[f'mask'][i]
    #         label_i = self.curr_obs_torch['mask_label'][i]
    #         num_instance = len(label_0) - 1
    #         pose_i = self.curr_obs_torch['pose'][i]
    #         pose_i = torch.cat([pose_i, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
    #         pose_i = torch.inverse(pose_i)
    #         K_i = self.curr_obs_torch['K'][i]
    #         depth_i = self.curr_obs_torch['depth'][i]
    #         camera_param_i = [K_i[0, 0].item(),
    #                           K_i[1, 1].item(),
    #                           K_i[0, 2].item(),
    #                           K_i[1, 2].item(),]
    #         new_mask_i = torch.zeros_like(mask_i).to(self.device, dtype=torch.uint8)
    #         for j in range(1, num_instance+1):
    #             mask_i_inst_j = (mask_i == j)
    #             pcd_i_np = depth2fgpcd(depth_i.cpu().numpy(), mask_i_inst_j.cpu().numpy(), camera_param_i)
    #             pcd_i = torch.from_numpy(pcd_i_np).to(self.device, dtype=torch.float32)
                
    #             # # numpy version for reference
    #             # pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    #             # pose = np.linalg.inv(pose)
    #             # trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
                
    #             # torch version
    #             pcd_i = torch.cat([pcd_i.T, torch.ones((1, pcd_i.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
    #             pcd_i = (pose_i @ pcd_i)[:3].T
                
    #             pts_2d, valid_mask, pts_depth = project_points_coords(pcd_i, pose_0[None], K_0[None]) # [1,N,2], [1,N], [1,N,1]
    #             pts_depth = pts_depth[...,0] # [1,N]
                
    #             # get interpolated depth and features
    #             inter_depth = interpolate_feats(depth_0[None,None],
    #                                             pts_2d,
    #                                             h = self.H,
    #                                             w = self.W,
    #                                             padding_mode='zeros',
    #                                             align_corners=True,
    #                                             inter_mode='nearest')[...,0] # [1,N]
    #             num_vis_list = [] # count the visibility of each point on mask_0
    #             for k in range(1, num_instance+1):
    #                 if label_i[j] != label_0[k]:
    #                     num_vis_list.append(0)
    #                     continue
    #                 mask_0_inst_k = (mask_0 == k)
    #                 inter_mask = interpolate_feats(mask_0_inst_k[None,None].to(torch.float32),
    #                                                pts_2d,
    #                                                h=self.H,
    #                                                w=self.W,
    #                                                padding_mode='zeros',
    #                                                align_corners=True,
    #                                                inter_mode='nearest')[...,0] # [1,N]
    #                 inter_mask = inter_mask > 0.5
                    
    #                 num_vis = ((torch.abs(pts_depth - inter_depth) < 0.03 ) & inter_mask).sum().item()
    #                 num_vis_list.append(num_vis)
    #             max_vis_arg = np.argmax(num_vis_list)
    #             new_mask_i[mask_i_inst_j] = max_vis_arg + 1
    #         self.curr_obs_torch[f'mask'][i] = new_mask_i
    
    def compute_alignment(self, v_i, v_j, inst_n, inst_m):
        # :param v_i, view index i
        # :param v_j, view index j
        # :param inst_n, instance index in view i
        # :param inst_m, instance index in view j
        # :return: visiblity score (# of visible points)
        
        # extract information from view i
        mask_i = self.curr_obs_torch['mask'][v_i] # [H,W]
        mask_i_inst_n = (mask_i == inst_n)
        depth_i = self.curr_obs_torch['depth'][v_i] # [H,W]
        pose_i = self.curr_obs_torch['pose'][v_i]
        pose_i_homo = torch.cat([pose_i, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        pose_i_inv = torch.inverse(pose_i_homo)
        K_i = self.curr_obs_torch['K'][v_i]
        depth_i = self.curr_obs_torch['depth'][v_i]
        camera_param_i = [K_i[0, 0].item(),
                            K_i[1, 1].item(),
                            K_i[0, 2].item(),
                            K_i[1, 2].item(),]
        
        pcd_i_np = depth2fgpcd(depth_i.cpu().numpy(), mask_i_inst_n.cpu().numpy(), camera_param_i)
        pcd_i = torch.from_numpy(pcd_i_np).to(self.device, dtype=torch.float32)
        pcd_i = torch.cat([pcd_i.T, torch.ones((1, pcd_i.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
        pcd_i = (pose_i_inv @ pcd_i)[:3].T
        
        # extract information from view j        
        mask_j = self.curr_obs_torch['mask'][v_j] # [H,W]
        mask_j_inst_m = (mask_j == inst_m)
        depth_j = self.curr_obs_torch['depth'][v_j] # [H,W]
        pose_j = self.curr_obs_torch['pose'][v_j]
        pose_j_homo = torch.cat([pose_j, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        pose_j_inv = torch.inverse(pose_j_homo)
        K_j = self.curr_obs_torch['K'][v_j]
        depth_j = self.curr_obs_torch['depth'][v_j]
        camera_param_j = [K_j[0, 0].item(),
                            K_j[1, 1].item(),
                            K_j[0, 2].item(),
                            K_j[1, 2].item(),]
        
        pcd_j_np = depth2fgpcd(depth_j.cpu().numpy(), mask_j_inst_m.cpu().numpy(), camera_param_j)
        pcd_j = torch.from_numpy(pcd_j_np).to(self.device, dtype=torch.float32)
        pcd_j = torch.cat([pcd_j.T, torch.ones((1, pcd_j.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
        pcd_j = (pose_j_inv @ pcd_j)[:3].T
        
        # project i in j
        pts_2d_i_in_j, valid_mask, pts_depth_i_in_j = project_points_coords(pcd_i, pose_j[None], K_j[None]) # [1,N,2], [1,N], [1,N,1]
        pts_depth_i_in_j = pts_depth_i_in_j[...,0] # [1,N]
        inter_depth_i_in_j = interpolate_feats(depth_j[None,None],
                                               pts_2d_i_in_j,
                                               h = self.H,
                                               w = self.W,
                                               padding_mode='zeros',
                                               align_corners=True,
                                               inter_mode='nearest')[...,0] # [1,N]
        
        inter_mask_i_in_j = interpolate_feats(mask_j_inst_m[None,None].to(torch.float32),
                                              pts_2d_i_in_j,
                                              h=self.H,
                                              w=self.W,
                                              padding_mode='zeros',
                                              align_corners=True,
                                              inter_mode='nearest')[...,0] # [1,N]
        inter_mask_i_in_j = inter_mask_i_in_j > 0.5
        
        # project j in i
        pts_2d_j_in_i, valid_mask, pts_depth_j_in_i = project_points_coords(pcd_j, pose_i[None], K_i[None]) # [1,N,2], [1,N], [1,N,1]
        pts_depth_j_in_i = pts_depth_j_in_i[...,0] # [1,N]
        inter_depth_j_in_i = interpolate_feats(depth_i[None,None],
                                               pts_2d_j_in_i,
                                               h = self.H,
                                               w = self.W,
                                               padding_mode='zeros',
                                               align_corners=True,
                                               inter_mode='nearest')[...,0]
        inter_mask_j_in_i = interpolate_feats(mask_i_inst_n[None,None].to(torch.float32),
                                                pts_2d_j_in_i,
                                                h=self.H,
                                                w=self.W,
                                                padding_mode='zeros',
                                                align_corners=True,
                                                inter_mode='nearest')[...,0] # [1,N]
        inter_mask_j_in_i = inter_mask_j_in_i > 0.5

        num_vis = ((torch.abs(pts_depth_i_in_j - inter_depth_i_in_j) < 0.03 ) & inter_mask_i_in_j).sum().item() +\
                    ((torch.abs(pts_depth_j_in_i - inter_depth_j_in_i) < 0.03 ) & inter_mask_j_in_i).sum().item()
        
        return num_vis
    
    def add_mask_in_j_use_i_inst_n(self, v_i, v_j, inst_n):
        # :param v_i, view index i with inst_n
        # :param v_j, view index j with missing mask
                # :param v_i, view index i
        # :param v_j, view index j
        # :param inst_n, instance index in view i
        # :param inst_m, instance index in view j
        # :return: visiblity score (# of visible points)
        
        # extract information from view i
        mask_i = self.curr_obs_torch['mask'][v_i] # [H,W]
        mask_i_inst_n = (mask_i == inst_n)
        depth_i = self.curr_obs_torch['depth'][v_i] # [H,W]
        pose_i = self.curr_obs_torch['pose'][v_i]
        pose_i_homo = torch.cat([pose_i, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        pose_i_inv = torch.inverse(pose_i_homo)
        K_i = self.curr_obs_torch['K'][v_i]
        depth_i = self.curr_obs_torch['depth'][v_i]
        camera_param_i = [K_i[0, 0].item(),
                            K_i[1, 1].item(),
                            K_i[0, 2].item(),
                            K_i[1, 2].item(),]
        
        mask_i_inst_n = mask_i_inst_n.cpu().numpy().astype(np.uint8)
        mask_i_inst_n = (cv2.erode(mask_i_inst_n * 255, np.ones([10, 10], np.uint8), iterations=1) / 255).astype(bool)
        pcd_i_np = depth2fgpcd(depth_i.cpu().numpy(), mask_i_inst_n, camera_param_i)
        pcd_i = torch.from_numpy(pcd_i_np).to(self.device, dtype=torch.float32)
        pcd_i = torch.cat([pcd_i.T, torch.ones((1, pcd_i.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
        pcd_i = (pose_i_inv @ pcd_i)[:3].T
        
        # extract information from view j        
        depth_j = self.curr_obs_torch['depth'][v_j] # [H,W]
        pose_j = self.curr_obs_torch['pose'][v_j]
        K_j = self.curr_obs_torch['K'][v_j]
        depth_j = self.curr_obs_torch['depth'][v_j]
        
        # project i in j
        pts_2d_i_in_j, valid_mask, pts_depth_i_in_j = project_points_coords(pcd_i, pose_j[None], K_j[None]) # [1,N,2], [1,N], [1,N,1]
        pts_depth_i_in_j = pts_depth_i_in_j[...,0] # [1,N]
        inter_depth_i_in_j = interpolate_feats(depth_j[None,None],
                                               pts_2d_i_in_j,
                                               h = self.H,
                                               w = self.W,
                                               padding_mode='zeros',
                                               align_corners=True,
                                               inter_mode='nearest')[...,0] # [1,N]
        pts_i_visible_in_j = ((torch.abs(pts_depth_i_in_j - inter_depth_i_in_j) < 0.02) & valid_mask)[0] # [N]
        
        pts_2d_i_in_j = pts_2d_i_in_j[0][pts_i_visible_in_j] # [M, 2]
        
        if pts_2d_i_in_j.shape[0] == 0:
            return None
        
        input_point = pts_2d_i_in_j.detach().cpu().numpy().astype(np.int32)
        input_label = np.ones((input_point.shape[0],), dtype=np.int32)

        self.sam_model.set_image(self.curr_obs_torch['color'][v_j])
        masks, _, _ = self.sam_model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        return masks[0]
        
    def compute_alignment_with_consensus(self, v_i, inst_n, consensus_mask_dict, consensus_i):
        curr_view_inst_dict = consensus_mask_dict[consensus_i]
        avg_align = []
        for curr_view, curr_inst in curr_view_inst_dict.items():
            if curr_view == v_i or curr_inst == -1:
                continue
            avg_align.append(self.compute_alignment(curr_view, v_i, curr_inst, inst_n))
        avg_align = np.mean(avg_align)
        return avg_align
    
    def update_with_missing_label(self, consensus_mask_label, consensus_mask_dict, v_i):
        len_i = len(self.curr_obs_torch['mask_label'][v_i])
        mask_label_i = self.curr_obs_torch['mask_label'][v_i]
        len_curr = len(consensus_mask_label)
        assert len_i < len_curr
        for label_i, label in enumerate(mask_label_i):
            align_ls = []
            for consensus_i, consensus_label in enumerate(consensus_mask_label):
                if label != consensus_label:
                    # skip when the label is not the same
                    align_ls.append(-1)
                    continue
                align_ls.append(self.compute_alignment_with_consensus(v_i, label_i, consensus_mask_dict, consensus_i))
            matched_consensus_i = np.argmax(align_ls)
            try:
                assert v_i not in consensus_mask_dict[matched_consensus_i]
            except:
                print(f'{v_i} is already in {consensus_mask_dict} at {matched_consensus_i}')
                raise AssertionError
            consensus_mask_dict[matched_consensus_i][v_i] = label_i
        # add the missing label
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            if v_i in consensus_mask_dict[consensus_i]:
                continue
            consensus_mask_dict[consensus_i][v_i] = -1
            
            # # add mask
            # existing_match = consensus_mask_dict[consensus_i]
            # successful_adding = False
            # for v_j, inst_j in existing_match.items():
            #     if inst_j == -1:
            #         continue
            #     add_mask = self.add_mask_in_j_use_i_inst_n(v_j, v_i, inst_j)
            #     if add_mask is None:
            #         continue
            #     else:
            #         successful_adding = True
            #         break
            # if not successful_adding:
            #     consensus_mask_dict[consensus_i][v_i] = -1
            # else:
            #     consensus_mask_dict[consensus_i][v_i] = len(self.curr_obs_torch['mask_label'][v_i])
            #     self.curr_obs_torch['mask'][v_i][add_mask > 0.5] = len(self.curr_obs_torch['mask_label'][v_i])
            #     self.curr_obs_torch['mask_label'][v_i].append(consensus_label)
            
        return consensus_mask_label, consensus_mask_dict
    
    def update_with_equal_label(self, consensus_mask_label, consensus_mask_dict, v_i):
        len_i = len(self.curr_obs_torch['mask_label'][v_i])
        mask_label_i = self.curr_obs_torch['mask_label'][v_i]
        len_curr = len(consensus_mask_label)
        assert len_i == len_curr
        for label_i, label in enumerate(mask_label_i):
            align_ls = []
            for consensus_i, consensus_label in enumerate(consensus_mask_label):
                if label != consensus_label:
                    # skip when the label is not the same
                    align_ls.append(-1)
                    continue
                align_ls.append(self.compute_alignment_with_consensus(v_i, label_i, consensus_mask_dict, consensus_i))
            matched_consensus_i = np.argmax(align_ls)
            try:
                assert v_i not in consensus_mask_dict[matched_consensus_i]
            except:
                print(f'{v_i} is already in {consensus_mask_dict} at {matched_consensus_i}')
                raise AssertionError
            consensus_mask_dict[matched_consensus_i][v_i] = label_i
        # assert whether all labels are matched
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            try:
                assert v_i in consensus_mask_dict[consensus_i]
            except:
                print(f'{v_i} is not in {consensus_mask_dict} at {consensus_i}')
                print(f'hint: curr view label {mask_label_i}')
                print(f'hint: curr consensus label {consensus_mask_label}')
                exit()
        return consensus_mask_label, consensus_mask_dict
    
    def update_with_additional_label(self, consensus_mask_label, consensus_mask_dict, v_i):
        len_i = len(self.curr_obs_torch['mask_label'][v_i])
        mask_label_i = self.curr_obs_torch['mask_label'][v_i]
        len_curr = len(consensus_mask_label)
        assert len_i > len_curr
        matched_label_idx_ls = []
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            align_ls = []
            for label_i, label in enumerate(mask_label_i):
                if label != consensus_label:
                    # skip when the label is not the same
                    align_ls.append(-1)
                    continue
                align_ls.append(self.compute_alignment_with_consensus(v_i, label_i, consensus_mask_dict, consensus_i))
            matched_label_i = np.argmax(align_ls)
            consensus_mask_dict[consensus_i][v_i] = matched_label_i
            matched_label_idx_ls.append(matched_label_i)
        
        # do nothing
        for inst_i in range(len_i):
            if inst_i in matched_label_idx_ls:
                continue
            consensus_mask_label.append(mask_label_i[inst_i])
            consensus_mask_dict[len_curr] = {v_i: inst_i}
            for prev_v_i in range(v_i):
                consensus_mask_dict[len_curr][prev_v_i] = -1
            len_curr += 1
        
        # # add additional label
        # for inst_i in range(len_i):
        #     if inst_i in matched_label_idx_ls:
        #         continue
        #     consensus_mask_label.append(mask_label_i[inst_i])
        #     consensus_mask_dict[len_curr] = {v_i: inst_i}
        #     for prev_v_i in range(v_i):
        #         add_mask = self.add_mask_in_j_use_i_inst_n(v_i, prev_v_i, inst_i)
        #         if add_mask is None:
        #             consensus_mask_dict[len_curr][prev_v_i] = -1
        #         else:
        #             consensus_mask_dict[len_curr][prev_v_i] = len(self.curr_obs_torch['mask_label'][prev_v_i])
        #             self.curr_obs_torch['mask'][prev_v_i][add_mask > 0.5] = len(self.curr_obs_torch['mask_label'][prev_v_i])
        #             self.curr_obs_torch['mask_label'][prev_v_i].append(mask_label_i[inst_i])
        #     len_curr += 1
                
        return consensus_mask_label, consensus_mask_dict

    def adjust_masks_using_consensus(self, consensus_mask_label, consensus_mask_dict, use_other):
        old_masks = self.curr_obs_torch['mask'].clone()
        new_masks = self.curr_obs_torch['mask'].clone()
        H, W = old_masks.shape[1:]
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            
            visible_views = []
            for v_i, label_i in consensus_mask_dict[consensus_i].items():
                if label_i != -1:
                    visible_views.append(v_i)
                    
            for v_i, label_i in consensus_mask_dict[consensus_i].items():
                if label_i == -1:
                    # continue
                    
                    # add additional label
                    for v_j in visible_views:
                        inst_j = consensus_mask_dict[consensus_i][v_j]
                        if use_other:
                            add_mask = self.add_mask_in_j_use_i_inst_n(v_j, v_i, inst_j)
                        else:
                            add_mask = None
                        if add_mask is None:
                            continue
                        consensus_mask_dict[consensus_i][v_i] = len(self.curr_obs_torch['mask_label'][v_i])
                        new_masks[v_i][add_mask > 0.5] = consensus_i
                        self.curr_obs_torch['mask_label'][v_i].append(consensus_label)
                        break
                else:
                    mask_i = old_masks[v_i]
                    mask_i_inst_label_i = (mask_i == label_i)
                    new_masks[v_i][mask_i_inst_label_i] = consensus_i
        self.curr_obs_torch['mask'] = new_masks
    
    def order_consensus(self, consensus_mask_label, consensus_mask_dict, queries):
        queries_wo_period = [query[:-1] if query[-1] == '.' else query for query in queries]
        # the order of consensus is the order of queries
        new_consesus_mask_label = [consensus_mask_label[0]]
        new_consesus_mask_dict = {0: consensus_mask_dict[0]}
        for query in queries_wo_period:
            # find all indices of query in consensus_mask_label
            for i, label in enumerate(consensus_mask_label):
                if label == query:
                    new_consesus_mask_label.append(label)
                    new_consesus_mask_dict[len(new_consesus_mask_label)-1] = consensus_mask_dict[i]
        return new_consesus_mask_label, new_consesus_mask_dict
                    
    def align_instance_mask_v2(self, queries, use_other=True):
        num_view = self.curr_obs_torch['color'].shape[0]
        assert num_view > 0
        consensus_mask_label = self.curr_obs_torch['mask_label'][0].copy()
        consensus_mask_dict = {} # map label idx to a dict, each dict contains the view index and the instance id
        for i, label in enumerate(consensus_mask_label):
            consensus_mask_dict[i] = {0: i}
            
            # if label not in consensus_mask_dict:
            #     consensus_mask_dict[label] = [{0: i},]
            # else:
            #     consensus_mask_dict[label].append({0: i})
        
        for i in range(1, num_view):
            len_i = len(self.curr_obs_torch['mask_label'][i])
            len_curr = len(consensus_mask_label)
            
            if len_i < len_curr:
                # some mask labels are missing in view i
                consensus_mask_label, consensus_mask_dict = self.update_with_missing_label(consensus_mask_label, consensus_mask_dict, i)
            elif len_i == len_curr:
                assert consensus_mask_label == self.curr_obs_torch['mask_label'][i]
                # all mask labels are matched
                consensus_mask_label, consensus_mask_dict = self.update_with_equal_label(consensus_mask_label, consensus_mask_dict, i)
            else:
                consensus_mask_label, consensus_mask_dict = self.update_with_additional_label(consensus_mask_label, consensus_mask_dict, i)
                consensus_mask_label, consensus_mask_dict = self.order_consensus(consensus_mask_label, consensus_mask_dict, queries)
        
        self.adjust_masks_using_consensus(consensus_mask_label, consensus_mask_dict, use_other = use_other)
        self.curr_obs_torch['consensus_mask_label'] = consensus_mask_label
    
    def align_with_prev_mask(self, mask):
        # :param new_mask: [num_cam, H, W, num_instance] torch tensor, the new detected mask
        out_mask = torch.zeros_like(mask).to(self.device, dtype=torch.bool)
        for cam_i in range(self.num_cam):
            for instance_i in range(len(self.track_ids)):
                mask_i = mask[cam_i, ..., instance_i]
                intersec_nums = (mask_i[..., None] & self.curr_obs_torch['mask'][cam_i].to(torch.bool)).sum(dim=(0,1)) # [num_instance]
                correct_label = intersec_nums.argmax()
                out_mask[cam_i, ..., correct_label] = mask_i
        out_mask = out_mask.to(self.device, dtype=torch.uint8)
        return out_mask
    
    # DEPRECATED
    # def text_query_for_inst_mask(self, query, threshold, use_sam=False):
    #     if 'color' not in self.curr_obs_torch:
    #         print('Please call update() first!')
    #         exit()
        
    #     if not self.xmem_first_mask_loaded:
    #         query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
    #         for i in range(self.num_cam):
    #             mask = grounded_instance_sam_np(self.curr_obs_torch['color'][i], query, self.ground_dino_model, self.sam_model, threshold)
    #             query_mask[i] = mask
    #         self.curr_obs_torch[f'mask_{query}'] = query_mask # [num_cam, H, W]
    #         # align instance mask id to the first frame
    #         self.align_instance_mask(query)
    #         self.curr_obs_torch[f'mask_{query}'] = self.xmem_process(self.curr_obs_torch['color'], query_mask)
    #     elif self.xmem_first_mask_loaded and not use_sam:
    #         self.curr_obs_torch[f'mask_{query}'] = self.xmem_process(self.curr_obs_torch['color'], None) # [num_cam, H, W, num_instance]
    #     elif self.xmem_first_mask_loaded and use_sam:
    #         query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
    #         for i in range(self.num_cam):
    #             mask = grounded_instance_sam_np(self.curr_obs_torch['color'][i], query, self.ground_dino_model, self.sam_model, threshold)
    #             query_mask[i] = mask
    #         query_mask = query_mask.to(torch.uint8)
    #         query_mask = instance2onehot(query_mask, len(self.track_ids)) # [num_cam, H, W, num_instance]
    #         query_mask = self.align_with_prev_mask(query_mask) # [num_cam, H, W, num_instance]
    #         query_mask = onehot2instance(query_mask) # [num_cam, H, W]
    #         self.curr_obs_torch[f'mask_{query}'] = self.xmem_process(self.curr_obs_torch['color'], query_mask) # [num_cam, H, W, num_instance]
    #         # self.curr_obs_torch[f'mask_{query}'] = query_mask # [num_cam, H, W]
    
    def text_queries_for_inst_mask_no_track(self, queries, thresholds, merge_all = False):
        query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
        labels = []
        for i in range(self.num_cam):
            mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
            labels.append(label)
            query_mask[i] = mask
        self.curr_obs_torch['mask'] = query_mask # [num_cam, H, W]
        self.curr_obs_torch['mask_label'] = labels # [num_cam, ] list of list
        _, idx = np.unique(labels[0], return_index=True)
        self.curr_obs_torch['semantic_label'] = list(np.array(labels[0])[np.sort(idx)]) # list of semantic label we have
        # verfiy the assumption that the mask label is the same for all cameras
        # for i in range(self.num_cam):
        #     try:
        #         assert self.curr_obs_torch['mask_label'][i] == self.curr_obs_torch['mask_label'][0]
        #     except:
        #         print('The mask label is not the same for all cameras!')
        #         print(self.curr_obs_torch['mask_label'])
        #         for j in range(self.num_cam):
        #             for k in range(len(self.curr_obs_torch['mask_label'][j])):
        #                 plt.subplot(1, len(self.curr_obs_torch['mask_label'][j]), k+1)
        #                 plt.imshow(self.curr_obs_torch['mask'][j].detach().cpu().numpy() == k)
        #             plt.show()
        #         raise AssertionError
        # align instance mask id to the first frame
        self.align_instance_mask_v2(queries)
        self.curr_obs_torch[f'mask'] = instance2onehot(self.curr_obs_torch[f'mask'].to(torch.uint8), len(self.curr_obs_torch['consensus_mask_label'])).to(dtype=torch.float32)
    
    def text_queries_for_inst_mask(self, queries, thresholds, use_sam=False, merge_all = False):
        if 'color' not in self.curr_obs_torch:
            print('Please call update() first!')
            exit()
        
        if not self.xmem_first_mask_loaded:
            query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
            labels = []
            for i in range(self.num_cam):
                mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
                labels.append(label)
                query_mask[i] = mask
            self.curr_obs_torch['mask'] = query_mask # [num_cam, H, W]
            self.curr_obs_torch['mask_label'] = labels # [num_cam, ] list of list
            _, idx = np.unique(labels[0], return_index=True)
            self.curr_obs_torch['semantic_label'] = list(np.array(labels[0])[np.sort(idx)]) # list of semantic label we have
            
            # fig, axs = plt.subplots(4, 4 + 1, sharex=True, figsize=(20,20))
            # for i in range(4):
            #     axs[i, 0].imshow(self.curr_obs_torch['color'][i])
            #     for j in range(4):
            #         axs[i, j + 1].imshow(query_mask.detach().cpu().numpy()[i] == j)
            # plt.show()
            
            # # verfiy the assumption that the mask label is the same for all cameras
            # for i in range(self.num_cam):
            #     try:
            #         assert self.curr_obs_torch['mask_label'][i] == self.curr_obs_torch['mask_label'][0]
            #     except:
            #         print('The mask label is not the same for all cameras!')
            #         print(self.curr_obs_torch['mask_label'])
            #         for j in range(self.num_cam):
            #             for k in range(len(self.curr_obs_torch['mask_label'][j])):
            #                 plt.subplot(1, len(self.curr_obs_torch['mask_label'][j]), k+1)
            #                 plt.imshow(self.curr_obs_torch['mask'][j].detach().cpu().numpy() == k)
            #             plt.show()
            #         raise AssertionError
            
            # align instance mask id to the first frame
            print(self.curr_obs_torch['mask_label'])
            self.align_instance_mask_v2(queries)
            self.curr_obs_torch[f'mask'] = self.xmem_process(self.curr_obs_torch['color'], self.curr_obs_torch['mask'])
        elif self.xmem_first_mask_loaded and not use_sam:
            self.curr_obs_torch[f'mask'] = self.xmem_process(self.curr_obs_torch['color'], None) # [num_cam, H, W, num_instance]
        elif self.xmem_first_mask_loaded and use_sam:
            query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
            for i in range(self.num_cam):
                mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
                query_mask[i] = mask
                # labels.append(label)
            query_mask = query_mask.to(torch.uint8)
            query_mask = instance2onehot(query_mask, len(self.track_ids)) # [num_cam, H, W, num_instance]
            query_mask = self.align_with_prev_mask(query_mask) # [num_cam, H, W, num_instance]
            query_mask = onehot2instance(query_mask) # [num_cam, H, W]
            self.curr_obs_torch[f'mask'] = self.xmem_process(self.curr_obs_torch['color'], query_mask) # [num_cam, H, W, num_instance]
            # self.curr_obs_torch[f'mask'] = query_mask # [num_cam, H, W]
    
    def get_query_obj_pcd(self):
        color = self.curr_obs_torch['color']
        depth = self.curr_obs_torch['depth'].detach().cpu().numpy()
        mask = self.curr_obs_torch['mask'].detach().cpu().numpy()
        mask = (mask[..., 1:].sum(axis=-1) > 0)
        for i in range(self.num_cam):
            mask[i] = (cv2.erode((mask[i] * 255).astype(np.uint8), np.ones([2, 2], np.uint8), iterations=1) / 255).astype(bool)
        K = self.curr_obs_torch['K'].detach().cpu().numpy()
        pose = self.curr_obs_torch['pose'].detach().cpu().numpy() # [num_cam, 3, 4]
        pad = np.tile(np.array([[[0,0,0,1]]]), [pose.shape[0], 1, 1])
        pose = np.concatenate([pose, pad], axis=1)
        pcd = aggr_point_cloud_from_data(color, depth, K, pose, downsample=False, masks=mask)
        return pcd
    
    def vis_3d(self, pts, res, params):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param params: dict for other useful params
        
        pts = pts[res['valid_mask']].cpu().numpy()
        dist = res['dist'][res['valid_mask']].cpu().numpy()
        # visualize dist
        dist_vol = go.Figure(data=[go.Scatter3d(x=pts[:,0],
                                                y=pts[:,1],
                                                z=pts[:,2],
                                                mode='markers',
                                                marker=dict(
                                                    size=2,
                                                    color=dist,
                                                    colorscale='Viridis',
                                                    colorbar=dict(thickness=20, ticklen=4),))],
                             layout=go.Layout(scene=dict(aspectmode='data'),))
        dist_vol.show()
        
        # # visualize features
        # features = res['dino_feats'][res['valid_mask']].cpu().numpy()
        # pca = params['pca']
        # features_pca = pca.transform(features)
        # for i in range(features_pca.shape[1]):
        #     features_pca[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
        # features_pca = (features_pca * 255).astype(np.uint8)
        # colors = []
        # for i in range(0, features_pca.shape[0], 1):
        #     colors.append(f'rgb({features_pca[i, 0]}, {features_pca[i, 1]}, {features_pca[i, 2]})')
        # features_vol = go.Figure(data=[go.Scatter3d(x=pts[:,0],
        #                                             y=pts[:,1],
        #                                             z=pts[:,2],
        #                                             mode='markers',
        #                                             marker=dict(
        #                                                 size=2,
        #                                                 color=colors,
        #                                                 colorbar=dict(thickness=20, ticklen=4),))],
        #                          layout=go.Layout(scene=dict(aspectmode='data'),))
        # features_vol.show()
        
        # # visualize masks
        # query_masks = res['query_masks'][res['valid_mask']].cpu().numpy()
        # NQ = res['query_masks'].shape[-1]
        # for i in range(NQ):
        #     mask_vol = go.Figure(data=[go.Scatter3d(x=pts[:,0],
        #                                             y=pts[:,1],
        #                                             z=pts[:,2],
        #                                             mode='markers',
        #                                             marker=dict(
        #                                                 size=2,
        #                                                 color=query_masks[:,i],
        #                                                 colorscale='Viridis',
        #                                                 colorbar=dict(thickness=20, ticklen=4),))],
        #                          layout=go.Layout(scene=dict(aspectmode='data'),))
        #     mask_vol.show()
    
    # visualize 3d field
    def interactive_corr(self, src_info, tgt_info, pts, res):
        # :param src_info: dict contains:
        #                  - 'color': (H, W, 3) np array, color image
        #                  - 'dino_feats': (H, W, f) torch tensor, dino features
        # :param tgt_info: dict contains:
        #                  - 'color': (K, H, W, 3) np array, color image
        #                  - 'pose': (K, 3, 4) torch tensor, pose of the camera
        #                  - 'K': (K, 3, 3) torch tensor, camera intrinsics
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        num_tgt = len(tgt_info['color'])
        sim_scale = 0.05
        imshow_scale = 0.6
        
        viridis_cmap = cm.get_cmap('viridis')
        
        def drawHeatmap(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                src_color_render_curr = draw_keypoints(src_info['color'], np.array([[x, y]]), colors=[(255, 0, 0)], radius=5)
                cv2.imshow('src', src_color_render_curr[..., ::-1])
                src_feat_tensor = src_info['dino_feats'][y, x]
                tgt_feat_sims_tensor = compute_similarity_tensor(res['dino_feats'], src_feat_tensor, scale=sim_scale, dist_type='l2') # [N]
                tgt_feat_sims_tensor = (tgt_feat_sims_tensor - tgt_feat_sims_tensor.min()) / (tgt_feat_sims_tensor.max() - tgt_feat_sims_tensor.min()) # [N]
                tgt_feat_sims = tgt_feat_sims_tensor.detach().cpu().numpy()
                tgt_feat_sim_imgs = (viridis_cmap(tgt_feat_sims)[..., :3] * 255)[..., ::-1]
                
                pts_2d, _, _ = project_points_coords(pts, tgt_info['pose'], tgt_info['K']) # [N, 2]
                pts_2d = pts_2d.detach().cpu().numpy()
                pts_2d = pts_2d[..., ::-1]
                
                max_sim_idx = tgt_feat_sims_tensor.argmax()
                match_pt_3d = pts[max_sim_idx][None]
                match_pt_2d, _, _ = project_points_coords(match_pt_3d, tgt_info['pose'], tgt_info['K'])
                match_pt_2d = match_pt_2d.detach().cpu().numpy()
                
                merge_imgs = []
                
                for idx in range(num_tgt):
                    heatmap = np.zeros_like(tgt_info['color'][idx])
                    heatmap = heatmap.reshape(self.H * self.W, 3)
                    pts_2d_i = pts_2d[idx].astype(np.int32)
                    valid_pts = (pts_2d_i[:, 0] >= 0) & (pts_2d_i[:, 0] < self.W) & (pts_2d_i[:, 1] >= 0) & (pts_2d_i[:, 1] < self.H)
                    pts_2d_flat_idx = np.ravel_multi_index(pts_2d_i[valid_pts].T, (self.H, self.W))
                    heatmap[pts_2d_flat_idx] = tgt_feat_sim_imgs[valid_pts]
                    heatmap = heatmap.reshape(self.H, self.W, 3).astype(np.uint8)
                    cv2.imshow(f'tgt_heatmap_{idx}', heatmap)
                    
                    tgt_imshow_curr = draw_keypoints(tgt_info['color'][idx], match_pt_2d[idx], colors=[(255, 0, 0)], radius=5)
                    # cv2.imshow(f'tgt_{idx}', tgt_imshow_curr[..., ::-1])
                    
                    merge_img = np.concatenate([heatmap, tgt_imshow_curr[..., ::-1]], axis=1)
                    merge_imgs.append(merge_img)
                    
                merge_imgs = np.concatenate(merge_imgs, axis=0)
                merge_imgs = cv2.resize(merge_imgs, (int(merge_imgs.shape[1] * imshow_scale), int(merge_imgs.shape[0] * imshow_scale)), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('merge', merge_imgs)
                
        cv2.imshow('src', src_info['color'][..., ::-1])
        # for idx in range(num_tgt):
        #     cv2.imshow(f'tgt_{idx}', tgt_info['color'][idx][..., ::-1])
        
        cv2.setMouseCallback('src', drawHeatmap)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def interactive_corr_img(self, src_info, tgt_info):
        src_img = src_info['color'][..., ::-1]
        src_dino = src_info['dino_feats']
        
        tgt_img = tgt_info['color'][..., ::-1]
        tgt_dino = tgt_info['dino_feats'][None].permute(0, 3, 1, 2)
        
        viridis_cmap = cm.get_cmap('viridis')
        
        def drawHeatmap(event, x, y, flags, param):
            # feat_x = int(x / src_img.shape[1] * src_dino.shape[1])
            # feat_y = int(y / src_img.shape[0] * src_dino.shape[0])
            if event == cv2.EVENT_MOUSEMOVE:
                src_img_curr = draw_keypoints(src_img, np.array([[x, y]]), colors=[(255, 0, 0)], radius=5)
                cv2.imshow('src', src_img_curr)
                src_feat = src_dino[y, x]
                tgt_feat_sim = compute_similarity_tensor(tgt_dino, src_feat, scale=0.5, dist_type='l2')[0].detach().cpu().numpy()
                tgt_feat_sim = (tgt_feat_sim - tgt_feat_sim.min()) / (tgt_feat_sim.max() - tgt_feat_sim.min())
                tgt_feat_sim_img = (viridis_cmap(tgt_feat_sim)[..., :3] * 255)[..., ::-1]
                # tgt_feat_sim_img = cv2.resize(tgt_feat_sim_img, (tgt_img.shape[1], tgt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                tgt_color_render_curr = (tgt_feat_sim_img * tgt_feat_sim[..., None] + tgt_img * (1.0 - tgt_feat_sim[..., None])).astype(np.uint8)
                cv2.imshow(f'tgt', tgt_color_render_curr)
                cv2.imshow(f'heatmap', tgt_feat_sim_img.astype(np.uint8))
                # cv2.imshow(f'tgt_{idx}', tgt_feat_sim_img.astype(np.uint8))
        
        cv2.imshow('src', src_img)
        cv2.imshow('tgt', tgt_img)

        cv2.setMouseCallback('src', drawHeatmap)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def extract_mesh(self, pts, res, grid_shape):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param grid_shape: (3) tuple, the shape of the grid
        dist = res['dist'].detach().cpu().numpy()
        dist = dist.reshape(grid_shape)
        smoothed_dist = mcubes.smooth(dist)
        vertices, triangles = mcubes.marching_cubes(smoothed_dist, 0)
        vertices = vertices.astype(np.int32)
        # vertices_flat = np.unravel_index(vertices, grid_shape)
        vertices_flat = np.ravel_multi_index(vertices.T, grid_shape)
        vertices_coords = pts.detach().cpu().numpy()[vertices_flat]
        
        return vertices_coords, triangles
    
    def extract_meshes(self, pts, res, grid_shape):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'mask': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param grid_shape: (3) tuple, the shape of the grid
        vertices_list = []
        triangles_list = []
        mask_label = self.curr_obs_torch['mask_label'][0]
        num_instance = len(mask_label)
        
        mask = res['mask']
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        for i in range(num_instance):
            mask_i = mask[:, i] > 0.6
            dist = res['dist'].clone()
            dist[~mask_i] = 1e3
            dist = dist.detach().cpu().numpy()
            dist = dist.reshape(grid_shape)
            smoothed_dist = mcubes.smooth(dist)
            vertices, triangles = mcubes.marching_cubes(smoothed_dist, 0)
            vertices = vertices.astype(np.int32)
            # vertices_flat = np.unravel_index(vertices, grid_shape)
            vertices_flat = np.ravel_multi_index(vertices.T, grid_shape)
            vertices_coords = pts.detach().cpu().numpy()[vertices_flat]
            
            vertices_list.append(vertices_coords)
            triangles_list.append(triangles)
        
        return vertices_list, triangles_list

    def create_mask_mesh(self, vertices, triangles, res):
        # :param vertices: (N, 3) numpy array, the vertices of the mesh
        # :param triangles: (M, 3) numpy array, the triangles of the mesh
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        query_masks = res['query_masks'].detach().cpu().numpy()
        mask_meshes = []
        for i in range(query_masks.shape[1]):
            vertices_color = trimesh.visual.interpolate(query_masks[:,i], color_map='viridis')
            mask_meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=vertices_color))
        return mask_meshes

    def create_instance_mask_mesh(self, vertices, triangles, res):
        # :param vertices: (N, 3) numpy array, the vertices of the mesh
        # :param triangles: (M, 3) numpy array, the triangles of the mesh
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'mask_*': (N, nq) torch tensor, the instance masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        mask_meshes = []
        for k in res.keys():
            if k.startswith('mask'):
                mask = res[k].detach().cpu().numpy()
                num_instance = mask.shape[1]
                mask = onehot2instance(mask) # (N, nq) -> (N)
                
                # mask_vis = np.zeros((mask.shape[0], 3))
                # mask_vis[mask == 0] = np.array(series_RGB[5])
                # mask_vis[mask == 1] = np.array(series_RGB[1])
                # mask_vis[mask == 2] = np.array(series_RGB[2])
                # mask_vis[mask == 3] = np.array(series_RGB[4])
                # mask_vis[mask == 4] = np.array(series_RGB[6])
                
                # mask_vis = np.concatenate([mask_vis, np.ones((mask_vis.shape[0], 1)) * 255], axis=1).astype(np.uint8)
                
                vertices_color = trimesh.visual.interpolate(mask / num_instance, color_map='jet')
                mask_meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=vertices_color))
        return mask_meshes

    # def create_descriptor_mesh(self, vertices, triangles, res, params):
    #     pca = params['pca']
    #     features = res['dino_feats'].detach().cpu().numpy()
    #     features_pca = pca.transform(features)
    #     for i in range(features_pca.shape[1]):
    #         features_pca[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
    #     features_pca = (features_pca * 255).astype(np.uint8)
    #     features_pca = np.concatenate([features_pca, np.ones((features_pca.shape[0], 1), dtype=np.uint8) * 255], axis=1)
    #     features_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=features_pca)
    #     return features_mesh
    
    def create_descriptor_mesh(self, vertices, triangles, res, params, mask_out_bg):
        pca = params['pca']
        features = res['dino_feats'].detach().cpu().numpy()
        mask = res['mask'].detach().cpu().numpy()
        mask = onehot2instance(mask) # (N, nq) -> (N)
        bg = (mask == 0)
        features_pca = pca.transform(features)
        
        features_rgb = np.zeros((features_pca.shape[0], 3))
        
        for i in range(features_pca.shape[1]):
            if not mask_out_bg:
                features_rgb[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
            else:
                features_rgb[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
                # features_rgb[~bg, i] = (features_pca[~bg, i] - features_pca[~bg, i].min()) / (features_pca[~bg, i].max() - features_pca[~bg, i].min())
                # features_rgb[bg, i] = 0.8
        features_rgb[bg] = np.ones(3) * 0.8
        features_rgb = features_rgb[..., ::-1]
            
        features_rgb = (features_rgb * 255).astype(np.uint8)
        features_rgb = np.concatenate([features_rgb, np.ones((features_rgb.shape[0], 1), dtype=np.uint8) * 255], axis=1)
        features_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=features_rgb)
        return features_mesh

    def create_color_mesh(self, vertices, triangles, res):
        colors = res['color_tensor'].detach().cpu().numpy()[..., ::-1]
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.uint8) * 255], axis=1)
        color_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=colors)
        return color_mesh
    
    def select_features(self, color, params):
        H, W = color.shape[:2]
        features = self.extract_features(color[None], params)
        features = F.interpolate(features.permute(0, 3, 1, 2),
                                 size=color.shape[:2],
                                 mode='bilinear',
                                 align_corners=True).permute(0, 2, 3, 1)[0]
        
        param = {'src_pts': []} # (col, row)
        
        color_cands = [(180,119,31),
                       (14,127,255),
                       (44,160,44),
                       (40,39,214),
                       (189,103,148),]
        
        def select_kypts(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                param['src_pts'].append([x, y])
                color_curr = draw_keypoints(color, np.array([[x, y]]), colors=[(0, 0, 255)], radius=5)
                cv2.imshow('src', color_curr[..., ::-1])
            elif event == cv2.EVENT_MOUSEMOVE:
                color_curr = draw_keypoints(color, np.array([[x, y]]), colors=[(255, 0, 0)], radius=5)
                for kp in param['src_pts']:
                    color_curr = draw_keypoints(color_curr, np.array([kp]), colors=[(0, 0, 255)], radius=5)
                cv2.imshow('src', color_curr[..., ::-1])
        
        cv2.imshow('src', color[..., ::-1])
        cv2.setMouseCallback('src', select_kypts, param)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
        
        src_feats = [features[p[1], p[0]] for p in param['src_pts']]
        src_feats = torch.stack(src_feats)
        
        N = src_feats.shape[0]
        try:
            assert N <= len(color_cands)
            colors = color_cands[:N]
        except:
            print('not enough color candidates')
            return
        color_with_kypts = color.copy()
        color_with_kypts = draw_keypoints(color_with_kypts, np.array(param['src_pts']), colors=colors, radius=int(5 * H / 360))
        
        return src_feats, color_with_kypts
    
    def select_features_rand(self, boundaries, N, per_instance=False, res = None, init_idx = -1):
        # randomly select N features for object {query_text} in 3D space 
        res = 0.001 if res is None else res
        dist_threshold = 0.005
        
        grid, grid_shape = create_init_grid(boundaries, res)
        grid = grid.to(self.device, dtype=torch.float32)
        
        # out = self.eval(init_grid)
        with torch.no_grad():
            out = self.batch_eval(grid, return_names=['mask'])
        
        dist_mask = torch.abs(out['dist']) < dist_threshold
        
        label = self.curr_obs_torch['consensus_mask_label']
        
        last_label = label[0]
        
        src_feats_list = []
        img_list = []
        src_pts_list = []
        mask = out['mask'] # (N, num_instances) where 0 is background
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        for i in range(1, len(label)):
            if label[i] == last_label and not per_instance:
                continue
            instance_mask = mask[:, i] > 0.6
            masked_pts = grid[instance_mask & dist_mask & out['valid_mask']]
            
            sample_pts, sample_idx, _ = fps_np(masked_pts.detach().cpu().numpy(), N, init_idx=init_idx)
            # src_feats_list.append(out['dino_feats'][sample_idx])
            src_feats_list.append(self.eval(torch.from_numpy(sample_pts).to(self.device, torch.float32))['dino_feats'])
            src_pts_list.append(sample_pts)
            
            num_pts = sample_pts.shape[0]
            pose = self.curr_obs_torch['pose'][0].detach().cpu().numpy()
            K = self.curr_obs_torch['K'][0].detach().cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            img = self.curr_obs_torch['color'][0]
            
            cmap = cm.get_cmap('viridis')
            colors = ((cmap(np.linspace(0, 1, num_pts))[:, :3]) * 255).astype(np.int32)
            
            sample_pts = np.concatenate([sample_pts, np.ones([num_pts, 1])], axis=-1) # [num_pts, 4]
            sample_pts = np.matmul(pose, sample_pts.T)[:3].T # [num_pts, 3]
            
            sample_pts_2d = sample_pts[:, :2] / sample_pts[:, 2:] # [num_pts, 2]
            sample_pts_2d[:, 0] = sample_pts_2d[:, 0] * fx + cx
            sample_pts_2d[:, 1] = sample_pts_2d[:, 1] * fy + cy
            
            sample_pts_2d = sample_pts_2d.astype(np.int32)
            sample_pts_2d = sample_pts_2d.reshape(num_pts, 2)
            img = draw_keypoints(img, sample_pts_2d, colors, radius=5)
            img_list.append(img)
            last_label = label[i]
        
        del out
        return src_feats_list, src_pts_list, img_list
    
    def select_features_rand_v2(self, boundaries, N, per_instance=False):
        N_per_cam = N // self.num_cam
        src_feats_list = []
        img_list = []
        src_pts_list = []
        label = self.curr_obs_torch['mask_label'][0]
        last_label = label[0]
        for i in range(1, len(label)):
            if label[i] == last_label and not per_instance:
                continue
            src_pts_np = []
            for cam_i in range(self.num_cam):
                instance_mask = (self.curr_obs_torch['mask'][cam_i, :, :, i]).detach().cpu().numpy().astype(bool)
                depth_i = self.curr_obs_torch['depth'][cam_i].detach().cpu().numpy()
                K_i = self.curr_obs_torch['K'][cam_i].detach().cpu().numpy()
                pose_i = self.curr_obs_torch['pose'][cam_i].detach().cpu().numpy()
                pose_i = np.concatenate([pose_i, np.array([[0, 0, 0, 1]])], axis=0)
                valid_depth = (depth_i > 0.0) & (depth_i < 1.5)
                instance_mask = instance_mask & valid_depth
                instance_mask = (instance_mask * 255).astype(np.uint8)
                # plt.subplot(1, 2, 1)
                # plt.imshow(instance_mask)
                instance_mask = cv2.erode(instance_mask, np.ones([15, 15], np.uint8), iterations=1)
                # plt.subplot(1, 2, 2)
                # plt.imshow(instance_mask)
                # plt.show()
                instance_mask_idx = np.array(instance_mask.nonzero()).T # (num_pts, 2)
                sel_idx, _, _ = fps_np(instance_mask_idx, N_per_cam)
                
                sel_depth = depth_i[sel_idx[:, 0], sel_idx[:, 1]]
                
                src_pts = np.zeros([N_per_cam, 3])
                src_pts[:, 0] = (sel_idx[:, 1] - K_i[0, 2]) * sel_depth / K_i[0, 0]
                src_pts[:, 1] = (sel_idx[:, 0] - K_i[1, 2]) * sel_depth / K_i[1, 1]
                src_pts[:, 2] = sel_depth
                
                # sample_pts = np.concatenate([sample_pts, np.ones([N, 1])], axis=-1) # [num_pts, 4]
                # sample_pts = np.matmul(pose, sample_pts.T)[:3].T # [num_pts, 3] # world to camera
                
                src_pts = np.matmul(np.linalg.inv(pose_i), np.concatenate([src_pts, np.ones([N_per_cam, 1])], axis=-1).T)[:3].T # [num_pts, 3] # camera to world
                
                src_pts_np.append(src_pts)
            sample_pts = np.concatenate(src_pts_np, axis=0)
            src_pts_list.append(sample_pts)
            src_feats_list.append(self.eval(torch.from_numpy(sample_pts).to(self.device, torch.float32))['dino_feats'])
            
            cmap = cm.get_cmap('jet')
            colors = ((cmap(np.linspace(0, 1, N))[:, :3]) * 255).astype(np.int32)
            
            pose = self.curr_obs_torch['pose'][0].detach().cpu().numpy()
            K = self.curr_obs_torch['K'][0].detach().cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            img = self.curr_obs_torch['color'][0]
            
            sample_pts = np.concatenate([sample_pts, np.ones([N, 1])], axis=-1) # [num_pts, 4]
            sample_pts = np.matmul(pose, sample_pts.T)[:3].T # [num_pts, 3]
            
            sample_pts_2d = sample_pts[:, :2] / sample_pts[:, 2:] # [num_pts, 2]
            sample_pts_2d[:, 0] = sample_pts_2d[:, 0] * fx + cx
            sample_pts_2d[:, 1] = sample_pts_2d[:, 1] * fy + cy
            
            sample_pts_2d = sample_pts_2d.astype(np.int32)
            sample_pts_2d = sample_pts_2d.reshape(N, 2)
            img = draw_keypoints(img, sample_pts_2d, colors, radius=5)
            img_list.append(img)
            last_label = label[i]

        return src_feats_list, src_pts_list, img_list
    
    # DEPRECATED: does not use mask information
    # def find_correspondences(self, src_feats, pts, res):
    #     # :param src_feats: (K, f) torch tensor
    #     # :param pts: (N, 3) torch tensor in world frame
    #     # :param res: dict contains:
    #     #             - 'dist': (N) torch tensor, dist to the closest point on the surface
    #     #             - 'dino_feats': (N, f) torch tensor, the features of the points
    #     #             - 'valid_mask': (N) torch tensor, whether the point is valid
    #     tgt_feats = res['dino_feats'][res['valid_mask']] # (N', f)
    #     pts = pts[res['valid_mask']] # (N', 3)
        
    #     sim_tensor = compute_similarity_tensor_multi(tgt_feats[None].permute(0, 2, 1),
    #                                                  src_feats,
    #                                                  scale = 0.5,
    #                                                  dist_type='l2') # (1, K, N')
    #     sim_tensor = sim_tensor[0].permute(1, 0) # (N', K)
    #     match_pts, conf = extract_kypts_gpu(sim_tensor, pts, match_metric='sum') # (K, 3)
    #     return match_pts, conf
    
    # DEPRECATED: only support one category
    # def find_correspondences_with_mask(self, src_feats, pts, res, debug=False, debug_info=None):
    #     # :param src_feats: (K, f) torch tensor
    #     # :param pts: (num_instances - 1, K, 3) torch tensor in world frame
    #     # :param res: dict contains:
    #     #             - 'dist': (N) torch tensor, dist to the closest point on the surface
    #     #             - 'dino_feats': (N, f) torch tensor, the features of the points
    #     #             - 'mask_*': (N, num_instances) torch tensor, whether the point is in the instance
    #     #             - 'valid_mask': (N) torch tensor, whether the point is valid
    #     mask = res['mask_shoe'] # (N, num_instances) where 0 is background
    #     mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
    #     # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
    #     # mask = instance2onehot(mask_onehot) # (N, num_instances)
    #     num_instances = mask.shape[1]
    #     match_pts_list = []
    #     for i in range(1, num_instances):
    #         # instance_mask = mask[:, i]
    #         instance_mask = mask[:, i] > 0.6
    #         tgt_feats = res['dino_feats'][res['valid_mask'] & instance_mask] # (N', f)
    #         pts_i = pts[res['valid_mask'] & instance_mask] # (N', 3)
    #         # tgt_feats = res['dino_feats'][res['valid_mask']] # (N', f)
    #         # pts_i = pts[res['valid_mask']] # (N', 3)
    #         instance_mask = instance_mask[res['valid_mask']] # (N', )
            
    #         if debug:
    #             pts_i_np = pts_i.cpu().numpy()
    #             full_pts = debug_info['full_pts']
    #             full_mask = debug_info['full_mask']
    #             full_pts_1 = full_pts[full_mask[:, 1].astype(bool)]
    #             full_pts_2 = full_pts[full_mask[:, 2].astype(bool)]
    #             vol = go.Figure(data=[go.Scatter3d(x=pts_i_np[:,0],
    #                                                     y=pts_i_np[:,1],
    #                                                     z=pts_i_np[:,2],
    #                                                     mode='markers',
    #                                                     marker=dict(
    #                                                         size=2,
    #                                                         colorscale='Viridis',
    #                                                         colorbar=dict(thickness=20, ticklen=4),)),
    #                                   go.Scatter3d(x=full_pts_1[:,0],
    #                                                y=full_pts_1[:,1],
    #                                                z=full_pts_1[:,2],
    #                                                mode='markers',
    #                                                marker=dict(
    #                                                    size=2,
    #                                                    colorscale='Viridis',
    #                                                    colorbar=dict(thickness=20, ticklen=4),)),
    #                                   go.Scatter3d(x=full_pts_2[:,0],
    #                                                y=full_pts_2[:,1],
    #                                                z=full_pts_2[:,2],
    #                                                mode='markers',
    #                                                marker=dict(
    #                                                    size=2,
    #                                                    colorscale='Viridis',
    #                                                    colorbar=dict(thickness=20, ticklen=4),))],
    #                                     layout=go.Layout(scene=dict(aspectmode='data'),))
    #             vol.show()
            
    #         sim_tensor = compute_similarity_tensor_multi(tgt_feats[None].permute(0, 2, 1),
    #                                                     src_feats,
    #                                                     scale = 0.5,
    #                                                     dist_type='l2') # (1, K, N')
    #         sim_tensor = sim_tensor[0].permute(1, 0) # (N', K)
    #         # sim_tensor = sim_tensor * torch.clamp_min(torch.log(instance_mask[:, None] + 1e-7) + 1, 0) # (N', K)
    #         match_pts = extract_kypts_gpu(sim_tensor, pts_i, match_metric='sum') # (K, 3)
    #         match_pts_list.append(match_pts)
    #     match_pts = torch.stack(match_pts_list, dim=0) # (num_instances - 1, K, 3)
    #     return match_pts
    
    def compute_conf(self, matched_pts, tgt_feats, conf_sigma):
        # :param matched_pts: (K, 3) torch tensor
        # :param tgt_feats: (K, f) torch tensor
        # :return conf: (K, ) torch tensor
        # matched_pts_eval = self.eval(matched_pts, return_names=['dino_feats'], return_inter=True)
        matched_pts_eval = self.eval(matched_pts, return_names=['dino_feats'])
        feat_dist = torch.norm(matched_pts_eval['dino_feats'] - tgt_feats, dim=1) # (K, )
        # inter_feat = matched_pts_eval['dino_feats_inter'] # (num_view, K, f)
        # inter_feat_dist = torch.norm(inter_feat - tgt_feats[None], dim=2) # (num_view, K)
        # inter_feat_conf = torch.exp(-inter_feat_dist / conf_sigma) # (num_view, K)
        # for i in range(inter_feat_conf.shape[0]):
        #     print(f'conf in view {i}: {inter_feat_conf[i,0].item()}')
        conf = torch.exp(-feat_dist / conf_sigma) * torch.exp(-torch.abs(matched_pts_eval['dist']) * 50) * matched_pts_eval['valid_mask'] # (K, )
        return conf
    
    def find_correspondences_with_mask(self,
                                       src_feat_info,
                                       pts,
                                       last_match_pts_list,
                                       res,
                                       debug=False,
                                       debug_info=None):
        # :param src_feat_info dict
        #        - key: 'object_name'
        #        - value: dict contains:
        #          - 'path': str, path to the image
        #          - 'params': dict contains:
        #            - 'patch_h': int, height of the patch
        #            - 'patch_w': int, width of the patch
        #            - 'sam_threshold': float, threshold for sam
        #          - 'src_feats': [-1, f] torch tensor, features to track
        # :param pts: (num_instances - 1, K, 3) torch tensor in world frame
        # :param last_match_pts_list: list of (K, 3) numpy array
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'mask_*': (N, num_instances) torch tensor, whether the point is in the instance
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        mask = res['mask'] # (N, num_instances) where 0 is background
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
        # mask = instance2onehot(mask_onehot) # (N, num_instances)
        num_instances = mask.shape[1]
        match_pts_list = []
        conf_list = []
        for i in range(1, num_instances):
            src_feats = src_feat_info[self.curr_obs_torch['mask_label'][0][i]]['src_feats']
            if last_match_pts_list is None:
                last_match_pts = None
            else:
                last_match_pts = torch.from_numpy(last_match_pts_list[i - 1]).to(self.device, dtype=torch.float32)
            
            # instance_mask = mask[:, i]
            instance_mask = mask[:, i] > 0.6
            tgt_feats = res['dino_feats'][res['valid_mask'] & instance_mask] # (N', f)
            pts_i = pts[res['valid_mask'] & instance_mask] # (N', 3)
            # tgt_feats = res['dino_feats'][res['valid_mask']] # (N', f)
            # pts_i = pts[res['valid_mask']] # (N', 3)
            instance_mask = instance_mask[res['valid_mask']] # (N', )
            
            if debug:
                pts_i_np = pts_i.cpu().numpy()
                full_pts = debug_info['full_pts']
                full_mask = debug_info['full_mask']
                full_pts_1 = full_pts[full_mask[:, 1].astype(bool)]
                full_pts_2 = full_pts[full_mask[:, 2].astype(bool)]
                vol = go.Figure(data=[go.Scatter3d(x=pts_i_np[:,0],
                                                        y=pts_i_np[:,1],
                                                        z=pts_i_np[:,2],
                                                        mode='markers',
                                                        marker=dict(
                                                            size=2,
                                                            colorscale='Viridis',
                                                            colorbar=dict(thickness=20, ticklen=4),)),
                                      go.Scatter3d(x=full_pts_1[:,0],
                                                   y=full_pts_1[:,1],
                                                   z=full_pts_1[:,2],
                                                   mode='markers',
                                                   marker=dict(
                                                       size=2,
                                                       colorscale='Viridis',
                                                       colorbar=dict(thickness=20, ticklen=4),)),
                                      go.Scatter3d(x=full_pts_2[:,0],
                                                   y=full_pts_2[:,1],
                                                   z=full_pts_2[:,2],
                                                   mode='markers',
                                                   marker=dict(
                                                       size=2,
                                                       colorscale='Viridis',
                                                       colorbar=dict(thickness=20, ticklen=4),))],
                                        layout=go.Layout(scene=dict(aspectmode='data'),))
                vol.show()
            
            sim_tensor = compute_similarity_tensor_multi(tgt_feats,
                                                        src_feats,
                                                        pts_i,
                                                        last_match_pts,
                                                        scale = 0.5,
                                                        dist_type='l2') # (N', K)
            # sim_tensor = sim_tensor * torch.clamp_min(torch.log(instance_mask[:, None] + 1e-7) + 1, 0) # (N', K)
            match_pts = extract_kypts_gpu(sim_tensor, pts_i, match_metric='sum') # (K, 3)
            # print('stddev of x: ', pts_i[:,0].std().item())
            # print('stddev of y: ', pts_i[:,1].std().item())
            # print('stddev of z: ', pts_i[:,2].std().item())
            # print('stddev:', pts_i.std(dim=0).norm().item())
            topk_sim_idx = torch.topk(sim_tensor, k=1000, dim=0)[1] # (100, K)
            topk_pts = pts_i[topk_sim_idx] # (100, K, 3)
            observability = self.compute_conf(match_pts, src_feats, conf_sigma=1000.) # (K, )
            stability = torch.exp(-topk_pts.std(dim=0).norm(dim=1) * 20.0) # (K, )
            # print('stability:', stability)
            # print('observability:', observability)
            conf = observability * stability
            match_pts_list.append(match_pts.detach().cpu().numpy())
            conf_list.append(conf.detach().cpu().numpy())
        # match_pts = torch.stack(match_pts_list, dim=0) # (num_instances - 1, K, 3)
        return match_pts_list, conf_list
    
    def find_correspondences(self,
                             src_feats,
                             boundaries,
                             instance_id,):
        # :param src_feats torch.Tensor (K, f)
        curr_res = 0.01
        
        curr_que_pts, grid_shape = create_init_grid(boundaries, curr_res)
        curr_que_pts = curr_que_pts.to(self.device, dtype=torch.float32)
        
        out = self.eval(curr_que_pts, return_names=['dino_v2_feats', 'mask'])
        # out = self.eval(curr_que_pts, return_names=['dino_v2_feats'])
        
        for i in range(3):
            # multi-instance tracking
            mask = out['mask'] # (N, num_instances) where 0 is background
            mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
            # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
            # mask = instance2onehot(mask_onehot) # (N, num_instances)
            curr_que_pts_list = []
            mask_instance = mask[:, instance_id] > 0.6 # (N,)
            try:
                assert mask_instance.max() > 0
            except:
                print('no instance found!')
                exit()
            tgt_feats = out['dino_v2_feats'][out['valid_mask'] & mask_instance] # (N', f)
            curr_valid_que_pts = curr_que_pts[out['valid_mask'] & mask_instance] # (N', 3)
            assert tgt_feats.shape[0] == curr_valid_que_pts.shape[0]
            sim_vol = compute_similarity_tensor_multi(tgt_feats,
                                                    src_feats,
                                                    None,
                                                    None,
                                                    scale = 0.5,
                                                    dist_type='l2') # (N', K)
            next_que_pts, next_res = octree_subsample(sim_vol,
                                                    curr_valid_que_pts,
                                                    curr_res,
                                                    topK=200)
            curr_que_pts_list.append(next_que_pts)
            curr_res = next_res
            del curr_que_pts
            curr_que_pts = torch.cat(curr_que_pts_list, dim=0)
            out_keys = list(out.keys())
            for k in out_keys:
                del out[k]
            del out
            out = self.batch_eval(curr_que_pts, return_names=['dino_v2_feats', 'mask'])
        # src_feat_info = {
        #     self.curr_obs_torch['mask_label'][0][instance_id]: {
        #         'src_feats': src_feats,
        #     }
        # }
        
        mask = out['mask'] # (N, num_instances) where 0 is background
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        # src_feats = src_feat_info[self.curr_obs_torch['mask_label'][0][i]]['src_feats']

        instance_mask = mask[:, instance_id] > 0.6
        tgt_feats = out['dino_v2_feats'][out['valid_mask'] & instance_mask] # (N', f)
        pts_i = curr_que_pts[out['valid_mask'] & instance_mask] # (N', 3)
        instance_mask = instance_mask[out['valid_mask']] # (N', )
        
        sim_tensor = compute_similarity_tensor_multi(tgt_feats,
                                                    src_feats,
                                                    None,
                                                    None,
                                                    scale = 0.5,
                                                    dist_type='l2') # (N', K)
        # sim_tensor = sim_tensor * torch.clamp_min(torch.log(instance_mask[:, None] + 1e-7) + 1, 0) # (N', K)
        match_pts = extract_kypts_gpu(sim_tensor, pts_i, match_metric='sum') # (K, 3)
        
        return match_pts

    def tracking(self,
                 src_feat_info,
                 last_match_pts_list,
                 boundaries,
                 rand_ptcl_num):
        # :param src_feat_info dict
        # :param last_match_pts_list list of [rand_ptcl_num, 3] np.array, could be None if no previous match
        
        # debug = (sel_time >= 99999)
        # if debug:
        #     curr_res = 0.002
            
        #     init_grid, grid_shape = create_init_grid(boundaries, curr_res)
        #     init_grid = init_grid.to(self.device, dtype=torch.float32)
            
        #     # out = self.eval(init_grid)
        #     out = self.batch_eval(init_grid, return_names=[])
            
        #     # extract mesh
        #     vertices, triangles = self.extract_mesh(init_grid, out, grid_shape)
        #     # mcubes.export_obj(vertices, triangles, 'sphere.obj')
        
        #     mask = self.curr_obs_torch['mask'].cpu().numpy() # [num_view, H, W, NQ]
        #     mask = onehot2instance(mask) # [num_view, H, W]
        #     mask = mask / mask.max() # [num_view, H, W]
        #     num_view, H, W = mask.shape
        #     cmap = cm.get_cmap('jet')
        #     mask_vis = cmap(mask.reshape(-1)).reshape(num_view, H, W, 4)[..., :3] # [num_view, H, W, 3]
        #     mask_vis = (mask_vis * 255).astype(np.uint8)
        #     merge_vis = np.concatenate([mask_vis[i] for i in range(mask_vis.shape[0])], axis=1)
            
        #     # eval mask and feature of vertices
        #     vertices_tensor = torch.from_numpy(vertices).to(self.device, dtype=torch.float32)
        #     out = self.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask'])
        #     # mask_meshes = self.create_mask_mesh(vertices, triangles, out)
        #     mask_meshes = self.create_instance_mask_mesh(vertices, triangles, out)
        #     full_mask = out['mask'].cpu().numpy() # [N, num_instances]
        #     for mask_mesh in mask_meshes:
        #         mask_mesh.show(smooth=True)
            
        #     plt.imshow(merge_vis)
        #     plt.show()
        #     # feature_mesh = self.create_descriptor_mesh(vertices, triangles, out, {'pca': pca})
        #     # feature_mesh.show(smooth=True)
        
        curr_res = 0.01
        
        curr_que_pts, grid_shape = create_init_grid(boundaries, curr_res)
        curr_que_pts = curr_que_pts.to(self.device, dtype=torch.float32)
        
        out = self.eval(curr_que_pts, return_names=['dino_feats', 'mask'])
        # out = self.eval(curr_que_pts, return_names=['dino_feats'])
        
        for i in range(3):
            
            # multi-instance tracking
            mask = out['mask'] # (N, num_instances) where 0 is background
            mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
            # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
            # mask = instance2onehot(mask_onehot) # (N, num_instances)
            curr_que_pts_list = []
            for instance_id in range(1, mask.shape[1]):
                src_feats = src_feat_info[self.curr_obs_torch['mask_label'][0][instance_id]]['src_feats'] # (K, f,)
                # mask_instance = mask[:, instance_id] # (N,)
                mask_instance = mask[:, instance_id] > 0.6 # (N,)
                try:
                    assert mask_instance.max() > 0
                except:
                    print('no instance found!')
                    exit()
                tgt_feats = out['dino_feats'][out['valid_mask'] & mask_instance] # (N', f)
                curr_valid_que_pts = curr_que_pts[out['valid_mask'] & mask_instance] # (N', 3)
                # tgt_feats = out['dino_feats'][out['valid_mask']] # (N', f)
                # curr_valid_que_pts = curr_que_pts[out['valid_mask']] # (N', 3)
                assert tgt_feats.shape[0] == curr_valid_que_pts.shape[0]
                # mask_instance = mask[:, instance_id][out['valid_mask']] # (N',)
                if last_match_pts_list is None:
                    last_match_pts_i = None
                else:
                    last_match_pts_i = torch.from_numpy(last_match_pts_list[instance_id - 1]).to(self.device, dtype=torch.float32) # (K, 3)
                sim_vol = compute_similarity_tensor_multi(tgt_feats,
                                                        src_feats,
                                                        curr_valid_que_pts,
                                                        last_match_pts_i,
                                                        scale = 0.5,
                                                        dist_type='l2') # (N', K)
                # sim_vol = sim_vol * mask_instance[:, None] # (N', K) # weighted using mask
                # sim_vol = sim_vol * torch.clamp_min(torch.log(mask_instance[:, None] + 1e-7) + 1, 0)
                next_que_pts, next_res = octree_subsample(sim_vol,
                                                        curr_valid_que_pts,
                                                        curr_res,
                                                        topK=1000)
                curr_que_pts_list.append(next_que_pts)
            curr_res = next_res
            del curr_que_pts
            curr_que_pts = torch.cat(curr_que_pts_list, dim=0)
            out_keys = list(out.keys())
            for k in out_keys:
                del out[k]
            del out
            out = self.batch_eval(curr_que_pts, return_names=['dino_feats', 'mask'])
            
            # DEPRECATED: single-instance tracking
            # tgt_feats = out['dino_feats'][out['valid_mask']] # (N', f)
            # curr_valid_que_pts = curr_que_pts[out['valid_mask']] # (N', 3)
            # assert tgt_feats.shape[0] == curr_valid_que_pts.shape[0]
    
            # sim_vol = compute_similarity_tensor_multi(tgt_feats[None].permute(0, 2, 1),
            #                                           src_feats,
            #                                           scale = 0.5,
            #                                           dist_type='l2')[0].permute(1, 0) # (N', K)
            # curr_que_pts, curr_res = octree_subsample(sim_vol,
            #                                           curr_valid_que_pts,
            #                                           curr_res,
            #                                           topK=1000)
            # out = self.eval(curr_que_pts, return_names=['dino_feats'])
        
        # match_pts = self.find_correspondences(src_feats, curr_que_pts, out).detach().cpu().numpy()
        # if debug:
        #     debug_info = {'full_pts': vertices,
        #                 'full_mask': full_mask,}
        # else:
        #     debug_info = None
        match_pts_list, conf_list = self.find_correspondences_with_mask(src_feat_info,
                                                                curr_que_pts,
                                                                last_match_pts_list,
                                                                out,
                                                                debug=False,
                                                                debug_info=None)
        semantic_conf_list = []
        for semantic_label in self.curr_obs_torch['semantic_label'][1:]:
            instance_indices = find_indices(self.curr_obs_torch['mask_label'][0], semantic_label)
            semantic_conf = np.zeros(rand_ptcl_num)
            for instance_idx in instance_indices:
                semantic_conf += conf_list[instance_idx - 1]
            semantic_conf /= len(instance_indices)
            semantic_conf_list.append(semantic_conf)
        
        del curr_que_pts
        out_keys = list(out.keys())
        for k in out_keys:
            del out[k]
        del out
        
        # match_pts_list, avg_conf_per_semantic
        return {'match_pts_list': match_pts_list,
                'semantic_conf_list': semantic_conf_list,
                'instance_conf_list': conf_list}

    def select_conf_pts(self,
                        src_feat_info,
                        avg_conf_per_semantic,
                        database,
                        output_dir):
        avg_window = 5
        full_pts_path = os.path.join(output_dir, 'full_pts')
        # re-plot top K tracking points
        topk_idx_list = []
        topk = 5
        for semantic_i, conf in enumerate(avg_conf_per_semantic):
            src_pts = src_feat_info[self.curr_obs_torch['semantic_label'][semantic_i + 1]]['src_pts']
            sorted_idx = np.argsort(conf)[::-1]
            topk_idx = []
            for idx in sorted_idx:
                if len(topk_idx) == 0 or np.min(np.linalg.norm(src_pts[topk_idx] - src_pts[idx], axis=1)) > 0.05:
                    topk_idx.append(idx)
                if len(topk_idx) == topk:
                    break
            topk_idx_list.append(topk_idx)
        topk_idx_list = np.stack(topk_idx_list, axis=0) # [semantic_num, topk]
        
        partial_vid_path = os.path.join(output_dir, 'partial_tracking.mp4')
        partial_vid = cv2.VideoWriter(partial_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (640 * 2, 360 * 2))
        topk_match_pts_over_time = []
        for sel_time in tqdm(database.get_times()[::5]):
            match_pts_3d = np.load(os.path.join(full_pts_path, f'{sel_time:06d}.npy')) # [instance_num, rand_ptcl_num, 3]
            topk_match_pts_list = []
            for semantic_idx, semantic_label in enumerate(self.curr_obs_torch['semantic_label'][1:]):
                instance_indices = find_indices(self.curr_obs_torch['mask_label'][0], semantic_label)
                for instance_idx in instance_indices:
                    topk_idx = topk_idx_list[semantic_idx]
                    topk_match_pts = match_pts_3d[instance_idx - 1][topk_idx]
                    topk_match_pts_list.append(topk_match_pts)
                    
            if sel_time < avg_window:
                topk_match_pts_over_time.append(topk_match_pts_list)
            else:
                topk_match_pts_over_time.pop(0)
                topk_match_pts_over_time.append(np.stack(topk_match_pts_list, axis=0)) # list of [instance_num, topk, 3]
                topk_match_pts_over_time_np = np.stack(topk_match_pts_over_time, axis=0) # [avg_window, instance_num, topk, 3]
                topk_match_pts_list = topk_match_pts_over_time_np.mean(axis=0) # [instance_num, topk, 3] 
                    
            merge_img = np.zeros((360 * 2, 640 * 2, 3), dtype=np.uint8)
            for view_i in range(4):
                h_idx = view_i // 2
                w_idx = view_i % 2
                merge_img[h_idx * 360:(h_idx + 1) * 360,
                          w_idx * 640:(w_idx + 1) * 640] = \
                              vis_tracking_multimodal_pts(database,
                                                          topk_match_pts_list,
                                                          None,
                                                          sel_time,
                                                          self.curr_obs_torch[f'mask'].detach().cpu().numpy(),
                                                          view_idx=view_i)

            cv2.imshow('merge_img', merge_img)
            partial_vid.write(merge_img)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        partial_vid.release()
    
    def smooth_tracking(self,
                        src_feat_info,
                        last_match_pts_list,
                        boundaries,
                        rand_ptcl_num):
        kernel_sigma = 0.05
        lr = 0.001
        iter_num = 100
        reg_w = 1.0
        dist_w = 0.01

        src_feats = [src_feat_info[k]['src_feats'] for k in src_feat_info.keys()]
        src_feats = torch.cat(src_feats, dim=0) # [num_instance * rand_ptcl_num, feat_dim]
        
        num_instance = len(last_match_pts_list)
        last_match_pts_np = np.concatenate(last_match_pts_list, axis=0) # [num_instance * rand_ptcl_num, 3]
        assert last_match_pts_np.shape[0] == num_instance * rand_ptcl_num
        
        last_match_pts_tensor = torch.from_numpy(last_match_pts_np).to(self.device, dtype=torch.float32)
        W = torch.zeros((num_instance * rand_ptcl_num, 3)).to(self.device, dtype=torch.float32)
        W.requires_grad_()
        K = torch.zeros((num_instance * rand_ptcl_num, num_instance * rand_ptcl_num)).to(self.device, dtype=torch.float32)
        for i in range(num_instance):
            key = list(src_feat_info.keys())[i]
            src_pts = src_feat_info[key]['src_pts'] # [rand_ptcl_num, 3]
            src_pts = torch.from_numpy(src_pts).to(self.device, dtype=torch.float32)
            self_dist = torch.norm(src_pts.unsqueeze(0) - src_pts.unsqueeze(1), dim=-1) # [rand_ptcl_num, rand_ptcl_num]
            K[i * rand_ptcl_num:(i + 1) * rand_ptcl_num, i * rand_ptcl_num:(i + 1) * rand_ptcl_num] = \
                torch.exp(-self_dist / kernel_sigma)
        
        optimizer = torch.optim.Adam([W], lr=lr, betas=(0.9, 0.999))
        
        loss_list = []
        feat_loss_list = []
        dist_loss_list = []
        reg_loss_list = []
        for iter_idx in range(iter_num):
            curr_match_pts = K @ W + last_match_pts_tensor
            out = self.eval(curr_match_pts, return_names=['dino_feats'])
            curr_feats = out['dino_feats'] # [num_instance * rand_ptcl_num, feat_dim]
            curr_dist = out['dist'] # [num_instance * rand_ptcl_num]
            feat_loss = torch.norm(curr_feats - src_feats, dim=-1).mean()
            dist_loss = dist_w * torch.clamp(curr_dist, min=0).mean()
            reg_loss = reg_w * torch.norm(W)
            loss = feat_loss + dist_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss for debug
            loss_list.append(loss.item())
            feat_loss_list.append(feat_loss.item())
            dist_loss_list.append(dist_loss.item())
            reg_loss_list.append(reg_loss.item())
            
        # plt.plot(loss_list, label='loss')
        # plt.plot(feat_loss_list, label='feat_loss')
        # plt.plot(dist_loss_list, label='dist_loss')
        # plt.plot(reg_loss_list, label='reg_loss')
        # plt.legend()
        # plt.show()
        
        match_pts_list = []
        for i in range(num_instance):
            match_pts_list.append(curr_match_pts[i * rand_ptcl_num:(i + 1) * rand_ptcl_num].detach().cpu().numpy())
        
        return {'match_pts_list': match_pts_list,}
    
    def rigid_tracking(self,
                       src_feat_info,
                       last_match_pts_list,
                       boundaries,
                       rand_ptcl_num):
        lr = 0.01
        iter_num = 100
        reg_w = 1.0
        dist_w = 100.0
        oob_w = 0.0

        src_feats = [src_feat_info[k]['src_feats'] for k in src_feat_info.keys()]
        src_feats = torch.cat(src_feats, dim=0) # [num_instance * rand_ptcl_num, feat_dim]
        
        num_instance = len(last_match_pts_list)
        last_match_pts_np = np.stack(last_match_pts_list, axis=0) # [num_instance, rand_ptcl_num, 3]
        assert last_match_pts_np.shape[:2] == (num_instance, rand_ptcl_num)
        
        last_match_pts_tensor = torch.from_numpy(last_match_pts_np).to(self.device, dtype=torch.float32)
        from pytorch3d.transforms import Transform3d
        from pytorch3d.transforms.so3 import so3_exp_map
        # T = torch.zeros((num_instance, 3, 4)).to(self.device, dtype=torch.float32)
        # T.requires_grad_()
        # T = SE3.Identity(1).to(self.device, dtype=torch.float32)
        t_params = torch.zeros(num_instance, 3, requires_grad=True)
        # r_params = torch.randn(num_instance, 3, 3, requires_grad=True)
        # r_params = torch.eye(3).repeat(num_instance, 1, 1).to(self.device, dtype=torch.float32).requires_grad_()
        log_r_params = torch.zeros(num_instance, 3, requires_grad=True)
        optimizer = torch.optim.Adam([t_params, log_r_params], lr=lr, betas=(0.9, 0.999))
        
        loss_list = []
        feat_loss_list = []
        dist_loss_list = []
        reg_loss_list = []
        oob_loss_list = []
        for iter_idx in range(iter_num):
            # last_match_pts_tensor_homo = torch.cat([last_match_pts_tensor, torch.ones((num_instance, rand_ptcl_num, 1)).to(self.device, dtype=torch.float32)], dim=-1).permute(0, 2, 1) # [num_instance, 4, rand_ptcl_num]
            # curr_match_pts = torch.bmm(T, last_match_pts_tensor_homo).permute(0, 2, 1) # [num_instance, rand_ptcl_num, 3]
            # curr_match_pts = curr_match_pts.reshape(-1, 3)
            r_params = so3_exp_map(log_r_params)
            T = Transform3d(device=self.device, dtype=torch.float32).rotate(r_params).translate(t_params)
            curr_match_pts = T.transform_points(last_match_pts_tensor).reshape(-1, 3)
            out = self.eval(curr_match_pts, return_names=['dino_feats'])
            curr_feats = out['dino_feats'] # [num_instance * rand_ptcl_num, feat_dim]
            curr_dist = out['dist'] * out['valid_mask'] # [num_instance * rand_ptcl_num]
            feat_loss = (torch.norm((curr_feats - src_feats), dim=-1) * out['valid_mask']).mean()
            dist_loss = dist_w * torch.clamp(curr_dist, min=0).mean()
            reg_loss = reg_w * (torch.norm(t_params) + torch.norm(log_r_params))
            oob_loss = oob_w * (torch.clamp(curr_match_pts[:, 0] - boundaries['x_upper'], min=0) +\
                    torch.clamp(boundaries['x_lower'] - curr_match_pts[:, 0], min=0) +\
                    torch.clamp(curr_match_pts[:, 1] - boundaries['y_upper'], min=0) +\
                    torch.clamp(boundaries['y_lower'] - curr_match_pts[:, 1], min=0) +\
                    torch.clamp(curr_match_pts[:, 2] - boundaries['z_upper'], min=0) +\
                    torch.clamp(boundaries['z_lower'] - curr_match_pts[:, 2], min=0)).mean()
            loss = feat_loss + dist_loss + reg_loss # + oob_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss for debug
            loss_list.append(loss.item())
            feat_loss_list.append(feat_loss.item())
            dist_loss_list.append(dist_loss.item())
            reg_loss_list.append(reg_loss.item())
            oob_loss_list.append(oob_loss.item())
            
        # plt.plot(loss_list, label='loss')
        # plt.plot(feat_loss_list, label='feat_loss')
        # plt.plot(dist_loss_list, label='dist_loss')
        # plt.plot(reg_loss_list, label='reg_loss')
        # plt.legend()
        # plt.show()
        
        match_pts_list = []
        for i in range(num_instance):
            match_pts_list.append(curr_match_pts[i * rand_ptcl_num:(i + 1) * rand_ptcl_num].detach().cpu().numpy())
        
        return {'match_pts_list': match_pts_list,}
    
if __name__ == '__main__':
    torch.cuda.set_device(0)
    fusion = Fusion(num_cam=4, feat_backbone='dinov2')
    
    