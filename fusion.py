import os
import sys
sys.path.append('/home/yixuan/general_dp/diffusion_policy/d3fields_dev')
import time
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
from PIL import Image
import open3d as o3d

import groundingdino
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import build_sam, SamPredictor
from XMem.model.network import XMem
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.inference.inference_core import InferenceCore
from XMem.dataset.range_transform import im_normalization
from utils.grounded_sam import grounded_instance_sam_new_ver
from utils.draw_utils import draw_keypoints, aggr_point_cloud_from_data
from utils.my_utils import  depth2fgpcd, fps_np

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
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=pts.dtype)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=pts.dtype)
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

def instance2onehot(instance, N = None):
    # :param instance: [**dim] numpy array uint8, val from 0 to N-1
    # :return: [**dim, N] numpy array bool
    if N is None:
        N = instance.max() + 1
    if type(instance) is np.ndarray:
        assert instance.dtype == np.uint8
        out = np.zeros(instance.shape + (N,), dtype=bool)
        for i in range(N):
            out[..., i] = (instance == i)
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

def _init_low_level_memory(lower_bound, higher_bound, voxel_size, voxel_num):
    def pcd_to_voxel(pcds):
        if type(pcds) == list:
            pcds = np.array(pcds)
        # The pc is in numpy array with shape (..., 3)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.floor((pcds - lower_bound) / voxel_size).astype(np.int32)
        return voxels

    def voxel_to_pcd(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The pc is in numpy array with shape (..., 3)
        pcds = voxels * voxel_size + lower_bound
        return pcds

    def voxel_to_index(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        indexes = (
            voxels[..., 0] * voxel_num[1] * voxel_num[2]
            + voxels[..., 1] * voxel_num[2]
            + voxels[..., 2]
        )
        return indexes

    def index_to_voxel(indexes):
        if type(indexes) == list:
            indexes = np.array(indexes)
        # The index is in numpy array with shape (...,)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.zeros((indexes.shape + (3,)), dtype=np.int32)
        voxels[..., 2] = indexes % voxel_num[2]
        indexes = indexes // voxel_num[2]
        voxels[..., 1] = indexes % voxel_num[1]
        voxels[..., 0] = indexes // voxel_num[1]
        return voxels

    def pcd_to_index(pcds):
        # The pc is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        voxels = pcd_to_voxel(pcds)
        indexes = voxel_to_index(voxels)
        return indexes

    def index_to_pcd(indexes):
        # The index is in numpy array with shape (...,)
        # The pc is in numpy array with shape (..., 3)
        voxels = index_to_voxel(indexes)
        pcds = voxel_to_pcd(voxels)
        return pcds

    return (
        pcd_to_voxel,
        voxel_to_pcd,
        voxel_to_index,
        index_to_voxel,
        pcd_to_index,
        index_to_pcd,
    )

def rm_mask_close_to_pcd(depth, mask, pcd, K, pose):
    # remove the mask that is close to the pcd
    # the mask is in the camera frame with intrinsics K and pose
    # the pcd is in the world frame
    # :param depth: (H, W) numpy array float32
    # :param mask: (H, W) numpy array bool
    # :param pcd: (N, 3) numpy array float32
    # :param K: (3, 3) numpy array float32
    # :param pose: (4, 4) numpy array float32, that transforms points from world frame to camera frame
    # :return: (H, W) numpy array bool
    cam_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    pcd_in_cam = depth2fgpcd(depth=depth, mask=mask, cam_params=cam_params, preserve_zero=True)
    pcd_in_world = np.linalg.inv(pose) @ np.concatenate([pcd_in_cam, np.ones([pcd_in_cam.shape[0], 1])], axis=-1).T # [4, N]
    pcd_in_world = pcd_in_world[:3].T # [N, 3]
    close_mask = np.linalg.norm(pcd_in_world[:, None, :] - pcd[None, ...], axis=-1).min(axis=-1) < 0.02
    mask_idx = np.where(mask)
    filter_mask_idx = (mask_idx[0][close_mask], mask_idx[1][close_mask])
    mask[filter_mask_idx] = False
    return mask

class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
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
        else:
            raise NotImplementedError
        self.dinov2_feat_extractor.eval()
        self.dinov2_feat_extractor.to(dtype=self.dtype)
        
        # load GroundedSAM model
        curr_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(groundingdino.__path__[0], 'config/GroundingDINO_SwinT_OGC.py')
        grounded_checkpoint = os.path.join(curr_path, 'ckpts/groundingdino_swint_ogc.pth')
        # config_file = os.path.join(curr_path, '../gdino_config/GroundingDINO_SwinB.cfg.py')
        # grounded_checkpoint = os.path.join(curr_path, 'ckpts/groundingdino_swinb_cogcoor.pth')
        if not os.path.exists(grounded_checkpoint):
            print('Downloading GroundedSAM model...')
            ckpts_dir = os.path.join(curr_path, 'ckpts')
            os.system(f'mkdir -p {ckpts_dir}')
            # os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth')
            # os.system(f'mv groundingdino_swinb_cogcoor.pth {ckpts_dir}')
            os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
            os.system(f'mv groundingdino_swint_ogc.pth {ckpts_dir}')
        sam_checkpoint = os.path.join(curr_path, 'ckpts/sam_vit_h_4b8939.pth')
        if not os.path.exists(sam_checkpoint):
            print('Downloading SAM model...')
            ckpts_dir = os.path.join(curr_path, 'ckpts')
            os.system(f'mkdir -p {ckpts_dir}')
            os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
            os.system(f'mv sam_vit_h_4b8939.pth {ckpts_dir}')
        self.ground_dino_model = GroundingDINOModel(config_file, grounded_checkpoint, device=self.device)

        self.sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        self.sam_model.model = self.sam_model.model.to(self.device)
        
        # load XMem model
        XMem_path = os.path.join(curr_path, 'XMem/saves/XMem.pth')
        if not os.path.exists(XMem_path):
            print('Downloading XMem model...')
            ckpts_dir = os.path.join(curr_path, 'XMem/saves')
            os.system(f'mkdir -p {ckpts_dir}')
            os.system(f'wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth')
            os.system(f'mv XMem.pth {ckpts_dir}')
        xmem_config = {
            'model': XMem_path,
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
        # inter_normal = interpolate_feats(self.curr_obs_torch['normals'].permute(0,3,1,2),
        #                                  pts_2d,
        #                                 h = self.H,
        #                                 w = self.W,
        #                                 padding_mode='zeros',
        #                                 align_corners=True,
        #                                 inter_mode='bilinear') # [rfn,pn,3]
        
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
        for i in range(0, pts.shape[0], batch_pts):
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
            features_dict = self.dinov2_feat_extractor.forward_features(imgs_tensor.to(dtype=self.dtype))
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
        return out_masks.to(self.device, dtype=self.dtype)
    
    def update(self, obs):
        # :param obs: dict contains:
        #             - 'color': (K, H, W, 3) np array, color image
        #             - 'depth': (K, H, W) np array, depth image
        #             - 'pose': (K, 4, 4) np array, camera pose
        #             - 'K': (K, 3, 3) np array, camera intrinsics
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {
            'patch_h': color.shape[1] // 10,
            'patch_w': color.shape[2] // 10,
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
        self.curr_obs_torch['color'] = color
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=self.dtype) / 255.0
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=self.dtype)
        
        _, self.H, self.W = obs['depth'].shape
    
    def voxel_downsample(self, pcd, voxel_size):
        # :param pcd: [N,3] numpy array
        # :param voxel_size: float
        # :return: [M,3] numpy array
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd_down.points)
    
    def pcd_iou(self, pcd_1, pcd_2, threshold):
        # :param pcd_1 [N,3] numpy array
        # :param pcd_2 [M,3] numpy array
        # voxel downsample
        # voxel_size = threshold
        # pcd_1 = self.voxel_downsample(pcd_1, voxel_size)
        # pcd_2 = self.voxel_downsample(pcd_2, voxel_size)
        dist = np.linalg.norm(pcd_1[:,None] - pcd_2[None], axis=-1) # [N,M]
        min_dist_from_1_to_2 = dist.min(axis=1) # [N]
        min_idx_from_1_to_2 = dist.argmin(axis=1) # [N]
        min_dist_from_2_to_1 = dist.min(axis=0) # [M]
        min_idx_from_2_to_1 = dist.argmin(axis=0) # [M]
        iou = ((min_dist_from_1_to_2 < threshold).sum() + (min_dist_from_2_to_1 < threshold).sum()) / (pcd_1.shape[0] + pcd_2.shape[0])
        iou_1 = (min_dist_from_1_to_2 < threshold).sum() / pcd_1.shape[0]
        iou_2 = (min_dist_from_2_to_1 < threshold).sum() / pcd_2.shape[0]
        overlap_idx_1 = np.where(min_dist_from_1_to_2 < threshold)[0]
        overlap_idx_2 = np.where(min_dist_from_2_to_1 < threshold)[0]
        return iou, iou_1, iou_2, overlap_idx_1, overlap_idx_2, min_idx_from_1_to_2, min_idx_from_2_to_1
    
    def merge_instances_from_new_view(self, instances_info, i, boundaries):
        mask_label = self.curr_obs_torch['mask_label'][i]
        assert mask_label[0] == 'background'
        for j, label in enumerate(mask_label):
            # if j == 0:
            #     continue
            pcd_i = self.extract_masked_pcd_in_views([j], [i], boundaries, downsample=True) # (N,3) in numpy array
            is_new_inst = True
            max_iou = 0
            max_iou_idx = -1
            for k, info in enumerate(instances_info):
                # Check 1: name matches
                if label != info['label']:
                    continue
                
                # Check 2: compute iou
                pcd_inst = np.concatenate([info['pcd'][view_idx] for view_idx in info['pcd'].keys()], axis=0)
                iou, _, _, _, _, _, _ = self.pcd_iou(pcd_i, pcd_inst, threshold=self.iou_threshold)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = k
            
            if max_iou > 0.25:
                is_new_inst = False
            
            # update instances_info
            if is_new_inst and (label != 'background' or i == 0):
                conf = self.curr_obs_torch['mask_conf'][i][j]
                instances_info.append({'label': label,
                                       'pcd': {i: pcd_i},
                                       'conf': {i: conf},
                                       'idx': {i: j}})
            else:
                # Additional Check 1: whether this segmentation is already in the instance
                if i in instances_info[max_iou_idx]['pcd'].keys():
                    # choose the one with higher iou with other pcd
                    other_pcd_ls = [instances_info[max_iou_idx]['pcd'][view_idx] for view_idx in instances_info[max_iou_idx]['pcd'].keys() if view_idx != i]
                    if len(other_pcd_ls) > 0:
                        other_pcd = np.concatenate([instances_info[max_iou_idx]['pcd'][view_idx] for view_idx in instances_info[max_iou_idx]['pcd'].keys() if view_idx != i], axis=0)
                        curr_iou = self.pcd_iou(pcd_i, other_pcd, threshold=self.iou_threshold)[0]
                        prev_pcd = instances_info[max_iou_idx]['pcd'][i]
                        prev_iou = self.pcd_iou(pcd_i, prev_pcd, threshold=self.iou_threshold)[0]
                        if curr_iou <= prev_iou:
                            continue
                
                conf = self.curr_obs_torch['mask_conf'][i][j]
                instances_info[max_iou_idx]['pcd'][i] = pcd_i
                instances_info[max_iou_idx]['conf'][i] = conf
                instances_info[max_iou_idx]['idx'][i] = j
        return instances_info
    
    def vox_idx_iou(self, vox_idx_1, vox_idx_2):
        intersection = len(
            set(vox_idx_1).intersection(set(vox_idx_2))
        )
        union = len(set(vox_idx_1).union(set(vox_idx_2)))
        return intersection / union, len(vox_idx_1) / union, len(vox_idx_2) / union
    
    def merge_instances_from_new_view_vox_ver(self, instances_info, i, boundaries):
        mask_label = self.curr_obs_torch['mask_label'][i]
        assert mask_label[0] == 'background'
        for j, label in enumerate(mask_label):
            # if j == 0:
            #     continue
            pcd_i = self.extract_masked_pcd_in_views([j], [i], boundaries) # (N,3) in numpy array
            index_i = self.pcd_to_index(pcd_i)
            is_new_inst = True
            max_iou = 0
            max_iou_idx = -1
            for k, info in enumerate(instances_info):
                # Check 1: name matches
                if label != info['label']:
                    continue
                
                # Check 2: compute iou
                vox_idx_inst = info['vox_idx']
                iou = self.vox_idx_iou(index_i, vox_idx_inst)[0]
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = k
            
            if max_iou > 0.20:
                is_new_inst = False
            
            # update instances_info
            if is_new_inst and (label != 'background' or i == 0):
                conf = self.curr_obs_torch['mask_conf'][i][j]
                conf_per_pt = {vox_i: [conf] for vox_i in index_i}
                instances_info.append({'label': label,
                                    #    'pcd': {i: pcd_i},
                                       'vox_idx': index_i,
                                       'conf_per_pt': conf_per_pt,
                                       'idx': {i: j}})
            else:
                conf = self.curr_obs_torch['mask_conf'][i][j]
                instances_info[max_iou_idx]['vox_idx'] = np.unique(np.concatenate([instances_info[max_iou_idx]['vox_idx'], index_i]))
                if i in instances_info[max_iou_idx]['idx']:
                    new_vox_idx = set(index_i).difference(set(instances_info[max_iou_idx]['vox_idx']))
                    update_idx = new_vox_idx
                else:
                    update_idx = set(index_i)
                for vox_i in update_idx:
                    if vox_i not in instances_info[max_iou_idx]['conf_per_pt']:
                        instances_info[max_iou_idx]['conf_per_pt'][vox_i] = []
                    instances_info[max_iou_idx]['conf_per_pt'][vox_i].append(conf)
                instances_info[max_iou_idx]['idx'][i] = j
        return instances_info
    
    def del_partial_pcd(self, instance_info, pcd_idx):
        start_idx = 0
        for view_idx in instance_info['pcd'].keys():
            end_idx = start_idx + instance_info['pcd'][view_idx].shape[0]
            pcd_idx_in_view = pcd_idx[np.logical_and(pcd_idx >= start_idx, pcd_idx < end_idx)]
            pcd_idx_in_view -= start_idx
            instance_info['pcd'][view_idx] = np.delete(instance_info['pcd'][view_idx], pcd_idx_in_view, axis=0)
            start_idx = end_idx
        return instance_info

    def del_partial_vox_idx(self, instance_info, vox_idx):
        curr_vox_idx = set(instance_info['vox_idx'])
        for vox_i in vox_idx:
            if vox_i in instance_info['conf_per_pt']:
                del instance_info['conf_per_pt'][vox_i]
            if vox_i in instance_info['vox_idx']:
                curr_vox_idx.remove(vox_i)
        instance_info['vox_idx'] = np.array(list(curr_vox_idx))
        return instance_info
    
    def filter_instances(self, instances_info):
        to_delete = []

        # Filter 1: filter instances that have a large IoU with other instances
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            for idx_j, instance_j in enumerate(instances_info):
                if idx_j <= idx_i:
                    continue
                if idx_j in to_delete:
                    continue
                
                pcd_i = np.concatenate([instance_i['pcd'][view_idx] for view_idx in instance_i['pcd'].keys()], axis=0)
                conf_per_pt_i = np.concatenate([np.ones(instance_i['pcd'][view_idx].shape[0]) * instance_i['conf'][view_idx] for view_idx in instance_i['pcd'].keys()])
                pcd_j = np.concatenate([instance_j['pcd'][view_idx] for view_idx in instance_j['pcd'].keys()], axis=0)
                conf_per_pt_j = np.concatenate([np.ones(instance_j['pcd'][view_idx].shape[0]) * instance_j['conf'][view_idx] for view_idx in instance_j['pcd'].keys()])
                iou, iou_1, iou_2, overlap_idx_1, overlap_idx_2, min_idx_from_1_to_2, min_idx_from_2_to_1 = self.pcd_iou(pcd_i, pcd_j, threshold=0.005)
                if iou > 0.25:
                    # we can only keep one
                    num_vis_view_i = len(instance_i['idx'])
                    num_vis_view_j = len(instance_j['idx'])
                    if num_vis_view_i > num_vis_view_j:
                        to_delete.append(idx_j)
                    elif num_vis_view_j > num_vis_view_i:
                        to_delete.append(idx_i)
                    else:
                        # # ver 1: delete the whole instance
                        # conf_i = np.mean([instance_i['conf'][view_idx] for view_idx in instance_i['conf'].keys()])
                        # conf_j = np.mean([instance_j['conf'][view_idx] for view_idx in instance_j['conf'].keys()])
                        # if conf_i > conf_j:
                        #     to_delete.append(idx_j)
                        # else:
                        #     to_delete.append(idx_i)
                        
                        # ver 2: delete the point with less confidence
                        overlap_conf_1 = conf_per_pt_i[overlap_idx_1]
                        overlap_conf_2_corr_to_1 = conf_per_pt_j[min_idx_from_1_to_2[overlap_idx_1]]
                        overlap_conf_2 = conf_per_pt_j[overlap_idx_2]
                        overlap_conf_1_corr_to_2 = conf_per_pt_i[min_idx_from_2_to_1[overlap_idx_2]]
                        pcd_i_del_idx = overlap_idx_1[overlap_conf_1 < overlap_conf_2_corr_to_1]
                        pcd_j_del_idx = overlap_idx_2[overlap_conf_2 < overlap_conf_1_corr_to_2]
                        
                        # remove points in instance_i
                        instance_i = self.del_partial_pcd(instance_i, pcd_i_del_idx)
                        instance_j = self.del_partial_pcd(instance_j, pcd_j_del_idx)
                
                # instance_i is a subset of instance_j
                elif iou_1 > 0.5:
                    # we may delete instance_i or remove overlapping points in instance_j
                    num_vis_view_i = len(instance_i['idx'])
                    num_vis_view_j = len(instance_j['idx'])
                    if (instance_j['label'] == 'background' and num_vis_view_i < self.num_cam // 2) or \
                        (instance_j['label'] != 'background' and num_vis_view_i < num_vis_view_j // 2):
                        # delete instance_i
                        to_delete.append(idx_i)
                    else:
                        # remove overlapping points in instance_j
                        instance_j = self.del_partial_pcd(instance_j, overlap_idx_2)
                
                # instance_j is a subset of instance_i
                elif iou_2 > 0.5:
                    # we may delete instance_j or remove overlapping points in instance_i
                    num_vis_view_i = len(instance_i['idx'])
                    num_vis_view_j = len(instance_j['idx'])
                    if (instance_i['label'] == 'background' and num_vis_view_j < self.num_cam // 2) or \
                        (instance_i['label'] != 'background' and num_vis_view_j < num_vis_view_i // 2):
                        # delete instance_j
                        to_delete.append(idx_j)
                    else:
                        # remove overlapping points in instance_i
                        instance_i = self.del_partial_pcd(instance_i, overlap_idx_1)
                
                # immediately put instance to delete if it is too small
                pcd_i = np.concatenate([instance_i['pcd'][view_idx] for view_idx in instance_i['pcd'].keys()], axis=0)
                if pcd_i.shape[0] < 10:
                    to_delete.append(idx_i)
                pcd_j = np.concatenate([instance_j['pcd'][view_idx] for view_idx in instance_j['pcd'].keys()], axis=0)
                if pcd_j.shape[0] < 10:
                    to_delete.append(idx_j)

        # Filter 2: filter certain instances used as background
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            bg_name_ls = ['table']
            if instance_i['label'] in bg_name_ls:
                to_delete.append(idx_i)
                continue
        
        # Filter 3: filter instances that are too small
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            pcd_i = np.concatenate([instance_i['pcd'][view_idx] for view_idx in instance_i['pcd'].keys()], axis=0)
            if pcd_i.shape[0] < 10:
                to_delete.append(idx_i)
                continue
        
        for idx in sorted(to_delete, reverse=True):
            del instances_info[idx]
        
        return instances_info
    
    def filter_instances_vox_ver(self, instances_info):
        to_delete = []

        # Filter 1: filter instances that have a large IoU with other instances
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            for idx_j, instance_j in enumerate(instances_info):
                if idx_j <= idx_i:
                    continue
                if idx_j in to_delete:
                    continue
                
                vox_idx_i = instance_i['vox_idx']
                conf_per_pt_i = instance_i['conf_per_pt']
                vox_idx_j = instance_j['vox_idx']
                conf_per_pt_j = instance_j['conf_per_pt']
                iou, iou_1, iou_2 = self.vox_idx_iou(vox_idx_i, vox_idx_j)
                if iou > 0.25 or iou_1 > 0.5 or iou_2 > 0.5:
                    # ver 2: delete the point with less views and less confidence
                    to_delete_i = []
                    for vox_i in conf_per_pt_i.keys():
                        # keep points that are not in vox_idx_j
                        if vox_i not in conf_per_pt_j:
                            continue
                        if len(conf_per_pt_i[vox_i]) < len(conf_per_pt_j[vox_i]):
                            to_delete_i.append(vox_i)
                        elif len(conf_per_pt_i[vox_i]) == len(conf_per_pt_j[vox_i]):
                            if np.mean(conf_per_pt_i[vox_i]) < np.mean(conf_per_pt_j[vox_i]):
                                to_delete_i.append(vox_i)
                    to_delete_j = []
                    for vox_j in conf_per_pt_j.keys():
                        # keep points that are not in vox_idx_i
                        if vox_j not in conf_per_pt_i:
                            continue
                        if len(conf_per_pt_j[vox_j]) < len(conf_per_pt_i[vox_j]):
                            to_delete_j.append(vox_j)
                        elif len(conf_per_pt_j[vox_j]) == len(conf_per_pt_i[vox_j]):
                            if np.mean(conf_per_pt_j[vox_j]) < np.mean(conf_per_pt_i[vox_j]):
                                to_delete_j.append(vox_j)
                    
                    # remove points in instance_i
                    instance_i = self.del_partial_vox_idx(instance_i, to_delete_i)
                    instance_j = self.del_partial_vox_idx(instance_j, to_delete_j)
                
                # immediately put instance to delete if it is too small
                if len(instance_i['vox_idx']) < 1:
                    to_delete.append(idx_i)
                if len(instance_j['vox_idx']) < 1:
                    to_delete.append(idx_j)

        # Filter 2: filter certain instances used as background
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            bg_name_ls = ['table']
            if instance_i['label'] in bg_name_ls:
                to_delete.append(idx_i)
                continue
        
        # Filter 3: filter instances that are too small
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            if len(instance_i['vox_idx']) < 1:
                to_delete.append(idx_i)
                continue
        
        for idx in sorted(to_delete, reverse=True):
            del instances_info[idx]
        
        return instances_info
    
    def reorder_instances(self, instances_info, query_texts):
        # reorder instances based on the query texts
        new_instances_info = []
        for query_text in ['background'] + query_texts:
            for instance_info in instances_info:
                if instance_info['label'] == query_text:
                    new_instances_info.append(instance_info)
        return new_instances_info

    def swap_instance_mask(self, instances_info):
        self.curr_obs_torch['mask'] = torch.zeros((self.num_cam, self.H, self.W), device=self.device, dtype=torch.uint8)
        for i in range(self.num_cam):
            new_mask = np.zeros_like(self.curr_obs_torch['mask_gs'][i][0], dtype=np.uint8)
            for inst_idx, instance_info in enumerate(instances_info):
                if i not in instance_info['idx']:
                    continue
                mask_idx = instance_info['idx'][i]
                new_mask[self.curr_obs_torch['mask_gs'][i][mask_idx]] = inst_idx
            self.curr_obs_torch['mask'][i] = torch.from_numpy(new_mask).to(device=self.device, dtype=torch.uint8)
    
    def align_instance_mask_v3(self, queries, boundaries, expected_labels=None):
        self.iou_threshold = 0.005
        instances_info = []
        # each instance is a dict containing:
        # - 'label': the label of the instance
        # - 'pcd': a dict mapping from view idx to point cloud
        # - 'conf': a dict mapping from view idx to confidence score
        # - 'idx': a dict mapping from view idx to mask index in that view
        lower_bound = np.array([boundaries['x_lower'], boundaries['y_lower'], boundaries['z_lower']])
        higher_bound = np.array([boundaries['x_upper'], boundaries['y_upper'], boundaries['z_upper']])
        voxel_size = 0.03
        self.voxel_num = ((higher_bound - lower_bound) / voxel_size).astype(np.int32)
        (
            self.pcd_to_voxel,
            self.voxel_to_pcd,
            self.voxel_to_index,
            self.index_to_voxel,
            self.pcd_to_index,
            self.index_to_pcd,
        ) = _init_low_level_memory(
            lower_bound, higher_bound, voxel_size, voxel_num=self.voxel_num
        )
        for i in range(self.num_cam):
            instances_info = self.merge_instances_from_new_view_vox_ver(instances_info, i, boundaries)
        instances_info = self.filter_instances_vox_ver(instances_info)
        # TODO: add another function to adjust mask according to removed pcd
        instances_info = self.reorder_instances(instances_info, queries)
        self.swap_instance_mask(instances_info)
        self.curr_obs_torch['consensus_mask_label'] = [instance_info['label'] for instance_info in instances_info]
        if expected_labels is not None and self.curr_obs_torch['consensus_mask_label'] != expected_labels:
            print('consensus mask label', self.curr_obs_torch['consensus_mask_label'])
    
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
    
    def text_queries_for_inst_mask_no_track(self, queries, thresholds, boundaries, merge_all = False, expected_labels=None, robot_pcd=None):
        masks = []
        labels = []
        mask_confs = []
        for i in range(self.num_cam):
            # mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
            mask, label, mask_conf = grounded_instance_sam_new_ver(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all, device=self.device)
            
            # filter out the mask close to robot_pcd
            if robot_pcd is not None:
                to_delete = []
                for inst_i in range(mask.shape[0]):
                    pose_i = self.curr_obs_torch['pose'][i].detach().cpu().numpy() # [3,4]
                    pose_i = np.concatenate([pose_i, np.array([[0,0,0,1]])], axis=0) # [4,4]
                    mask[inst_i] = rm_mask_close_to_pcd(depth=self.curr_obs_torch['depth'][i].detach().cpu().numpy(),
                                            mask=mask[inst_i],
                                            pcd=robot_pcd,
                                            K=self.curr_obs_torch['K'][i].detach().cpu().numpy(),
                                            pose=pose_i,)
                    if mask[inst_i].sum() < 10:
                        to_delete.append(inst_i)
                mask = np.delete(mask, to_delete, axis=0)
                for to_del_i in sorted(to_delete, reverse=True):
                    del label[to_del_i]
                mask_conf = np.delete(mask_conf, to_delete, axis=0)
            
            labels.append(label)
            masks.append(mask)
            mask_confs.append(mask_conf)
        self.curr_obs_torch['mask_gs'] = masks # list of [num_obj, H, W]
        self.curr_obs_torch['mask_label'] = labels # [num_cam, ] list of list
        self.curr_obs_torch['mask_conf'] = mask_confs # [num_cam, ] list of list
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
        self.align_instance_mask_v3(queries, boundaries, expected_labels)
        self.curr_obs_torch[f'mask'] = instance2onehot(self.curr_obs_torch[f'mask'].to(torch.uint8), len(self.curr_obs_torch['consensus_mask_label'])).to(dtype=self.dtype)
    
    def text_queries_for_inst_mask(self, queries, thresholds, boundaries, use_sam=False, merge_all = False, expected_labels=None, robot_pcd=None):
        if 'color' not in self.curr_obs_torch:
            print('Please call update() first!')
            exit()
        
        if not self.xmem_first_mask_loaded:
            masks = []
            labels = []
            mask_confs = []
            for i in range(self.num_cam):
                # mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
                mask, label, mask_conf = grounded_instance_sam_new_ver(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all, device=self.device)
                
                # filter out the mask close to robot_pcd
                if robot_pcd is not None:
                    to_delete = []
                    for inst_i in range(mask.shape[0]):
                        pose_i = self.curr_obs_torch['pose'][i].detach().cpu().numpy() # [3,4]
                        pose_i = np.concatenate([pose_i, np.array([[0,0,0,1]])], axis=0) # [4,4]
                        mask[inst_i] = rm_mask_close_to_pcd(depth=self.curr_obs_torch['depth'][i].detach().cpu().numpy(),
                                             mask=mask[inst_i],
                                             pcd=robot_pcd,
                                             K=self.curr_obs_torch['K'][i].detach().cpu().numpy(),
                                             pose=pose_i,)
                        if mask[inst_i].sum() < 10:
                            to_delete.append(inst_i)
                    mask = np.delete(mask, to_delete, axis=0)
                    for to_del_i in sorted(to_delete, reverse=True):
                        del label[to_del_i]
                    mask_conf = np.delete(mask_conf, to_delete, axis=0)
                
                labels.append(label)
                masks.append(mask)
                mask_confs.append(mask_conf)
            self.curr_obs_torch['mask_gs'] = masks # list of [num_obj, H, W]
            self.curr_obs_torch['mask_label'] = labels # [num_cam, ] list of list
            self.curr_obs_torch['mask_conf'] = mask_confs # [num_cam, ] list of list
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
            self.align_instance_mask_v3(queries, boundaries, expected_labels)
            self.curr_obs_torch[f'mask'] = self.xmem_process(self.curr_obs_torch['color'], self.curr_obs_torch['mask']).to(dtype=self.dtype)
        elif self.xmem_first_mask_loaded and not use_sam:
            self.curr_obs_torch[f'mask'] = self.xmem_process(self.curr_obs_torch['color'], None).to(dtype=self.dtype) # [num_cam, H, W, num_instance]
        elif self.xmem_first_mask_loaded and use_sam:
            raise NotImplementedError
            query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
            mask_confs = []
            for i in range(self.num_cam):
                # mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
                mask, label, mask_conf = grounded_instance_sam_new_ver(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
                query_mask[i] = mask
                mask_confs.append(mask_conf)
                # labels.append(label)
            query_mask = query_mask.to(torch.uint8)
            query_mask = instance2onehot(query_mask, len(self.track_ids)) # [num_cam, H, W, num_instance]
            query_mask = self.align_with_prev_mask(query_mask) # [num_cam, H, W, num_instance]
            query_mask = onehot2instance(query_mask) # [num_cam, H, W]
            self.curr_obs_torch[f'mask'] = self.xmem_process(self.curr_obs_torch['color'], query_mask).to(dtype=self.dtype) # [num_cam, H, W, num_instance]
            self.curr_obs_torch[f'mask_conf'] = mask_confs
            # self.curr_obs_torch[f'mask'] = query_mask # [num_cam, H, W]
    
    def get_inst_num(self):
        # NOTE: including the background
        return len(self.curr_obs_torch['consensus_mask_label'])
    
    def extract_masked_pcd(self, inst_idx_ls, boundaries=None):
        # extract point cloud of the object instance with index inst_idx
        color = self.curr_obs_torch['color']
        depth = self.curr_obs_torch['depth'].detach().cpu().numpy()
        mask = self.curr_obs_torch['mask'].detach().cpu().numpy()
        sel_mask = np.zeros(mask.shape[:3]).astype(bool)
        for inst_idx in inst_idx_ls:
            sel_mask = sel_mask | mask[..., inst_idx].astype(bool)
        for i in range(self.num_cam):
            sel_mask[i] = (cv2.erode((sel_mask[i] * 255).astype(np.uint8), np.ones([2, 2], np.uint8), iterations=1) / 255).astype(bool)
        K = self.curr_obs_torch['K'].detach().cpu().numpy()
        pose = self.curr_obs_torch['pose'].detach().cpu().numpy() # [num_cam, 3, 4]
        pad = np.tile(np.array([[[0,0,0,1]]]), [pose.shape[0], 1, 1])
        pose = np.concatenate([pose, pad], axis=1)
        pcd, _ = aggr_point_cloud_from_data(color, depth, K, pose, downsample=False, masks=sel_mask, out_o3d=False, boundaries=boundaries)
        return pcd
    
    def extract_masked_pcd_in_views(self, inst_idx_ls, view_idx_ls, boundaries, downsample=True):
        assert len(view_idx_ls) == 1
        # extract point cloud of the object instance with index inst_idx
        color = self.curr_obs_torch['color'][view_idx_ls]
        depth = self.curr_obs_torch['depth'].detach().cpu().numpy()[view_idx_ls]
        mask = np.stack([self.curr_obs_torch['mask_gs'][view_idx] for view_idx in view_idx_ls], axis=0).transpose(0, 2, 3, 1) # [num_view, H, W, num_inst]
        K = self.curr_obs_torch['K'].detach().cpu().numpy()[view_idx_ls]
        pose = self.curr_obs_torch['pose'].detach().cpu().numpy()[view_idx_ls]
        # if len(mask.shape) == 3:
        #     mask = instance2onehot(mask.astype(np.uint8))
        sel_mask = np.zeros(mask.shape[:3]).astype(bool)
        for inst_idx in inst_idx_ls:
            sel_mask = sel_mask | mask[..., inst_idx].astype(bool)
        for view_idx in range(sel_mask.shape[0]):
            sel_mask[view_idx] = (cv2.erode((sel_mask[view_idx] * 255).astype(np.uint8), np.ones([2, 2], np.uint8), iterations=1) / 255).astype(bool)
        pad = np.tile(np.array([[[0,0,0,1]]]), [pose.shape[0], 1, 1])
        pose = np.concatenate([pose, pad], axis=1)
        pcd, _ = aggr_point_cloud_from_data(color, depth, K, pose, downsample=downsample, masks=sel_mask, out_o3d=False, boundaries=boundaries)
        return pcd
    
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
    
    def select_features_from_pcd(self, pcd, N, per_instance=False, init_idx = -1, vis = False):
        # pcd: (N, 3) numpy array
        dist_threshold = 0.005
        
        pcd_tensor = torch.from_numpy(pcd).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            out = self.batch_eval(pcd_tensor, return_names=['mask'])
        
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
            masked_pts = pcd_tensor[instance_mask & dist_mask & out['valid_mask']]
            if masked_pts.shape[0] == 0:
                continue
            
            sample_pts, sample_idx, _ = fps_np(masked_pts.detach().cpu().numpy(), N, init_idx=init_idx)
            # src_feats_list.append(out['dino_feats'][sample_idx])
            src_feats_list.append(self.eval(torch.from_numpy(sample_pts).to(self.device, self.dtype))['dino_feats'])
            src_pts_list.append(sample_pts)
            
            last_label = label[i]
            
            if not vis:
                continue
            
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

    def vis_curr_mask(self):
        # return vis_mask, which is a numpy array of shape [num_cam, H, W, 3]. It can be visualized using cv2
        vis_mask = np.zeros((self.num_cam, self.H, self.W, 3))
        for i in range(self.num_cam):
            color = (self.curr_obs_torch['color'][i] * 255).astype(np.uint8)
            mask = self.curr_obs_torch['mask'][i].detach().cpu().numpy() # [H, W, num_instance]
            mask = onehot2instance(mask) # [H, W]
            jet_cmap = cm.get_cmap('jet')
            mask_colors = (jet_cmap(mask / mask.max())[..., :3] * 255).astype(np.uint8)
            vis_mask[i] = (0.5 * color + 0.5 * mask_colors).astype(np.uint8)
        return vis_mask.astype(np.uint8)
    
    def clear_xmem_memory(self):
        for xmem_proc in self.xmem_processors:
            xmem_proc.clear_memory()
        self.xmem_first_mask_loaded = False
    
    def close(self):
        del self.curr_obs_torch
        del self.ground_dino_model
        del self.sam_model
        del self.dinov2_feat_extractor
        if self.feat_backbone != 'dinov2':
            del self.feat_extractor
        for xmem_proc in self.xmem_processors:
            del xmem_proc

def compare_dino_dinov2_time():
    model_type = 'dinov2'
    device = 'cuda'
    fusion = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float16)
    for i in range(11):
        if i == 1:
            start_time = time.time()
        img = (np.random.randn(4, 240, 320, 3) * 255).astype(np.uint8)
        params = {
            'patch_h': img.shape[1] // 10,
            'patch_w': img.shape[2] // 10,
        }
        fusion.extract_features(img, params)
    print(f'{model_type} time: {time.time() - start_time}')

def compare_float_prec():
    model_type = 'dinov2'
    device = 'cuda'
    fusion = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float16)
    img = (np.random.randn(4, 240, 320, 3) * 255).astype(np.uint8)
    params = {
        'patch_h': img.shape[1] // 10,
        'patch_w': img.shape[2] // 10,
    }
    feats = fusion.extract_features(img, params)
    
    fusion_full = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float32)
    feats_full = fusion_full.extract_features(img, params)
    
    print('feat diff: ', torch.norm(feats - feats_full, dim=-1).mean())
    print('feat diff max: ', torch.max(torch.abs(feats - feats_full)))


def test_grounded_sam():
    img = cv2.imread("/home/yixuan/bdai/general_dp/left_bottom_view_color_0.png")
    queries = ["mug", "rack", "table"]
    thresholds = [0.3]
    device = 'cuda'
    curr_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(curr_path, 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
    grounded_checkpoint = os.path.join(curr_path, 'ckpts/groundingdino_swint_ogc.pth')
    sam_checkpoint = os.path.join(curr_path, 'ckpts/sam_vit_h_4b8939.pth')
    if not os.path.exists(grounded_checkpoint):
        print('Downloading GroundedSAM model...')
        ckpts_dir = os.path.join(curr_path, 'ckpts')
        os.system(f'mkdir -p {ckpts_dir}')
        # os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth')
        # os.system(f'mv groundingdino_swinb_cogcoor.pth {ckpts_dir}')
        os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
        os.system(f'mv groundingdino_swint_ogc.pth {ckpts_dir}')
    if not os.path.exists(sam_checkpoint):
        print('Downloading SAM model...')
        ckpts_dir = os.path.join(curr_path, 'ckpts')
        os.system(f'mkdir -p {ckpts_dir}')
        os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
        os.system(f'mv sam_vit_h_4b8939.pth {ckpts_dir}')
    ground_dino_model = GroundingDINOModel(config_file, grounded_checkpoint, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    sam_model.model = sam_model.model.to(device)
    mask, label, mask_conf = grounded_instance_sam_new_ver(img, queries, ground_dino_model, sam_model, thresholds)
    plt.imshow(mask[0])
    plt.show()

if __name__ == '__main__':
    # compare_dino_dinov2_time()
    # compare_float_prec()
    test_grounded_sam()
