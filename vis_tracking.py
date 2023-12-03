import os
import sys
sys.path.append(os.getcwd())
import copy
import time

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import cm
import pickle

from fusion import Fusion
from utils.draw_utils import aggr_point_cloud_from_data
from utils.track_vis import TrackVis

fusion = Fusion(num_cam=4)

num_cam = 4

x_upper = 0.4
x_lower = -0.4
y_upper = 0.3
y_lower = -0.4
z_upper = 0.02
z_lower = -0.3

boundaries = {'x_lower': x_lower,
              'x_upper': x_upper,
              'y_lower': y_lower,
              'y_upper': y_upper,
              'z_lower': z_lower,
              'z_upper': z_upper,}

kypts_boundaries = {'x_lower': x_lower,
                    'x_upper': x_upper,
                    'y_lower': y_lower,
                    'y_upper': y_upper,
                    'z_lower': -0.2,
                    'z_upper': -0.02,}

vis_o3d = True

def gen_dense_kypts(data_path, src_feat_info):
    colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'0.png')) for i in range(num_cam)], axis=0)# [N, H, W, C]
    depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'0.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000. # [N, H, W]

    extrinsics = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy')) for i in range(num_cam)])
    cam_param = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy')) for i in range(num_cam)])
    intrinsics = np.zeros((num_cam, 3, 3))
    intrinsics[:, 0, 0] = cam_param[:, 0]
    intrinsics[:, 1, 1] = cam_param[:, 1]
    intrinsics[:, 0, 2] = cam_param[:, 2]
    intrinsics[:, 1, 2] = cam_param[:, 3]
    intrinsics[:, 2, 2] = 1

    # multi-category tracking
    query_texts = list(src_feat_info.keys())
    query_thresholds = [src_feat_info[k]['params']['sam_threshold'] for k in query_texts]

    # create output dir
    full_pts_path = os.path.join(data_path, 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num
    os.system(f'mkdir -p {full_pts_path}')
    
    track_vis = TrackVis(poses=extrinsics, Ks=intrinsics, output_dir=full_pts_path, vis_o3d=vis_o3d)
    
    time_skip = 1
    times = list(range(len(os.listdir(os.path.join(data_path, f'camera_0', 'color')))))
    
    for t in tqdm(times[::time_skip]):
        colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png')) for i in range(num_cam)], axis=0) # [N, H, W, C]
        depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000. # [N, H, W]

        if vis_o3d:
            pcd = aggr_point_cloud_from_data(colors[..., ::-1], depths, intrinsics, extrinsics, downsample=True, boundaries=boundaries)
        else:
            pcd = None
        
        obs = {
            'color': colors,
            'depth': depths,
            'pose': extrinsics[:, :3], # (N, 3, 4)
            'K': intrinsics,
        }
        
        fusion.update(obs)
        fusion.text_queries_for_inst_mask(query_texts, query_thresholds, boundaries=boundaries)
        
        # initialize for tracking
        if t == 0:
            rand_ptcl_num = 100
            src_feats_list, src_pts_list, color_list = fusion.select_features_rand(kypts_boundaries, rand_ptcl_num, per_instance=True)
            
            # save src_feats_list and src_pts_list
            src_feats_np_list = [src_feats.detach().cpu().numpy() for src_feats in src_feats_list]
            src_pts_np_list = [src_pts for src_pts in src_pts_list]
            pickle.dump(src_feats_np_list, open(os.path.join(full_pts_path, f'src_feats_list.pkl'), 'wb'))
            pickle.dump(src_pts_np_list, open(os.path.join(full_pts_path, f'src_pts_list.pkl'), 'wb'))
            
            # save make label
            pickle.dump(fusion.curr_obs_torch['mask_label'][0], open(os.path.join(full_pts_path, f'mask_label.pkl'), 'wb'))
            
            last_k = ""
            rep_idx = 0
            for k_i, k in enumerate(fusion.curr_obs_torch['mask_label'][0][1:]):
                if k == last_k:
                    rep_idx += 1
                    src_feat_info[k+f'_{rep_idx}'] = copy.copy(src_feat_info[k])
                    src_feat_info[k+f'_{rep_idx}']['src_feats'] = src_feats_list[k_i]
                    src_feat_info[k+f'_{rep_idx}']['src_color'] = color_list[k_i]
                    src_feat_info[k+f'_{rep_idx}']['src_pts'] = src_pts_list[k_i]
                    src_feat_loc_norm = (src_feat_info[k]['src_pts'][:, 0] - src_feat_info[k]['src_pts'][:, 0].min()) / \
                        (src_feat_info[k]['src_pts'][:, 0].max() - src_feat_info[k]['src_pts'][:, 0].min())
                    cmap = cm.get_cmap('viridis')
                    colors = (cmap(src_feat_loc_norm)[:, :3] * 255).astype(np.uint8)[:, ::-1]
                    src_feat_info[k+f'_{rep_idx}']['src_pts_color'] = colors
                else:
                    rep_idx = 0
                    src_feat_info[k]['src_feats'] = src_feats_list[k_i]
                    src_feat_info[k]['src_color'] = color_list[k_i]
                    src_feat_info[k]['src_pts'] = src_pts_list[k_i]
                    src_feat_loc_norm = (src_feat_info[k]['src_pts'][:, 0] - src_feat_info[k]['src_pts'][:, 0].min()) / \
                        (src_feat_info[k]['src_pts'][:, 0].max() - src_feat_info[k]['src_pts'][:, 0].min())
                    cmap = cm.get_cmap('viridis')
                    colors = (cmap(src_feat_loc_norm)[:, :3] * 255).astype(np.uint8)[:, ::-1]
                    src_feat_info[k]['src_pts_color'] = colors
                last_k = k
            match_pts_list = src_pts_list.copy()
        
        tracking_res = fusion.rigid_tracking(src_feat_info,
                                            match_pts_list,
                                            boundaries,
                                            rand_ptcl_num)
        match_pts_list = tracking_res['match_pts_list']
        
        track_vis.visualize_match_pts(match_pts_list, pcd, obs['color'][..., ::-1], src_feat_info)
        
        pickle.dump(match_pts_list, open(os.path.join(full_pts_path, f'{t:06d}.pkl'), 'wb'))

if __name__ == '__main__':
    src_feat_info = {
        'shoe':
            {'params': {'sam_threshold': 0.6},
            'src_feats_path': None},
    }
    gen_dense_kypts('data/2023-09-14-17-06-38-562096', src_feat_info)
