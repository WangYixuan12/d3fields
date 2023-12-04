import os
import sys
sys.path.append(os.getcwd())
import pickle

import cv2
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import torch
import trimesh
import argparse

from fusion import Fusion, create_init_grid
from utils.draw_utils import aggr_point_cloud_from_data

def main(data_path,
         pca_path,
         query_texts,
         query_thresholds,
         x_lower,
         x_upper,
         y_lower,
         y_upper,
         z_lower,
         z_upper,
         num_cam=4,
         t=50,
         step=0.004,):
            
    boundaries = {'x_lower': x_lower,
                'x_upper': x_upper,
                'y_lower': y_lower,
                'y_upper': y_upper,
                'z_lower': z_lower,
                'z_upper': z_upper,}

    pca = pickle.load(open(pca_path, 'rb'))

    fusion = Fusion(num_cam=4, feat_backbone='dinov2')

    colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png')) for i in range(num_cam)], axis=0) # [N, H, W, C]
    depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000. # [N, H, W]

    H, W = colors.shape[1:3]

    extrinsics = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy')) for i in range(num_cam)])
    cam_param = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy')) for i in range(num_cam)])
    intrinsics = np.zeros((num_cam, 3, 3))
    intrinsics[:, 0, 0] = cam_param[:, 0]
    intrinsics[:, 1, 1] = cam_param[:, 1]
    intrinsics[:, 0, 2] = cam_param[:, 2]
    intrinsics[:, 1, 2] = cam_param[:, 3]
    intrinsics[:, 2, 2] = 1

    obs = {
        'color': colors,
        'depth': depths,
        'pose': extrinsics[:, :3], # (N, 3, 4)
        'K': intrinsics,
    }

    pcd = aggr_point_cloud_from_data(colors[..., ::-1], depths, intrinsics, extrinsics, downsample=True, boundaries=boundaries)
    pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)

    fusion.update(obs)
    fusion.text_queries_for_inst_mask_no_track(query_texts, query_thresholds, boundaries=boundaries)

    ### 3D vis
    device = 'cuda'

    # visualize mesh
    init_grid, grid_shape = create_init_grid(boundaries, step)
    init_grid = init_grid.to(device=device, dtype=torch.float32)

    print('eval init grid')
    with torch.no_grad():
        out = fusion.batch_eval(init_grid, return_names=[])

    # extract mesh
    print('extract mesh')
    vertices, triangles = fusion.extract_mesh(init_grid, out, grid_shape)

    # eval mask and feature of vertices
    vertices_tensor = torch.from_numpy(vertices).to(device, dtype=torch.float32)
    print('eval mesh vertices')
    with torch.no_grad():
        out = fusion.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask', 'color_tensor'])

    cam = trimesh.scene.Camera(resolution=(1920, 1043), fov=(60, 60))

    cam_matrix = np.array([[ 0.87490918, -0.24637599,  0.41693261,  0.63666708],
                        [-0.44229374, -0.75717002,  0.4806972,   0.66457463],
                        [ 0.19725663, -0.60497308, -0.77142556, -1.16125645],
                        [ 0.        , -0.        , -0.        ,  1.        ]])

    # create mask mesh
    mask_meshes = fusion.create_instance_mask_mesh(vertices, triangles, out)
    for mask_mesh in mask_meshes:
        mask_scene = trimesh.Scene(mask_mesh, camera=cam, camera_transform=cam_matrix)
        mask_scene.show()

    # create feature mesh
    feature_mesh = fusion.create_descriptor_mesh(vertices, triangles, out, {'pca': pca}, mask_out_bg=True)
    feature_scene = trimesh.Scene(feature_mesh, camera=cam, camera_transform=cam_matrix)
    feature_scene.show()

    # create color mesh
    color_mesh = fusion.create_color_mesh(vertices, triangles, out)
    color_scene = trimesh.Scene(color_mesh, camera=cam, camera_transform=cam_matrix)
    color_scene.show()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, help='path to data')
    argparser.add_argument('--pca_path', type=str, help='path to pca')
    argparser.add_argument('--query_texts', nargs='+', help='query texts')
    argparser.add_argument('--query_thresholds', type=float, nargs='+', help='query thresholds')
    argparser.add_argument('--x_lower', type=float, help='x lower bound')
    argparser.add_argument('--x_upper', type=float, help='x upper bound')
    argparser.add_argument('--y_lower', type=float, help='y lower bound')
    argparser.add_argument('--y_upper', type=float, help='y upper bound')
    argparser.add_argument('--z_lower', type=float, help='z lower bound')
    argparser.add_argument('--z_upper', type=float, help='z upper bound')
    argparser.add_argument('--num_cam', type=int, default=4, help='number of cameras')
    argparser.add_argument('--t', type=int, default=50, help='time step')
    argparser.add_argument('--step', type=float, default=0.004, help='step size')
    args = argparser.parse_args()
    main(**vars(args))
