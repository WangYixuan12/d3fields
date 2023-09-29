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

from fusion import Fusion, create_init_grid
from utils.draw_utils import aggr_point_cloud_from_data

scene = 'shoe' # 'mug', 'fork', 'shoe'
if scene == 'mug':
    data_path = 'data/2023-09-15-13-21-56-171587' # mug
    pca_path = 'pca_model/mug.pkl'
    query_texts = ['mug']
    query_thresholds = [0.3]
elif scene == 'fork':
    data_path = 'data/2023-09-15-14-15-01-238216' # fork
    pca_path = 'pca_model/fork.pkl'
    query_texts = ['fork']
    query_thresholds = [0.25]
elif scene == 'shoe':
    data_path = 'data/2023-09-11-14-15-50-607452' # shoe
    pca_path = 'pca_model/shoe.pkl'
    query_texts = ['shoe']
    query_thresholds = [0.5]

# hyper-parameter
t = 50
num_cam = 4

step = 0.004

x_upper = 0.4
x_lower = -0.4
y_upper = 0.3
y_lower = -0.4
z_upper = 0.02
z_lower = -0.2
        
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
fusion.text_queries_for_inst_mask_no_track(query_texts, query_thresholds)

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

# create mask mesh
mask_meshes = fusion.create_instance_mask_mesh(vertices, triangles, out)
for mask_mesh in mask_meshes:
    mask_scene = trimesh.Scene(mask_mesh, camera=cam)
    mask_scene.show()

# create feature mesh
feature_mesh = fusion.create_descriptor_mesh(vertices, triangles, out, {'pca': pca}, mask_out_bg=True)
feature_scene = trimesh.Scene(feature_mesh, camera=cam)
feature_scene.show()

# create color mesh
color_mesh = fusion.create_color_mesh(vertices, triangles, out)
color_scene = trimesh.Scene(color_mesh, camera=cam)
color_scene.show()
