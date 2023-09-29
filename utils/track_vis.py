import os
import numpy as np
import cv2
from matplotlib import cm
import open3d as o3d

from utils.draw_utils import draw_keypoints

def vis_tracking_multimodal_pts(img, K, pose, match_pts_list, preset_colors = None):
    # :param match_pts_list: list of [num_pts, 3]
    # :mask: [num_view, H, W, NQ] numpy array
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    for i, match_pts in enumerate(match_pts_list):
        # topk = min(5,match_pts.shape[0])
        # conf = conf_list[i]
        # topk_conf_idx = np.argpartition(conf, -topk)[-topk:]
        num_pts = match_pts.shape[0]
        # colors = color_cands[:num_pts]
        cmap = cm.get_cmap('viridis')
        if preset_colors is None:
            colors = (cmap(np.linspace(0, 1, num_pts))[:, :3] * 255).astype(np.int32)[::-1, ::-1]
        else:
            colors = preset_colors
        match_pts = np.concatenate([match_pts, np.ones([num_pts, 1])], axis=-1) # [num_pts, 4]
        match_pts = np.matmul(pose, match_pts.T)[:3].T # [num_pts, 3]
        
        match_pts_2d = match_pts[:, :2] / match_pts[:, 2:] # [num_pts, 2]
        match_pts_2d[:, 0] = match_pts_2d[:, 0] * fx + cx
        match_pts_2d[:, 1] = match_pts_2d[:, 1] * fy + cy
        
        match_pts_2d = match_pts_2d.astype(np.int32)
        match_pts_2d = match_pts_2d.reshape(num_pts, 2)
        # img = draw_keypoints(img, match_pts_2d[topk_conf_idx], colors[topk_conf_idx], radius=5)
        img = draw_keypoints(img, match_pts_2d, colors, radius=5)
    
    return img

class TrackVis():
    def __init__(self, poses, Ks, output_dir, vis_o3d=True):
        self.poses = poses
        self.Ks = Ks
        self.output_dir = output_dir
        self.vis_o3d = vis_o3d
        self.t = 0
        self.num_cam = poses.shape[0]
        
        # vis utils
        self.cmap = cm.get_cmap('viridis')
        
        if vis_o3d:
            self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Visualizer")
            self.vis.add_geometry(self.origin)
            
            self.pcl = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcl)
            
            self.o3d_vid_path = os.path.join(self.output_dir, 'o3d.mp4')
            self.o3d_path = os.path.join(self.output_dir, 'o3d')
            os.system(f'mkdir -p {self.o3d_path}')
        
        self.vid_path = os.path.join(self.output_dir, 'tracking.mp4')
        self.imshow_ratio = 0.7
    
    def visualize_match_pts(self, match_pts_list, full_pcd, colors, track_info):
        # :param match_pts_list: list of (ptcl_num, 3) np.ndarray
        # :param full_pcd: open3d point cloud
        # :return: None
        # visualize the match_pts_list
        self.ptcl_num = match_pts_list[0].shape[0]
        instance_num = len(match_pts_list)
        if self.vis_o3d:
            self.pcl.points = full_pcd.points
            self.pcl.colors = full_pcd.colors
            self.vis.update_geometry(self.pcl)
        
            if self.t == 0:
                # append sphere to visualize
                self.sphere_list = []
                for k_i, k in enumerate(track_info.keys()):
                    for sphere_i in range(self.ptcl_num):
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=10, create_uv_map=True)
                        sphere.compute_vertex_normals()
                        sphere.paint_uniform_color(track_info[k]['src_pts_color'][sphere_i, ::-1] / 255.0)
                        self.base_sphere_vertex = np.array(sphere.vertices)
                        self.vis.add_geometry(sphere)
                        self.sphere_list.append(sphere)
                    
            # update sphere
            for sphere_i in range(len(self.sphere_list)):
                self.sphere_list[sphere_i].vertices = o3d.utility.Vector3dVector(self.base_sphere_vertex + match_pts_list[sphere_i // self.ptcl_num][sphere_i % self.ptcl_num])
                self.vis.update_geometry(self.sphere_list[sphere_i])
            
            view_control = self.vis.get_view_control()
            view_control.set_front([ -0.3091336095239805, 0.10370358297217445, -0.94534754367978802 ])
            view_control.set_lookat([ 0.10592095570082928, 0.039109756311030912, 0.1852527727811189 ])
            view_control.set_up([ -0.022245326124056605, 0.9929763673752956, 0.11620275082714933 ])
            view_control.set_zoom(1.6800000000000008)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            self.vis.run()
            self.vis.capture_screen_image(f"{self.o3d_path}/{self.t:06}.png")
        
            if self.t == 0:
                o3d_img = cv2.imread(f"{self.o3d_path}/{self.t:06}.png")
                self.o3d_vid = cv2.VideoWriter(self.o3d_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (o3d_img.shape[1], o3d_img.shape[0]))
                self.o3d_vid.write(o3d_img)
            else:
                self.o3d_vid.write(cv2.imread(f"{self.o3d_path}/{self.t:06}.png"))
        
        self.H, self.W = colors.shape[1:3]
        if self.t == 0:
            self.vid = cv2.VideoWriter(self.vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(self.W * self.imshow_ratio) * 2, int(self.H * self.imshow_ratio) * 2))
        merge_img = np.zeros((int(self.H * self.imshow_ratio) * 2, int(self.W * self.imshow_ratio) * 2, 3), dtype=np.uint8)
        for view_i in range(self.num_cam):
            h_idx = view_i // 2
            w_idx = view_i % 2
            vis_img = vis_tracking_multimodal_pts(colors[view_i][..., ::-1],
                                                  self.Ks[view_i],
                                                  self.poses[view_i],
                                                  match_pts_list,
                                                  preset_colors=np.concatenate([track_info[k]['src_pts_color'] for k in track_info.keys()], axis=0),)
            merge_img[h_idx * int(self.H * self.imshow_ratio):(h_idx + 1) * int(self.H * self.imshow_ratio),
                        w_idx * int(self.W * self.imshow_ratio):(w_idx + 1) * int(self.W * self.imshow_ratio)] = \
                            cv2.resize(vis_img, (int(self.W * self.imshow_ratio), int(self.H * self.imshow_ratio)), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('merge_img', merge_img)
        self.vid.write(merge_img)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
        self.t += 1
