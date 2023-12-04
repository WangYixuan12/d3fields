# D<sup>3</sup>Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xps4Ji_Xl8-riXF9cNYfkfmFiW-UWV0j?usp=sharing)

### [Website](https://robopil.github.io/d3fields/) | [Paper](https://arxiv.org/abs/2309.16118/) | [Colab](https://colab.research.google.com/drive/1xps4Ji_Xl8-riXF9cNYfkfmFiW-UWV0j?usp=sharing)

***D<sup>3</sup>Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation***

<a target="_blank" href="https://wangyixuan12.github.io/">Yixuan Wang</a><sup>1*</sup>,
<a target="_blank" href="https://robopil.github.io/d3fields/">Zhuoran Li</a><sup>2, 3*</sup>,
<a target="_blank" href="https://robo-alex.github.io/">Mingtong Zhang</a><sup>1</sup>,
<a target="_blank" href="https://ece.illinois.edu/about/directory/faculty/krdc">Katherine Driggs-Campbell</a><sup>1</sup>,
<a target="_blank" href="https://jiajunwu.com/">Jiajun Wu</a><sup>2</sup>,
<a target="_blank" href="https://profiles.stanford.edu/fei-fei-li">Li Fei-Fei</a><sup>2</sup>,
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1, 2</sup>
            
<sup>1</sup>University of Illinois Urbana-Champaign,
<sup>2</sup>Stanford University,
<sup>3</sup>National University of Singapore<br>

https://github.com/WangYixuan12/d3fields/assets/32333199/a3fced3d-e827-4e7e-ad6a-e80889809fca

### Try it in Colab!

In this [notebook](https://colab.research.google.com/drive/1xps4Ji_Xl8-riXF9cNYfkfmFiW-UWV0j?usp=sharing), we show how to build D<sup>3</sup>Fields and visualize reconstructed mesh, mask fields, and descriptor fields. We also demonstrate how to track keypoints of a video.

### Installation
```
# create conda environment
conda env create -f env.yaml
conda activate d3fields
pip install -e GroundingDINO/
python scripts/install_pytorch3d.py

# download pretrained models
bash scripts/download_ckpts.sh
bash scripts/download_data.sh
```

### Visualization
```
python vis_repr.py # visualize the representation
python vis_tracking.py # visualize the tracking
```

### Code Explanation
`Fusion` is the core class of D<sup>3</sup>Fields. It contains the following key functions:
- `update`: it takes in the observation and updates the internal states.
- `text_queries_for_inst_mask`: it will query the instance mask according to the text query and thresholds.
- `text_queries_for_inst_mask_no_track`: it is similar to `text_queries_for_inst_mask`, but it will not invoke the underlying [XMem](https://github.com/hkchengrex/XMem) tracking module.
- `eval`: it will evaluate associated features for arbitrary 3D points.
- `batch_eval`: for a large batch of points, it will evaluate them batch by batch to avoid out-of-memory error.
The important attributes of `Fusion` are:
- `curr_obs_torch`: a dictionary containing the following keys:
    - `color`: multiview color images in the format of np.uint8 BGR numpy arrays
    - `color_tensor`: multiview color images in the format of float32 BGR torch tensors
    - `depth`: multiview depth images in the format of np.float32 torch tensors, unit in meters
    - `mask`: multiview instance mask images in the format of np.uint8 torch tensors (V, H, W, num_inst)
    - `consensus_mask_label`: mask labels aggregated from all views in the format of a list of strings.

### Customized Dataset
To run D<sup>3</sup>Fields on your own dataset, you could follow the following steps:
1. Prepare dataset in the following structure:
```
dataset_name
├── camera_0
│   ├── color
|   |   ├── 0.png
|   |   ├── 1.png
|   |   ├── ...
│   ├── depth
|   |   ├── 0.png
|   |   ├── 1.png
|   |   ├── ...
│   ├── camera_extrinsics.npy
│   ├── camera_params.npy
├── camera_1
├── ...
```
The definition of `camera_extrinsics.npy` and `camera_params.npy` is defined as follows:
```
camera_extrinsics.npy: (4, 4) numpy array, the extrinsics of the camera, which transforms a point from world coordinate to camera coordinate
camera_params.npy: (4,) numpy array, the camera parameters in the following order: fx, fy, cx, cy
```
2. Prepare the PCA pickle file for the query texts. Find four images of the queries texts (e.g. mug) with clean bakcground and central objects. Change `obj_type` within `scripts/prepare_pca.py` and run it.
3. Specify the workspace boundary as x_lower, x_upper, y_lower, y_upper, z_lower, z_upper.
4. Run `python vis_repr_custom.py`, such as `python vis_repr_custom.py --data_path data/2023-09-15-13-21-56-171587 --pca_path pca_model/mug.pkl --query_texts mug --query_thresholds 0.3 --x_lower -0.4 --x_upper 0.4 --y_upper 0.3 --y_lower -0.4 --z_upper 0.02 --z_lower -0.2`

Tips for debugging:
- Make sure the transformation is right by visualizing `pcd` within `vis_repr_custom.py` using Open3D.
- If the GPU is out of memory, run `vis_repr_custom.py` with smaller `step`. This will generate a more sparse voxel grid.
- Make sure Grounded SAM outputs reasonable results by checking `curr_obs_torch['mask']` and `curr_obs_torch['consensus_mask_label']` of `Fusion` class.

### Citation

If you find this repo useful for your research, please consider citing the paper
```
@article{wang2023d3fields,
    title={D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation},
    author={Wang, Yixuan and Li, Zhuoran and Zhang, Mingtong and Driggs-Campbell, Katherine and Wu, Jiajun and Fei-Fei, Li and Li, Yunzhu},
    journal={arXiv preprint arXiv:2309.16118},
    year={2023}
}
```
