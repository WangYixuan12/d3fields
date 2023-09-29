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
