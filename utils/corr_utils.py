import numpy as np
import torch

def compute_similarity(src_feat_map, tgt_feat, scale, dist_type='l2'):
    # :param src_feat_map: [B, H, W, C] numpy array
    # :param tgt_feat: [C] numpy array
    # :param scale: float
    # :param dist_type: str ('l2', 'square')
    # :return: [B, H, W] numpy array
    assert src_feat_map.shape[-1] == tgt_feat.shape[0]
    if dist_type == 'square':
        feat_dist = np.sum((src_feat_map - tgt_feat[None, None, None, :]) ** 2, axis=-1)
    elif dist_type == 'l2':
        feat_dist = np.linalg.norm(src_feat_map - tgt_feat[None, None, None, :], axis=-1)
    else:
        raise NotImplementedError
    feat_dist = np.exp(-feat_dist * scale)
    assert feat_dist.shape == src_feat_map.shape[:3]
    return feat_dist

def compute_similarity_tensor(src_feat_map, tgt_feat, scale, dist_type='l2'):
    # :param src_feat_map: [B, C, **dim] torch tensor
    # :param tgt_feat: [C] torch tensor
    # :param scale: float
    # :param dist_type: str ('l2', 'square')
    # :return: [B, **dim] torch tensor
    assert src_feat_map.shape[1] == tgt_feat.shape[0]
    tgt_feat = tgt_feat.unsqueeze(0)
    dim = len(src_feat_map.shape)
    for _ in range(dim - 2):
        tgt_feat = tgt_feat.unsqueeze(-1)
    if dist_type == 'square':
        feat_dist = torch.sum((src_feat_map - tgt_feat) ** 2, dim=1)
    elif dist_type == 'l2':
        feat_dist = torch.norm(src_feat_map - tgt_feat, dim=1)
    else:
        raise NotImplementedError
    # feat_dist = torch.exp(-feat_dist * scale)
    feat_dist = torch.softmax(-feat_dist * scale, dim=0)
    # assert feat_dist.shape[1:] == src_feat_map.shape[2:]
    assert feat_dist.shape[0] == src_feat_map.shape[0]
    return feat_dist

def compute_dist_tensor(src_feat_map, tgt_feat, dist_type='l2'):
    # :param src_feat_map: [B, C, **dim] torch tensor
    # :param tgt_feat: [C] torch tensor
    # :param scale: float
    # :param dist_type: str ('l2', 'square')
    # :return: [B, **dim] torch tensor
    assert src_feat_map.shape[1] == tgt_feat.shape[0]
    tgt_feat = tgt_feat.unsqueeze(0)
    dim = len(src_feat_map.shape)
    for _ in range(dim - 2):
        tgt_feat = tgt_feat.unsqueeze(-1)
    if dist_type == 'square':
        feat_dist = torch.sum((src_feat_map - tgt_feat) ** 2, dim=1)
    elif dist_type == 'l2':
        feat_dist = torch.norm(src_feat_map - tgt_feat, dim=1)
    else:
        raise NotImplementedError
    return feat_dist

def compute_similarity_tensor_multi(src_feat_map, tgt_feats, src_pts, last_match_pts, scale, dist_type='l2'):
    # :param src_feat_map: [B1, C] torch tensor
    # :param tgt_feat: [B2, C] torch tensor
    # :param src_pts: [B1, 3] torch tensor
    # :param last_match_pts: [B2, 3] torch tensor
    # :param scale: float
    # :param dist_type: str ('l2', 'square')
    # :return: feat_dist [B1, B2] torch tensor, where feat_dist.sum(dim=0)=1
    assert src_feat_map.shape[1] == tgt_feats.shape[1]
    assert len(src_feat_map.shape) == 2
    assert len(tgt_feats.shape) == 2
    tgt_feats = tgt_feats.unsqueeze(0)
    src_feat_map = src_feat_map.unsqueeze(1)
    # dim = len(src_feat_map.shape)
    # for _ in range(dim - 3):
    #     tgt_feats = tgt_feats.unsqueeze(-1)
    if dist_type == 'square':
        feat_dist = torch.sum((src_feat_map - tgt_feats) ** 2, dim=2)
    elif dist_type == 'l2':
        try:
            feat_dist = torch.norm(src_feat_map - tgt_feats, dim=2)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # batchrize the computation
                batchsize = 5000
                for i in range(0, src_feat_map.shape[0], batchsize):
                    upper = min(i + batchsize, src_feat_map.shape[0])
                    feat_dist_batch = torch.norm(src_feat_map[i:upper] - tgt_feats, dim=2)
                    if i == 0:
                        feat_dist = feat_dist_batch
                    else:
                        feat_dist = torch.cat([feat_dist, feat_dist_batch], dim=0)
    else:
        raise NotImplementedError
    # time_reg = 10 * torch.norm(src_pts.unsqueeze(1) - last_match_pts.unsqueeze(0), dim=2)
    # print('reg norm: ', time_reg.norm())
    # print('feat_dist norm: ', feat_dist.norm())
    # feat_dist += time_reg # [B1, B2]
    # feat_dist = torch.exp(-feat_dist * scale)
    feat_dist = torch.softmax(-feat_dist * scale, dim=0)
    # assert feat_dist.shape[2:] == src_feat_map.shape[2:]
    assert feat_dist.shape[0] == src_feat_map.shape[0]
    assert feat_dist.shape[1] == tgt_feats.shape[1]
    return feat_dist
