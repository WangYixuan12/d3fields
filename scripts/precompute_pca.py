import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.append(os.getcwd())

# patch_h = 224 // 14
# patch_w = 224 // 14
patch_h = 75
patch_w = 75
# feat_dim = 384 # vits14
# feat_dim = 768 # vitb14
feat_dim = 1024 # vitl14
# feat_dim = 1536 # vitg14

obj_type = 'mug'

device = 'cuda'

img_paths = [f'data/dino_pca/{obj_type}/0.png',
             f'data/dino_pca/{obj_type}/1.png',
             f'data/dino_pca/{obj_type}/2.png',
             f'data/dino_pca/{obj_type}/3.png',]

transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)

print(model)

features = torch.zeros(4, patch_h * patch_w, feat_dim).to(device)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).to(device)
for i in range(4):
    img_path = img_paths[i]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (14 * patch_w, 14 * patch_h))
    # img = Image.open(img_path).convert('RGB')
    img = Image.fromarray(img)
    imgs_tensor[i] = transform(img)[:3]
with torch.no_grad():
    features_dict = model.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize

features = features.reshape(4 * patch_h * patch_w, feat_dim).detach().cpu().numpy()

fg_seg = PCA(n_components=3)
fg_seg.fit(features)
pca_features = fg_seg.transform(features)

plt.subplot(1, 3, 1)
plt.hist(pca_features[:, 0])
plt.subplot(1, 3, 2)
plt.hist(pca_features[:, 1])
plt.subplot(1, 3, 3)
plt.hist(pca_features[:, 2])
plt.show()
plt.close()

pca_features_bg = pca_features[:, 0] > -15
pca_features_fg = ~pca_features_bg

# plot the pca_feature_bg
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
plt.show()

pca = PCA(n_components=3)
fg_features = features[pca_features_fg]
num_fg_list = []
for i in range(4):
    num_fg_list.append(pca_features_fg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].sum())
pca.fit(fg_features)
pca_features_rem = pca.transform(fg_features)

# save pca model
import pickle
with open(f'pca_model/{obj_type}.pkl', 'wb') as f:
    pickle.dump(pca, f)

for i in range(3):
    start_idx = 0
    end_idx = num_fg_list[0]
    for j in range(4):
        print('min: ', pca_features_rem[start_idx:end_idx, i].min())
        print('max: ', pca_features_rem[start_idx:end_idx, i].max())
        pca_features_rem[start_idx:end_idx, i] = \
            (pca_features_rem[start_idx:end_idx, i] - pca_features_rem[start_idx:end_idx, i].min()) /\
                (pca_features_rem[start_idx:end_idx, i].max() - pca_features_rem[start_idx:end_idx, i].min())
        start_idx = end_idx
        end_idx += num_fg_list[j+1] if j < 3 else 0

pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_bg] = 0
pca_features_rgb[pca_features_fg] = pca_features_rem

pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(pca_features_rgb[i][..., ::-1])
plt.show()
plt.close()
