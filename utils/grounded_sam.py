import argparse
import os
import copy
from typing import List

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import Model as GroundingDINOModel
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor

# segment anything
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.my_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def transform_image(image_np):
    image_pil = Image.fromarray(image_np) # load image
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    # logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    # boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def get_grounding_output_batch_captions(model, image, captions, box_thresholds, text_threshold, with_logits=True, device="cpu"):
    for i, caption in enumerate(captions):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions[i] = caption
    num_captions = len(captions)
    model = model.to(device)
    image = image.to(device)
    image = image[None].repeat(num_captions, 1, 1, 1)
    with torch.no_grad():
        outputs = model(image, captions=captions)
    logits = outputs["pred_logits"].sigmoid()  # (num_captions, nq, 256)
    boxes = outputs["pred_boxes"]  # (num_captions, nq, 4)

    # filter output
    logits_filt_list = []
    boxes_filt_list = []
    for i in range(num_captions):
        logits_filt = logits[i].clone()
        boxes_filt = boxes[i].clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_thresholds[i]
        logits_filt = logits_filt[filt_mask] # num_filt, 256
        boxes_filt = boxes_filt[filt_mask] # num_filt, 4
        logits_filt_list.append(logits_filt)
        boxes_filt_list.append(boxes_filt)

    return boxes_filt_list, logits_filt_list

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    
def grounded_sam(image_path, text_prompt, dino_model, sam_model, box_threshold):
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # load image
    image_pil, image = load_image(image_path)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device=device, dtype=boxes_filt.dtype)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # vectorize the operation above
    # boxes_filt = boxes_filt * torch.Tensor([[W, H, W, H]])
    # boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    # boxes_filt[:, 2:] += boxes_filt[:, :2]

    # boxes_filt = boxes_filt.cpu()
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.detach().cpu().numpy(), plt.gca(), label)

    save_fn = get_current_YYYY_MM_DD_hh_mm_ss_ms()+str(np.random.randint(9999))+".jpg"
    plt.axis('off')
    plt.savefig(
        save_fn, 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    mask_viz = cv2.imread(save_fn)
    os.system(f"rm {save_fn}")

    return masks[0, 0].detach().cpu().numpy(), mask_viz

def grounded_sam_batch_queries(image_path, text_prompts, dino_model, sam_model, box_thresholds):
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # load image
    image_pil, image = load_image(image_path)

    # run grounding dino model
    boxes_filt_list = get_grounding_output_batch_captions(
        dino_model, image, text_prompts, box_thresholds, text_threshold, device=device
    )

    # initialize SAM
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for boxes_filt in boxes_filt_list:
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device=device, dtype=boxes_filt.dtype)
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # vectorize the operation above
    # boxes_filt = boxes_filt * torch.Tensor([[W, H, W, H]])
    # boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    # boxes_filt[:, 2:] += boxes_filt[:, :2]

    # boxes_filt = boxes_filt.cpu()
    boxes_filt_total = torch.zeros(0, 4).to(device=device, dtype=torch.float32)
    ending_idx = torch.zeros(len(boxes_filt_list)+1, dtype=torch.int64).to(device=device)
    for i, boxes_filt in enumerate(boxes_filt_list):
        if boxes_filt.size(0) > 0:
            boxes_filt_total = torch.cat([boxes_filt_total, boxes_filt], dim=0)
        ending_idx[i+1] = boxes_filt_total.size(0)
    if boxes_filt_total.size(0) == 0:
        return np.zeros((len(boxes_filt_list), H, W), dtype=bool)
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt_total, image.shape[:2]) # [num_total_boxes, 4]

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    ) # [num_total_boxes, 1, H, W]
    
    final_masks = torch.zeros((len(boxes_filt_list), H, W), device=device, dtype=torch.float32)
    for i in range(len(boxes_filt_list)):
        if ending_idx[i+1] > ending_idx[i]:
            final_masks[i] = torch.clamp_max(masks[ending_idx[i]:ending_idx[i+1]].sum(dim=0), 1.0)[0]
    final_masks = (final_masks > 0.5)

    return final_masks.detach().cpu().numpy()

def grounded_instance_sam(image_path, text_prompt, dino_model, sam_model, box_thresholds):
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # load image
    image_pil, image = load_image(image_path)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, image, text_prompt, box_thresholds, text_threshold, device=device
    )

    # initialize SAM
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    # for boxes_filt in boxes_filt_list:
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device=device, dtype=boxes_filt.dtype)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # vectorize the operation above
    # boxes_filt = boxes_filt * torch.Tensor([[W, H, W, H]])
    # boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    # boxes_filt[:, 2:] += boxes_filt[:, :2]

    # boxes_filt = boxes_filt.cpu()
    
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    
    masks = masks[:, 0, :, :].detach().cpu().numpy()
    
    aggr_mask = np.zeros(masks[0].shape)
    for obj_i in range(masks.shape[0]):
        aggr_mask[masks[obj_i]] = obj_i + 1
    aggr_mask = aggr_mask.astype(np.uint8)

    return aggr_mask

def grounded_instance_sam_np(image, text_prompt, dino_model, sam_model, box_thresholds):
    # :param image_tensor: [3, H, W]
    assert len(image.shape) == 3
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # load image
    _, image_tensor = transform_image(image)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, image_tensor, text_prompt, box_thresholds, text_threshold, device=device
    )

    # initialize SAM
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    H, W = image_tensor.shape[1:]
    # for boxes_filt in boxes_filt_list:
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device=device, dtype=boxes_filt.dtype)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # vectorize the operation above
    # boxes_filt = boxes_filt * torch.Tensor([[W, H, W, H]])
    # boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    # boxes_filt[:, 2:] += boxes_filt[:, :2]

    # boxes_filt = boxes_filt.cpu()
    
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

    if len(boxes_filt) == 0:
        return torch.zeros((H, W), device=device, dtype=torch.float32)
    
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )

    masks = masks[:, 0, :, :] # [num_total_boxes, H, W]
    
    aggr_mask = torch.zeros(masks[0].shape).to(device=device, dtype=torch.uint8)
    for obj_i in range(masks.shape[0]):
        aggr_mask[masks[obj_i]] = obj_i + 1

    return aggr_mask

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def grounded_instance_sam_new_ver(image,
                                  text_prompts,
                                  dino_model : GroundingDINOModel,
                                  sam_model : SamPredictor,
                                  box_thresholds,
                                  merge_all=False,
                                  device="cuda"):
    # :param image: [H, W, 3] BGR
    assert len(image.shape) == 3
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # detect objects
    detections = dino_model.predict_with_classes(
        image=image,
        # classes=enhance_class_name(class_names=text_prompts),
        classes=text_prompts,
        box_threshold=box_thresholds[0],
        text_threshold=text_threshold,
    )
    
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_model,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    labels = ['background']
    for query_i in detections.class_id.tolist():
        labels.append(text_prompts[query_i])
    
    # add detections mask for background
    bg_mask = ~np.bitwise_or.reduce(detections.mask, axis=0)
    bg_conf = 1.0
    detections.mask = np.concatenate([np.expand_dims(bg_mask, axis=0), detections.mask], axis=0)
    detections.confidence = np.concatenate([np.array([bg_conf]), detections.confidence], axis=0)

    return detections.mask, labels, detections.confidence

def grounded_instance_sam_bacth_queries_np(image, text_prompts, dino_model, sam_model, box_thresholds, merge_all=False):
    # :param image: [H, W, 3] RGB
    assert len(image.shape) == 3
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    # load image
    _, image_tensor = transform_image(image)

    # run grounding dino model
    boxes_filt_list, logits_filt_list = get_grounding_output_batch_captions(
        dino_model, image_tensor, text_prompts, box_thresholds, text_threshold, device=device
    )

    # initialize SAM
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    H, W = image.shape[:2]
    for boxes_filt in boxes_filt_list:
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device=device, dtype=boxes_filt.dtype)
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # vectorize the operation above
    # boxes_filt = boxes_filt * torch.Tensor([[W, H, W, H]])
    # boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    # boxes_filt[:, 2:] += boxes_filt[:, :2]

    # boxes_filt = boxes_filt.cpu()
    
    boxes_filt_total = torch.zeros(0, 4).to(device=device, dtype=torch.float32)
    ending_idx = torch.zeros(len(boxes_filt_list)+1, dtype=torch.int64).to(device=device)
    for i, boxes_filt in enumerate(boxes_filt_list):
        if boxes_filt.size(0) > 0:
            boxes_filt_total = torch.cat([boxes_filt_total, boxes_filt], dim=0)
        ending_idx[i+1] = boxes_filt_total.size(0)
    if boxes_filt_total.size(0) == 0:
        return torch.zeros((H, W), dtype=torch.bool, device=device), ['background']
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt_total, image.shape[:2]) # [num_total_boxes, 4]
    
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )

    masks = masks[:, 0, :, :] # [num_total_boxes, H, W]
    labels = ['background']
    for query_i in range(len(boxes_filt_list)):
        labels = labels + [text_prompts[query_i][:-1]] * (ending_idx[query_i+1] - ending_idx[query_i])
    
    # remove masks where IoU are large
    logits_filt_list_cat = torch.cat(logits_filt_list, dim=0)
    num_masks = masks.shape[0]
    to_rm = []
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
            if IoU > 0.9:
                if logits_filt_list_cat[i].max().item() > logits_filt_list_cat[j].max().item():
                    to_rm.append(j)
                else:
                    to_rm.append(i)
    to_rm = np.unique(to_rm)
    to_keep = np.setdiff1d(np.arange(num_masks), to_rm)
    to_keep = torch.from_numpy(to_keep).to(device=device, dtype=torch.int64)
    masks = masks[to_keep]
    labels = [labels[i + 1] for i in to_keep]
    labels.insert(0, 'background')
    
    aggr_mask = torch.zeros(masks[0].shape).to(device=device, dtype=torch.uint8)
    for obj_i in range(masks.shape[0]):
        aggr_mask[masks[obj_i]] = obj_i + 1
    if merge_all:
        try:
            assert len(text_prompts) == 1
        except:
            raise ValueError("Only one text prompt is allowed when merge_all is True.")
        aggr_mask[aggr_mask != 0] = 1
        labels = ['background', text_prompts[0][:-1]]

    return aggr_mask, labels

def grounded_sam_batch_queries_np(image, text_prompts, dino_model, sam_model, box_thresholds):
    # :param image_tensor: [3, H, W]
    assert len(image.shape) == 3
    # cfg
    text_threshold = 0.25
    device = "cuda:0"

    _, image_tensor = transform_image(image)

    # run grounding dino model
    boxes_filt_list = get_grounding_output_batch_captions(
        dino_model, image_tensor, text_prompts, box_thresholds, text_threshold, device=device
    )

    # initialize SAM
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    sam_model.set_image(image)

    H, W = image_tensor.shape[1:]
    for boxes_filt in boxes_filt_list:
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device=device, dtype=boxes_filt.dtype)
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
    
    # vectorize the operation above
    # boxes_filt = boxes_filt * torch.Tensor([[W, H, W, H]])
    # boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    # boxes_filt[:, 2:] += boxes_filt[:, :2]

    # boxes_filt = boxes_filt.cpu()
    boxes_filt_total = torch.zeros(0, 4).to(device=device, dtype=torch.float32)
    ending_idx = torch.zeros(len(boxes_filt_list)+1, dtype=torch.int64).to(device=device)
    for i, boxes_filt in enumerate(boxes_filt_list):
        if boxes_filt.size(0) > 0:
            boxes_filt_total = torch.cat([boxes_filt_total, boxes_filt], dim=0)
        ending_idx[i+1] = boxes_filt_total.size(0)
    if boxes_filt_total.size(0) == 0:
        return torch.zeros((len(boxes_filt_list), H, W), dtype=torch.bool, device=device)
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt_total, image.shape[:2]) # [num_total_boxes, 4]

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    ) # [num_total_boxes, 1, H, W]
    
    final_masks = torch.zeros((len(boxes_filt_list), H, W), device=device, dtype=torch.float32)
    for i in range(len(boxes_filt_list)):
        if ending_idx[i+1] > ending_idx[i]:
            final_masks[i] = torch.clamp_max(masks[ending_idx[i]:ending_idx[i+1]].sum(dim=0), 1.0)[0]
    final_masks = (final_masks > 0.5)

    return final_masks
