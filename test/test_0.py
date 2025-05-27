import numpy as np
import torch
import os
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from collections import defaultdict
import collections
import json

import cv2

import pdb

def eval_iou(iou_list):
    # Set the IoU threshold
    threshold = 0.5

    # Convert IoU values to binary predictions based on threshold
    predictions = [1 if iou > threshold else 0 for iou in iou_list]

    # Calculate accuracy of predictions
    accuracy = sum(predictions) / len(predictions)
    return accuracy

class gaze_dataset(data.Dataset):
    def __init__(self, data_sample):
        self.data_sample = data_sample
        
    def __len__(self):
        return len(self.data_sample)
    
    def __getitem__(self, idx):
        d = self.data_sample[idx]

        fixation = d['fixation']
        fixation = Image.open(fixation)
        fixation = np.array(fixation)
        fixation = np.where(fixation == 255)
        fixation = np.array([list(zip(fixation[1], fixation[0]))])

        frame = d['frame']
        image = Image.open(frame).convert('RGB')
        image = np.array(image)

        ann_map = d['mask']
        ann_map = Image.open(ann_map).convert('L')
        ann_map = np.array(ann_map)
        ann_map = (ann_map/255.).astype(np.uint8)
        ann_map = np.expand_dims(ann_map, axis=0)

        new_fixation = d['fixation']
        new_fixation = Image.open(new_fixation)
        new_fixation = np.array(new_fixation)
        new_fixation = np.where(new_fixation == 255)
        new_fixation = np.array([list(zip(new_fixation[1], new_fixation[0]))])

        return {'image': image, 'ann_map': ann_map, 'new_fixation': new_fixation, 'name': d['file']}

# Load model
checkpoint = "../sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Set training parameters
predictor.model.sam_mask_decoder.train(False) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(False) # enable training of prompt encoder

data_dir = '/mnt/e/AI_project/gaze_rebuttle/6_dataset_full/6_dataset/benchmark/object/1/testing'

# Initialize an empty list to store the data paths
data_sample = []
recordings = os.listdir(data_dir)

frame_dir = os.path.join(data_dir, 'frame')
fixation_dir = os.path.join(data_dir, 'feature_fixation')
mask_dir = os.path.join(data_dir, 'frame_groundtruth')

# List the files in each of the directories
for frame, fixation, mask in zip(
        sorted(os.listdir(frame_dir)), sorted(os.listdir(fixation_dir)),
        sorted(os.listdir(mask_dir))):
    # Append a dictionary with all the paths and class info to the data list
    data_sample.append(
        {
            'file': frame,
            'frame': os.path.join(frame_dir, frame),
            'fixation': os.path.join(fixation_dir, fixation),
            'mask': os.path.join(mask_dir, mask)
        }
    )

test_data_sample = data_sample

batch_size = 1

test_gaze_dataset = gaze_dataset(test_data_sample)
test_dataloader = data.DataLoader(test_gaze_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
n_test = len(test_gaze_dataset)

mask_label = np.ones((batch_size, 1))

tolerance = 0
best_miou = 0

test_miou = []

test_path = "./qualitative/exp_1"
if not os.path.exists(test_path):
    os.makedirs(test_path)

ground_truth_label = []
result = defaultdict(list)
result_acc = defaultdict(list)

iou_list = []
acc_list = []
for idx, d in enumerate(test_dataloader):
    image = d['image']
    ann_map = d['ann_map'].cuda().float()
    new_fixation = d['new_fixation'].cuda()
    frame = d['name'][0]

    image_batch = [img.detach().cpu().numpy() for img in image]

    predictor.set_image_batch(image_batch) # apply SAM image encoder to the image

    with torch.no_grad():
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(new_fixation, mask_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords[:, 0], labels),boxes=None,masks=None,)

        # mask decoder
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=False,repeat_image=False,high_res_features=high_res_features)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

    # Segmentaion Loss caclulation 1, 1, 512, 512
    gt_mask = ann_map[:, 0]
    prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
    prd_mask_vis = prd_mask.clone()
    prd_mask_vis = prd_mask_vis[0].cpu().detach().numpy()
    prd_mask_vis = (prd_mask_vis > 0.5).astype(np.uint8)

    # Prediction
    cv2.imwrite(os.path.join(test_path, frame), (prd_mask_vis*255).astype(np.uint8))

    # Score loss calculation (intersection over union) IOU
    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)

    if (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter).sum() == 0:
        iou = 1
        result[int(gt_mask.sum().item())].append(1)
        result_acc[int(gt_mask.sum().item())].append(1)
        iou_list.append(1)
        acc_list.append(1)
    else:
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        result[int(gt_mask.sum().item())].append(iou[0].item())
        if iou > 0.5:
            result_acc[int(gt_mask.sum().item())].append(1)
        else:
            result_acc[int(gt_mask.sum().item())].append(0)
        iou_list.append(iou[0].item())
        acc_list.append(1 if iou > 0.5 else 0)

result = collections.OrderedDict(sorted(result.items()))
np.savez('results/test0.npz', iou=np.array(iou_list), acc=np.array(acc_list))

pdb.set_trace()

small = 10000

small_result = {key: value for key, value in result.items() if key < small}
large_result =  {key: value for key, value in result.items() if small <= key}

def flatten(lss): 
    return [x for xs in lss for x in xs]

all_iou = np.array(flatten(list(result.values()))).mean()
small_iou = np.array(flatten(list(small_result.values()))).mean()
large_iou = np.array(flatten(list(large_result.values()))).mean()

print('IoU: All: {} Small: {} large: {}'.format(round(all_iou, 2), round(small_iou, 2), round(large_iou, 2)))

small_acc_result = {key: value for key, value in result_acc.items() if key < small}
large_acc_result =  {key: value for key, value in result_acc.items() if small <= key}

all_acc = np.array(flatten(list(result_acc.values()))).mean()
small_acc = np.array(flatten(list(small_acc_result.values()))).mean()
large_acc = np.array(flatten(list(large_acc_result.values()))).mean()

print('Acc: All: {} Small: {} large: {}'.format(round(all_acc, 4), round(small_acc, 4), round(large_acc, 4)))
pdb.set_trace()

# print(test_path)
# pdb.set_trace()

# acc = eval_iou(test_miou)
# print("Accuracy: ", acc)
# print("miou: ", np.mean(test_miou))
# # pdb.set_trace()
