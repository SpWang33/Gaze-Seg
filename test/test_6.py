import numpy as np
import torch
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils import data
from PIL import Image
import cv2
import torch.nn as nn
from einops import rearrange
import torchvision
from collections import defaultdict
import collections


def eval_iou(iou_list):
    # Set the IoU threshold
    threshold = 0.5

    # Convert IoU values to binary predictions based on threshold
    predictions = [1 if iou > threshold else 0 for iou in iou_list]

    # Calculate accuracy of predictions
    accuracy = sum(predictions) / len(predictions)
    return accuracy

class fuse_model(nn.Module):
    def __init__(self):
        super(fuse_model, self).__init__()
        self.heatmap_model = torchvision.models.resnet34(weights="DEFAULT")
        # extract the feature vector of the last layer of model
        self.heatmap_model = nn.Sequential(*list(self.heatmap_model.children())[:-4])

        self.convex_hull_model = torchvision.models.resnet34(weights="DEFAULT")
        self.convex_hull_model.fc = nn.Identity()
        # extract the feature vector of the last layer of model
        self.convex_hull_model = nn.Sequential(*list(self.convex_hull_model.children())[:-4])

        self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.1)

        self.sigmoid = nn.Sigmoid()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.incrase_dim = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Initialization to prevent gradient vanish or explode
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        heatmap, convex_hull = input
        heatmap = torch.cat([heatmap, heatmap, heatmap], dim=1)
        heatmap = self.heatmap_model(heatmap)

        convex_hull = torch.cat([convex_hull, convex_hull, convex_hull], dim=1)
        convex_hull = self.convex_hull_model(convex_hull)

        bs, n, h, w = convex_hull.size()
        convex_hull = rearrange(convex_hull, 'b n h w -> b (h w) n')
        heatmap = rearrange(heatmap, 'b n h w -> b (h w) n')
        query, key, value = heatmap, heatmap, convex_hull
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = rearrange(attn_output, 'b (h w) n -> b n h w', h=h, w=w)
        attn_output = self.incrase_dim(attn_output)
        fuse_feature = self.upsampling(attn_output)
        return fuse_feature


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

        convex_hull = d['convex_hull']
        convex_hull = Image.open(convex_hull)
        convex_hull = np.array(convex_hull)
        convex_hull_vis = convex_hull.copy()
        convex_hull = convex_hull/255.
        convex_hull = cv2.resize(convex_hull, (256, 256), interpolation=cv2.INTER_NEAREST)
        convex_hull = np.expand_dims(convex_hull, axis=0)

        heatmap = d['heatmap']
        heatmap = Image.open(heatmap).convert('L')
        heatmap = np.array(heatmap)
        heatmap = np.array(heatmap)/255.
        heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_NEAREST) 
        heatmap = np.expand_dims(heatmap, axis=0)

        return {'image': image, 'ann_map': ann_map, 'new_fixation': new_fixation, 'heatmap': heatmap,
                'convex_hull': convex_hull, 'convex_hull_vis': convex_hull_vis, 'name': d['file']}

# Load model
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
fuse_features = fuse_model().cuda()

# Set training parameters
predictor.model.sam_prompt_encoder.mask_downscaling = fuse_model().cuda()
predictor.model.sam_mask_decoder.train(False) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(False) # enable training of prompt encoder

# load weight
trained_weight = ''
predictor.model.load_state_dict(torch.load(trained_weight))

# Add your data directory here
data_dir = ''

# Initialize an empty list to store the data paths
data_sample = []
recordings = os.listdir(data_dir)

frame_dir = os.path.join(data_dir, 'frame')
fixation_dir = os.path.join(data_dir, 'feature_fixation')
convex_hull_dir = os.path.join(data_dir, 'feature_fixationconvexhull_5')
heatmap_dir = os.path.join(data_dir, 'feature_fixationweightedheatmap_5')
mask_dir = os.path.join(data_dir, 'frame_groundtruth')

# List the files in each of the directories
for frame, fixation, convex_hull, heatmap, mask in zip(
        sorted(os.listdir(frame_dir)), sorted(os.listdir(fixation_dir)),
        sorted(os.listdir(convex_hull_dir)), sorted(os.listdir(heatmap_dir)), sorted(os.listdir(mask_dir))):

    # Append a dictionary with all the paths and class info to the data list
    data_sample.append(
        {
            'file': frame,
            'frame': os.path.join(frame_dir, frame),
            'fixation': os.path.join(fixation_dir, fixation),
            'convex_hull': os.path.join(convex_hull_dir, convex_hull),
            'heatmap': os.path.join(heatmap_dir, heatmap),
            'mask': os.path.join(mask_dir, mask)
        }
    )

test_data_sample = data_sample

batch_size = 1

test_gaze_dataset = gaze_dataset(test_data_sample)
test_dataloader = data.DataLoader(test_gaze_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
n_test = len(test_gaze_dataset)

mask_label = np.ones((batch_size, 1))

test_miou = []

test_path = "./qualitative/exp_6"
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
    convex_hull = d['convex_hull'].cuda().float()
    convex_hull_vis = d['convex_hull_vis']
    heatmap = d['heatmap'].cuda().float()
    frame = d['name'][0]

    image_batch = [img.detach().cpu().numpy() for img in image]
    convex_hull_vis = convex_hull_vis.detach().cpu().numpy()

    predictor.set_image_batch(image_batch) # apply SAM image encoder to the image

    with torch.no_grad():
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(new_fixation, mask_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords[:, 0], labels), boxes=None, masks=(heatmap, convex_hull),)

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

    cv2.imwrite(os.path.join(test_path, frame), (prd_mask_vis*255).astype(np.uint8))

    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

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
np.savez('results/test6.npz', iou=np.array(iou_list), acc=np.array(acc_list))

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
