# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# This script use a single image batch, if you want to train with multi image per batch check this script:
# https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

TORCH_CUDNN_SDPA_ENABLED=1

import numpy as np
import torch
import cv2
import os

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from tqdm import tqdm
from torch import optim
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import torch.nn as nn
import log
import torchvision
from einops import rearrange


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

        self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8)

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
        query, key, value = heatmap, convex_hull, convex_hull
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
                'convex_hull': convex_hull, 'convex_hull_vis': convex_hull_vis}

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Set training parameters
predictor.model.sam_prompt_encoder.mask_downscaling = fuse_model().cuda()
predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

'''
#The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
predictor.model.image_encoder.train(True)
#Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
'''

optimizer=torch.optim.Adam(params=predictor.model.parameters(),lr=5e-5, weight_decay=0.0002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, min_lr=1e-7, verbose=True)

epochs = 100

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
            'frame': os.path.join(frame_dir, frame),
            'fixation': os.path.join(fixation_dir, fixation),
            'convex_hull': os.path.join(convex_hull_dir, convex_hull),
            'heatmap': os.path.join(heatmap_dir, heatmap),
            'mask': os.path.join(mask_dir, mask)
        }
    )


np.random.seed(100)
np.random.shuffle(data_sample)

# Create res directory
res_dir =  './training_info'
res_dir = os.path.join(res_dir, 'exp_6', str(datetime.datetime.now()).replace(' ', '_').split('.')[0].replace(':', '_'))
print('Model save in {}'.format(res_dir))

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

train_path = os.path.join(res_dir, 'train_vis')
if not os.path.exists(train_path):
    os.makedirs(train_path)

# Create params folder
val_path = os.path.join(res_dir, 'val_vis')
if not os.path.exists(val_path):
    os.makedirs(val_path)

logger = log.get_logger(os.path.join(res_dir, 'log.txt'))

training_data_sample = data_sample[:int(len(data_sample)*0.8)]
validation_data_sample = data_sample[int(len(data_sample)*0.8):]

batch_size = 4

train_gaze_dataset = gaze_dataset(training_data_sample)
train_dataloader = data.DataLoader(train_gaze_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
n_train = len(train_gaze_dataset)

val_gaze_dataset = gaze_dataset(validation_data_sample)
val_dataloader = data.DataLoader(val_gaze_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2, drop_last=True)
n_val = len(val_gaze_dataset)

mask_label = torch.ones((batch_size, 1), dtype=torch.float32).cuda()

tolerance = 0
best_miou = 0
for epoch in range(epochs):
    train_loss = []
    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
    train_miou = []
    with tqdm(total=int(n_train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for idx, d in enumerate(train_dataloader):
            image = d['image']
            ann_map = d['ann_map'].cuda().float()
            new_fixation = d['new_fixation'].cuda()
            convex_hull = d['convex_hull'].cuda().float()
            convex_hull_vis = d['convex_hull_vis']
            heatmap = d['heatmap'].cuda().float()

            optimizer.zero_grad()

            image_batch = [img.detach().cpu().numpy() for img in image]
            convex_hull_vis = convex_hull_vis.detach().cpu().numpy()

            predictor.set_image_batch(image_batch) # apply SAM image encoder to the image

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
            if idx%10==0:
                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(image_batch[0][:, :, 0]*0.5 + prd_mask_vis*255*0.5)
                plt.scatter(new_fixation[0][0][0][0].item(), new_fixation[0][0][0][1].item(), c='r', s=10)
                plt.axis('off')
                plt.subplot(2, 2, 2)
                plt.imshow(convex_hull_vis[0])
                plt.scatter(new_fixation[0][0][0][0].item(), new_fixation[0][0][0][1].item(), c='r', s=10)
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.imshow(ann_map[0][0].detach().cpu().numpy()*255)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(train_path, 'prd_mask_{}.jpg'.format(idx)), dpi=300)
                plt.close()

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            if torch.isnan(iou).any():
                iou = torch.where(torch.isnan(iou), torch.ones_like(iou), iou) # Replace nan with 0
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.1

            iou = np.mean(iou.cpu().detach().numpy())
            train_miou += [iou]

            # apply back propogation
            predictor.model.zero_grad() # empty gradient
            loss.backward()
            optimizer.step()

            train_loss += [loss.item()]

            pbar.update(labels.shape[0])
            pbar.set_postfix(**{'total_loss': loss.item(), 'iou': iou})

    predictor.model.sam_mask_decoder.train(False) # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(False) # enable training of prompt encoder
    val_loss = []
    val_miou = []
    with tqdm(total=int(n_val), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for idx, d in enumerate(val_dataloader):
            image = d['image']
            ann_map = d['ann_map'].cuda().float()
            new_fixation = d['new_fixation'].cuda()
            convex_hull = d['convex_hull'].cuda().float()
            convex_hull_vis = d['convex_hull_vis']
            heatmap = d['heatmap'].cuda().float()

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
            if idx%10==0:
                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(image_batch[0][:, :, 0]*0.5 + prd_mask_vis*255*0.5)
                plt.scatter(new_fixation[0][0][0][0].item(), new_fixation[0][0][0][1].item(), c='r', s=10)
                plt.axis('off')
                plt.subplot(2, 2, 2)
                plt.imshow(convex_hull_vis[0])
                plt.scatter(new_fixation[0][0][0][0].item(), new_fixation[0][0][0][1].item(), c='r', s=10)
                plt.axis('off')
                plt.subplot(2, 2, 3)
                plt.imshow(ann_map[0][0].detach().cpu().numpy()*255)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(val_path, 'prd_mask_{}.jpg'.format(idx)), dpi=300)
                plt.close()

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            if torch.isnan(iou).any():
                iou = torch.where(torch.isnan(iou), torch.ones_like(iou), iou) # Replace nan with 0
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.1

            iou = np.mean(iou.cpu().detach().numpy())
            val_miou += [iou]

            val_loss += [loss.item()]

            pbar.update(labels.shape[0])
            pbar.set_postfix(**{'total_loss': loss.item(), 'iou': iou})

    logger.info('epoch: %f, train_iou: %f, val_iou: %f' %
            (epoch,
            np.mean(np.array(train_miou)),
            np.mean(np.array(val_miou))))

    ave_val_miou = np.mean(np.array(val_miou))
    scheduler.step(ave_val_miou)
    if ave_val_miou > best_miou:
        print("Last miou", best_miou, "current miou", ave_val_miou)
        best_miou = ave_val_miou
        torch.save(predictor.model.state_dict(), os.path.join(res_dir, 'best_weight_{}.torch'.format(epoch)))
        torch.save(predictor.model.state_dict(), os.path.join(res_dir, 'best_weight.torch'.format(epoch)))

        print("save model")
        tolerance = 0
    else:
        tolerance += 1
        print('Results not improve: {}'.format(tolerance))
        if tolerance > 10:
            print("Early stop")
            break
