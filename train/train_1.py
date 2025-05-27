# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# This script use a single image batch, if you want to train with multi image per batch check this script:
# https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import torch
import os

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from tqdm import tqdm
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import log


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

        return {'image': image, 'ann_map': ann_map, 'new_fixation': new_fixation}

# Load model
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Set training parameters
predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

'''
#The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
predictor.model.image_encoder.train(True)
#Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
'''

optimizer=torch.optim.Adam(params=predictor.model.parameters(),lr=1e-5, weight_decay=0)

epochs = 100

# Add your data directory here
data_dir = ''

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
            'frame': os.path.join(frame_dir, frame),
            'fixation': os.path.join(fixation_dir, fixation),
            'mask': os.path.join(mask_dir, mask)
        }
    )

np.random.seed(100)
np.random.shuffle(data_sample)

# Create res directory
res_dir = "../training_info"
res_dir = os.path.join(res_dir, 'exp_1', str(datetime.datetime.now()).replace(' ', '_').split('.')[0].replace(':', '_'))
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
train_dataloader = data.DataLoader(train_gaze_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
n_train = len(train_gaze_dataset)

val_gaze_dataset = gaze_dataset(validation_data_sample)
val_dataloader = data.DataLoader(val_gaze_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
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

            optimizer.zero_grad()

            image_batch = [img.detach().cpu().numpy() for img in image]

            predictor.set_image_batch(image_batch) # apply SAM image encoder to the image

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
            if idx%10==0:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(image_batch[0][:, :, 0]*0.5 + prd_mask_vis*255*0.5)
                plt.scatter(new_fixation[0][0][0][0].item(), new_fixation[0][0][0][1].item(), c='r', s=10)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(ann_map[0][0].detach().cpu().numpy()*255)
                plt.axis('off')
                plt.savefig(os.path.join(train_path, 'prd_mask_{}.jpg'.format(idx)), dpi=300)
                plt.close()

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            if torch.isnan(iou).any():
                iou = torch.where(torch.isnan(iou), torch.ones_like(iou), iou) # Replace nan with 0
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

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

            image_batch = [img.detach().cpu().numpy() for img in image]

            predictor.set_image_batch(image_batch) # apply SAM image encoder to the image

            with torch.no_grad():
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(new_fixation, mask_label, box=None, mask_logits=None, normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords[:, 0], labels),boxes=None, masks=None,)

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
                plt.subplot(1, 2, 1)
                plt.imshow(image_batch[0][:, :, 0]*0.5 + prd_mask_vis*255*0.5)
                plt.scatter(new_fixation[0][0][0][0].item(), new_fixation[0][0][0][1].item(), c='r', s=10)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(ann_map[0][0].detach().cpu().numpy()*255)
                plt.axis('off')
                plt.savefig(os.path.join(val_path, 'prd_mask_{}.jpg'.format(idx)), dpi=300)
                plt.close()

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            if torch.isnan(iou).any():
                iou = torch.where(torch.isnan(iou), torch.ones_like(iou), iou) # Replace nan with 0
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

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
