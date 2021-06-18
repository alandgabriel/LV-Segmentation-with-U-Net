#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 08:01:42 2021

@author: alan
"""

from unet import UNet
import math
import random
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
import glob
from skimage.io import imread
import tensorflow as tf
from util import enumerateWithEstimate
from torch.optim import SGD, Adam



class segmentationUnet:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = Adam(self.segmentation_model.parameters())

    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=3,
            n_classes=1,   #output channels
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        augmentation_model = SegmentationAugmentation({'flip': True, 'offset': .03, 'scale': .2, 'rotate':True, 'noise': 25.0})


        return segmentation_model, augmentation_model    

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()
    
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
    
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
    
            self.optimizer.step()
    
        self.totalTrainingSamples_count += trnMetrics_g.size(1)
    
        return trnMetrics_g.to('cpu')
    
    def doValidation( self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()
    
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
    
        return valMetrics_g.to('cpu')
    
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup
    
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
    
        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)
    
        prediction_g = self.segmentation_model(input_g)
    
        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)
    
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)
    
        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)
    
            tp = (     predictionBool_g *  label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) *  label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])
    
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp
    
        return diceLoss_g.mean() + fnLoss_g.mean() * 8
    
    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])
    
        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)
    
        return 1 - diceRatio_g
    
    




class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)


    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output

class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:2],
                input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                affine_t, padding_mode='border',
                align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                affine_t, padding_mode='border',
                align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2,i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t




class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
    

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LOSS_NDX = 1
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10
    
#Detecting GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Impoting data

img_files = glob.glob('DATA/frames/*.png')
mask_files = [glob.glob('DATA/masks/' + os.path.basename(im))[0] for im in img_files]
N = len (img_files)

# Spliting data 

ixRand  = list(range(N))
random.shuffle(ixRand)
train_data = [img_files[e] for e in ixRand[:round(N*.8)]]
train_labels = [mask_files[e] for e in ixRand[:round(N*.8)]]

test_data = [img_files[e] for e in ixRand[round(N*.8):]]
test_labels = [mask_files[e] for e in ixRand[round(N*.8):]]

# torch needs that data comes from an instance with getitem and len methods (Map-style datasets)

training_dataset = SegmentationDataSet(inputs=train_data, targets=train_labels, transform=True)
training_dataloader = data.DataLoader(dataset=training_dataset, batch_size=32, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=8)

val_dataset = SegmentationDataSet(inputs=test_data, targets=test_labels, transform=True)
val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=8)


#%%
epochs = 20
best_score = 0.0
validation_cadence = 5
seg = segmentationUnet()
for epoch_ndx in range(1, epochs + 1):
  
    trnMetrics_t = seg.doTraining(epoch_ndx, training_dataloader)

    if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
        # if validation is wanted
        valMetrics_t = seg.doValidation(epoch_ndx, val_dataloader)


#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segmentation_model = UNetWrapper(
    in_channels=3,
    n_classes=1,   #output channels
    depth=3,
    wf=4,
    padding=True,
    batch_norm=True,
    up_mode='upconv',
)
segmentation_model.to(device)