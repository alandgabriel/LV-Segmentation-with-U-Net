#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 01:30:26 2021

@author: alan
"""


import tensorflow as tf
import glob
import random
import tensorflow.keras.layers as layers
import numpy as np
from skimage.io import imread
import os
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from packaging import version
import datetime
from tensorboard.plugins.hparams import api as hp
import time

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

tf.debugging.set_log_device_placement(True)


#Detecting GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def unet():
  inputs = tf.keras.Input((112, 112, 3))

  # Entry block
  x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for filters in [64, 128, 256]:
      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.SeparableConv2D(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

      # Project residual
      residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
          previous_block_activation
      )
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual

  ### [Second half of the network: upsampling inputs] ###

  for filters in [256, 128, 64, 32]:
      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.Activation("relu")(x)
      x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
      x = layers.BatchNormalization()(x)

      x = layers.UpSampling2D(2)(x)

      # Project residual
      residual = layers.UpSampling2D(2)(previous_block_activation)
      residual = layers.Conv2D(filters, 1, padding="same")(residual)
      x = layers.add([x, residual])  # Add back residual
      previous_block_activation = x  # Set aside next residual

  # Add a per-pixel classification layer
  outputs = layers.Conv2D(1, (1, 1), activation='sigmoid') (x)

  # Define the model
  model = tf.keras.Model(inputs, outputs)
  return model



class data(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = plt.imread(path)
        y = np.zeros((self.batch_size,) + self.img_size , dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = plt.imread(path)
            y[j] = img
        return x, y



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

training_dataset = data(32,(112,112), train_data, train_labels)

val_dataset = data(32,(112,112), test_data, test_labels)


model = unet()
model.compile(optimizer='adam',
               loss='binary_crossentropy',
              metrics=['accuracy'])

'''
tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)
'''

# Train the model, doing validation at the end of each epoch.
epochs = 20
start = time.time()
history = model.fit(training_dataset, epochs=epochs, validation_data=val_dataset)
end = time.time()
elapsed = end-start
#%%

#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('$Model_{Accuracy}$')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('$Model_{Loss}$') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


#%%  Display some results

pred = model.predict(val_dataset)
savePred = [cv2.imwrite('DATA/pred/' +  os.path.basename(test_data[i]),  np.squeeze(np.array(pred[i]>.5, dtype='uint8'),-1) *255) for i in range (len(pred))]
plt.figure()
plt.subplot(121)
plt.imshow(np.squeeze(pred[6,:,:,:],-1) + cv2.cvtColor(plt.imread(test_data[6]), cv2.COLOR_BGR2GRAY))
plt.title('Prediction')
plt.subplot(122)
plt.imshow( cv2.cvtColor(plt.imread(test_data[6]), cv2.COLOR_BGR2GRAY) + plt.imread(test_labels[6]))
plt.title('Ground trouth')

#%% Get metrics for evaluation of segmentation
'''
import seg_metrics.seg_metrics as sg
import csv

csv_file = 'metrics.csv'
pred_path = glob.glob('DATA/pred/*.png')
gdth_path = [glob.glob('DATA/masks/' + os.path.basename(im))[0] for im in pred_path]


metrics = [sg.write_metrics(labels = [255], gdth_path=gdth_path[i], pred_path=pred_path[i], csv_file=csv_file) for i in range(len(pred))]

keys =  list(metrics[0].keys())
keys.remove('filename')

means = [ sum(d[k][0] for d in metrics) / len(metrics) for k in keys]

metrics_mean = dict (zip(keys,means))

with open('metrics_mean.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, metrics_mean.keys())
    w.writeheader()
    w.writerow(metrics_mean)
    '''