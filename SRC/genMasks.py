        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:07:20 2021

@author: alan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import cv2
from functools import reduce
import operator
import math
from scipy.spatial import distance as dist
from scipy.interpolate import interp1d
import matplotlib
import glob
import os 


#IMPORTING DATA 

e = sys.float_info.epsilon
#paths = glob.glob('/home/alan/Documents/finalProject/EchoNet-Dynamic/Videos/*.avi')
#files = [os.path.basename(x) for x in paths]
df = pd.read_csv('/home/alan/Documents/finalProject/EchoNet-Dynamic/VolumeTracings.csv')
files = df ['FileName'].values
files = np.unique(files)
ixTrac = df['Frame'].values [df['FileName'].values == files[0]]
N = len(files)
ls = [0,-1]          #indice diastole y sistole
state = ['diastole', 'sistole']
#EXTRACTING FRAMES

for i in range (N):
    for j in range (len(ls)):
        echoPath = '/home/alan/Documents/finalProject/EchoNet-Dynamic/Videos/' + files [i]
        cap= cv2.VideoCapture(echoPath)  
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frames.append(frame)                    
        cap.release()
        if len (frames) != 0:
            ixF = df['FileName'].values == files[i]
            ixD = df['Frame'].values == df['Frame'].values[ixF][ls[j]]
            ixFD = ixF * ixD
            frame = frames[ df['Frame'].values[ixF][ls[j]]]
            
            #ORDENANDO COORDENADAS VECINAS
    
            x = np.array(np.concatenate([df['X1'].values [ixFD],df['X2'].values [ixFD]]) ) 
            y = np.array(np.concatenate([df['Y1'].values [ixFD],df['Y2'].values [ixFD]]) ) 
            
            coords = tuple(zip(x, y)) 
            center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
            coordSorted = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
            x = np.array( list(zip(*coordSorted))[0])
            y = np.array( list(zip(*coordSorted))[1])
            
            #INTERPOLACION 
            
            f = interp1d(np.arange(0,len(x)), x, kind='cubic')
            x = f(np.linspace(0, len(x)-1,len(x)*5))
            f = interp1d(np.arange(0,len(y)), y, kind='cubic')
            y = f(np.linspace(0, len(y)-1,len(y)*5))
            ixumbral = ((x >= 0) * (x < frame.shape[1]-1)) * ((y >= 0) * (y < frame.shape[0]-1))
            x = x [ixumbral]
            y = y[ixumbral]
    
            #plt.scatter(x,y,c='r')
            
            polygon = np.array([x, y]).T
            
            uL = math.floor(min(x))
            uR = math.ceil(max(x))+1
            lL =  math.floor(min(y))
            lR = math.ceil(max(y))+1
            
            xbox = list(range( uL, uR ))
            
            ybox = list(range(lL , lR))
            
            
            # EXTRAYENDO MASCARA
            
            points = np.array([xbox * len(ybox), [ybox[i]  for i in range (len(ybox)) for j in range(len(xbox))]]).T
            path = matplotlib.path.Path(polygon)
            mask = path.contains_points(points)
            mask.shape = (len(ybox), len(xbox)) 
            heart =   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask2 = np.zeros((heart.shape[0],heart.shape[1]))
            mask2 [lL:lR,uL:uR] = mask
            #plt.imshow(heart)
            #plt.figure()
            #plt.imshow(mask2)
            #plt.figure()
            mask2 [mask2 == 1] = 2
            mask2 [mask2 == 0] = 1
            cv2.imwrite('/home/alan/Documents/finalProject/EchoNet-Dynamic/masks/' + state[j]+'{}'.format(i) + '.png', mask2)
            cv2.imwrite('/home/alan/Documents/finalProject/EchoNet-Dynamic/frames/' + state[j]+'{}'.format(i) + '.png', frame)
        


