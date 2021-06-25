#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:27:58 2021

@author: alan
"""

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

x = [5,10]
y = [2983.56, 6105.2 ]
f_cpu = interpolate.interp1d(x, y,'slinear')

xn = list(range(5,11,1))
fn = f_cpu(xn)

y2 = [279.15, 569.54]
f_gpu = interpolate.interp1d(x, y2,'slinear')
fn2 = f_gpu(xn)


plt.figure()

plt.plot(xn, fn/fn2, color = 'black')
plt.xlabel('Épocas')
plt.title ('Speed-up en el proceso de entrenamiento')



plt.figure()

plt.plot(xn, fn, color = 'black', label = 'CPU')
plt.plot(xn, fn2, label = 'GPU')
legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
legend.get_frame().set_facecolor('pink') 
plt.xlabel('Épocas')
plt.ylabel ('Tiempo de ejecución (s)')


