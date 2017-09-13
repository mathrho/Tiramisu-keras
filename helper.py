import cv2
import numpy as np

from helper import *
import os

"""
Preprocessing images helpers
"""

def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]= b
    norm[:,:,1]= g
    norm[:,:,2]= r
    mean = norm.mean()
    std = norm.std()
    norm -= mean
    norm /= std

    return norm

def one_hot_it(labels,w,h):
    x = np.zeros([w,h,12])
    for i in range(w):
        for j in range(h):
            x[i,j,labels[i][j]]=1
    return x
