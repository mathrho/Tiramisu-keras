import cv2
import numpy as np

from helper import *
import os

DataPath = './CamVid/'
data_shape = 224*224


def load_data(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd()+txt[i][0][7:])[136:,256:]),2))
        label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[136:,256:][:,:,0],224,224))
    return np.array(data), np.array(label)

train_data, train_label = load_data("train")
test_data, test_label = load_data("test")
val_data, val_label = load_data("val")

train_data = np.transpose(train_data, (0, 2, 3, 1))
train_label = np.transpose(train_label, (0, 1, 2, 3))

test_data = np.transpose(test_data, (0, 2, 3, 1))
test_label = np.transpose(test_label, (0, 1, 2, 3))

val_data = np.transpose(val_data, (0, 2, 3, 1))
val_label = np.transpose(val_label, (0, 1, 2, 3))

print(train_data.shape)

np.save("data/train_data", train_data)
np.save("data/train_label", train_label)

np.save("data/test_data", test_data)
np.save("data/test_label", test_label)

np.save("data/val_data", val_data)
np.save("data/val_label", val_label)

