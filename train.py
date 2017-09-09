from Tiramisu import Tiramisu
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

from keras import callbacks
import math
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose

K.set_image_dim_ordering('tf')

import cv2
import numpy as np
import json


# load train data
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')

test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_label.npy')

tiramisu = Tiramisu()
model = tiramisu.model

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
optimizer = Adam(lr=1e-3, decay=0.995)

tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=True)
filepath="weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False, mode='max')

callbacks_list = [checkpoint]

nb_epoch = 150
batch_size = 2

history = tiramisu.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_list, class_weight=class_weighting,verbose=1, validation_data=(test_data, test_label), shuffle=True)

tiramisu.save_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay{}.hdf5'.format(nb_epoch))

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


