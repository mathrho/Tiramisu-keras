from keras.models import Model
from keras.layers import Input, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.layers.merge import concatenate
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
import pydot
import graphviz

class Tiramisu(object):
    def __init__(self):
        n_pool = 5
        growth_rate = 16
        layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        input_layer = Input(shape=(224, 224, 3))
        t = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

        #dense block
        nb_features = 48
        skip_connections = []
        for i in range(n_pool):
            for _ in range(layer_per_block[i]):
                tmp = t
                t = BatchNormalization(mode=0, axis=1,
                                        gamma_regularizer=l2(0.0001),
                                        beta_regularizer=l2(0.0001))(t)

                t = Activation('relu')(t)
                t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
                t = Dropout(0.2)(t)
                #t = concatenate([t, tmp])
            skip_connections.append(t)

            t = BatchNormalization(mode=0, axis=1,
                                    gamma_regularizer=l2(0.0001),
                                    beta_regularizer=l2(0.0001))(t)
            t = Activation('relu')(t)
            nb_features += growth_rate * layer_per_block[i]
            t = Conv2D(nb_features, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
            t = Dropout(0.2)(t)
            t = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last')(t)

        for i in range(layer_per_block[5]):
            tmp = t
            t = BatchNormalization(mode=0, axis=1,
                                    gamma_regularizer=l2(0.0001),
                                    beta_regularizer=l2(0.0001))(t)

            t = Activation('relu')(t)
            t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
            t = Dropout(0.2)(t)
            #t = concatenate([t, tmp])

        skip_connections = skip_connections[::-1]

        for i in range(n_pool):
            keep_nb_features = growth_rate * layer_per_block[n_pool + i]
            t = Conv2DTranspose(keep_nb_features, strides=2, kernel_size=(3, 3), padding='same', data_format='channels_last')(t)


            #t = concatenate([t, skip_connections[i]])
            for j in range(layer_per_block[n_pool+i+1]):
                tmp = t
                t = BatchNormalization(mode=0, axis=1,
                                        gamma_regularizer=l2(0.0001),
                                        beta_regularizer=l2(0.0001))(t)

                t = Activation('relu')(t)
                t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
                t = Dropout(0.2)(t)
                #t = concatenate([t, tmp])

        t = Conv2D(12, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
        output_layer = Activation('softmax')(t)
        self.model = Model(inputs=input_layer, outputs=output_layer)


tiramisu = Tiramisu()
model = tiramisu.model
plot_model(model, to_file='model.pdf')
model.summary()
