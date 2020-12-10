'''
sdss_model_constructors - keras sequential API model constructors for use in sdss image CNNs
'''

import keras 
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D, Dense
import numpy as np


def build_baseline_CNN(input_shape):
    '''
    Building convolutional neural network with fixed hyperparameters of similar 
    architecture detailed in D-S 2018.
    https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.3661D/abstract
    
    Depending on how long model takes to train, hyperparameter training 
    will be implemented.
    '''
    model = Sequential()
    
    # 2D convolution with rectified linear unit activation. 
    # 32 convolutional filters, each with a size of 6x6.
    model.add(Conv2D(32, kernel_size=(6, 6), activation='relu', 
                     input_shape=input_shape))
    model.add(Dropout(0.5))
    
    # 2D convolution with rectified linear unit activation. 
    # 64 convolutional filters, each with a size of 5x5.
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    
    # 2D convolution with rectified linear unit activation. 
    # 128 convolutional filters, each with a size of 2x2.
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    
    # 2D convolution with rectified linear unit activation. 
    # 128 convolutional filters, each with a size of 3x3.
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.25))
    
    # 2D convolution with rectified linear unit activation. 
    # 128 convolutional filters, each with a size of 3x3.
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.25))
    
    # Flattening and output dense layers.
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    return model