#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:18:31 2019

@author: sleek-eagle
"""

import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import numpy as np
from keras.layers.merge import concatenate


NUM_CLASSES = 6

##########
##dnn_1 is the smaller dnn with seperate feature calculations
##########
def get_audio_dnn_1(input_shape):
    data = layers.Input(name='data',shape = input_shape)
    
    dense_1    = layers.Dense(name = 'dense_1', units = input_shape[0], use_bias = True,activation = 'relu')(data)
    bn_1 = layers.BatchNormalization(name = 'dense_1/bn', axis = -1, epsilon = 9.999999747378752e-4, center = True, scale = True)(dense_1)
    relu_1 = layers.Activation(name='bn_1/relu', activation='relu')(bn_1)
    
    dense_2    = layers.Dense(name = 'dense_2', units = 20, use_bias = True,activation = 'relu')(relu_1)
    bn_2 = layers.BatchNormalization(name = 'dense_2/bn', axis = -1, epsilon = 9.999999747378752e-4, center = True, scale = True)(dense_2)
    relu_2 = layers.Activation(name='bn_2/relu', activation='relu')(bn_2)
    
    dense_3    = layers.Dense(name = 'dense_3', units = NUM_CLASSES, use_bias = True,activation = 'relu')(relu_2)
    bn_3 = layers.BatchNormalization(name = 'dense_3/bn', axis = -1, epsilon = 9.999999747378752e-4, center = True, scale = True)(dense_3)
    relu_3 = layers.Activation(name='bn_3/relu', activation='relu')(bn_3)
    
    softmax = layers.Activation(name='relu_3/softmax', activation='softmax')(relu_3)
    
    
    model = Model(inputs = [data], outputs = [softmax])
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    model.summary()
    return model



##############
##audio smaller network with confidence estimation
#############
lmbd = 0.2   
def weightedLoss(yTrue,yPred):
    predTrue = yTrue[:,0:NUM_CLASSES]
    confTrue = yTrue[:,NUM_CLASSES]
    global c;

    predPred = yPred[:,0:NUM_CLASSES]
    confPred = yPred[:,NUM_CLASSES]
    c=K.tf.reshape(confPred,[-1,1])
    
    #interpolated value with c
    confPred_mat = K.tf.tile(c,[1,NUM_CLASSES])
    p_interp = K.tf.multiply(confPred_mat,predPred) + K.tf.multiply((1-confPred_mat),predTrue)
    
    #calc losses
    L_c = -K.log(confPred)
    L_t = K.categorical_crossentropy(predTrue,p_interp)
    
    L = L_t + 0.2* L_c
    return L


def get_audio_dnn_with_confidence(input_shape):

    data = layers.Input(name='data',shape = input_shape)
    dense_1    = layers.Dense(name = 'dense_1', units = input_shape[0], use_bias = True,activation = 'relu')(data)
    bn_1 = layers.BatchNormalization(name = 'dense_1/bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(dense_1)
    relu_1 = layers.Activation(name='bn_1/relu', activation='relu')(bn_1)
    
    dense_2    = layers.Dense(name = 'dense_2', units = 20, use_bias = True,activation = 'relu')(relu_1)
    bn_2 = layers.BatchNormalization(name = 'dense_2/bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(dense_2)
    relu_2 = layers.Activation(name='bn_2/relu', activation='relu')(bn_2)
    
    dense_3    = layers.Dense(name = 'dense_3', units = NUM_CLASSES, use_bias = True,activation = 'relu')(relu_2)
    bn_3 = layers.BatchNormalization(name = 'dense_3/bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(dense_3)
    relu_3 = layers.Activation(name='bn_3/relu', activation='relu')(bn_3)
    
    softmax = layers.Activation(name='relu_3/softmax', activation='softmax')(relu_3)


    #confidence network
    conf_dense_1    = layers.Dense(name = 'conf_dense_1', units = input_shape[0], use_bias = True,activation = 'relu')(data)
    conf_bn_1 = layers.BatchNormalization(name = 'conf_dense_1/bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(conf_dense_1)
    conf_relu_1 = layers.Activation(name='conf_bn_1/relu', activation='relu')(conf_bn_1)
    
    conf_dense_2    = layers.Dense(name = 'conf_dense_2', units = 20, use_bias = True,activation = 'relu')(conf_relu_1)
    conf_bn_2 = layers.BatchNormalization(name = 'conf_dense_2/bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(conf_dense_2)
    conf_relu_2 = layers.Activation(name='conf_bn_2/relu', activation='relu')(conf_bn_2)
    
    confidence = layers.Dense(1, activation='sigmoid',name = 'confidence')(conf_relu_2)

    #merge two networks together
    singleOut = concatenate([softmax,confidence])


    model = Model(inputs= data, outputs=[singleOut])
    model.compile(optimizer='rmsprop',loss=[weightedLoss],metrics=['accuracy'])
    model.summary()
    return model
