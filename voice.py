#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:44:21 2018

@author: sleek_eagle
"""


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import wave
import struct
import numpy as np


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import random


FFT_WINDOW_SIZE = 25 #in ms
FFT_OVERLAP = 0.6 #overlap raio of the window
WINDOW_LEN = 4 #in seconds. this is the window length for classification

NUM_FREQ = 601
NUM_TIMES = 398
NUM_CLASSES = 8


def training():
    rootDir = '/home/sleek_eagle/research/Emotion_classification/voice_sing'
    files = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            files.append(dirName + "/" + fname)
            
    spectrograms = []
    for i in files:
        print(i)
        spectrograms.append(get_spectrogram(i)[2])
    spectrograms = np.array(spectrograms).reshape(-1,NUM_FREQ,NUM_TIMES,1)
    
    classes = []
    for i in files:
        s_cls = get_class_from_filename(i)
        cls = to_onehot(int(s_cls))
        classes.append(cls)
        
    data = []
    for i in range(len(classes)):
        data.append([spectrograms[i],classes[i]])
    #randomize instances   
    random.shuffle(data)
    #devide into train test and validation sets
    TRAIN_RATIO = 0.6
    TEST_RATIO = 0.2
    
    train_data = data[0:int(len(data)*TRAIN_RATIO)]
    test_data = data[int(len(data)*TRAIN_RATIO) : int(len(data)*(TRAIN_RATIO + TEST_RATIO))]
    validation_data = data[int(len(data)*(TRAIN_RATIO + TEST_RATIO)) : len(data)]
    
    
    train_values = np.array([i[0] for i in train_data])
    train_labels = np.array([i[1] for i in train_data])
    
    test_values = np.array([i[0] for i in test_data]) 
    test_labels = np.array([i[1] for i in test_data]) 
    
    validation_values = np.array([i[0] for i in validation_data]) 
    validation_labels = np.array([i[1] for i in validation_data]) 
    
    #define the CNN
    batch_size = 64
    epochs = 20
    
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(NUM_FREQ,NUM_TIMES,1),padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    #fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    fashion_model.summary()
    
    fashion_train = fashion_model.fit(train_values, train_labels, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(validation_values, validation_labels))
    
    test_eval = fashion_model.evaluate(pixcels_test, labels_test, verbose=0)
    
        
    
    
    path = '/home/sleek_eagle/research/Emotion_classification/audio.wav'
    path = '/home/sleek_eagle/research/Emotion_classification/voice_sing/Actor_08/03-02-05-02-01-01-08.wav'
    times,frequencies,norm_spectrogram = get_spectrogram(path)
    
    plt.pcolormesh(times, frequencies, norm_spectrogram)
    
def spectro_time_to_seconds(time):
    start = (1-FFT_OVERLAP)*FFT_WINDOW_SIZE/1000*time
    middle = start + FFT_WINDOW_SIZE/(1000*2)
    return middle

#spectrogram should be a ndarray
def get_effective_time(spectrogram):
    avgs = spectrogram.mean(axis=0)
    start = 0
    end = 0
    for i in range(0,len(avgs)):
        if(avgs[i] > 0.02):
            start = i
            break
    for i in range(0,len(avgs)+1):   
        if(avgs[len(avgs) - 1 - i] > 0.02):
            end = (len(avgs) -1 -i)
            break
    return start,end
    

def to_onehot(num):
  num-=1
  onehot = np.zeros(NUM_CLASSES)
  onehot[num] = 1
  return onehot


def get_class_from_filename(path):
    ar = path.split("/")
    s = ar[len(ar)-1]
    name = s.split("-")
    cls = name[2]
    return cls

def read_whole(path):
    wav_r = wave.open(path, 'r')
    sample_rate = wav_r.getframerate()
    channels = wav_r.getnchannels()
    sw = wav_r.getsampwidth()
    print(sample_rate)
    print(channels)
    print(sw)
    s = '<h'
    for i in range(0,channels-1):
        s+='h'
        
    data = []
    while wav_r.tell() < wav_r.getnframes():
        decoded = struct.unpack(s, wav_r.readframes(1))[0]
        data.append(decoded)
    return data,sample_rate

def get_spectrogram(path):
    samples,sample_rate = read_whole(path)
    #zero pad data when less than 4s long
    required_datapoints = WINDOW_LEN * sample_rate
    pad_width = required_datapoints - len(samples)
    if (pad_width < 0):
        pad_width = 0
    
    samples = np.pad(samples,pad_width=(0,pad_width),mode='constant',constant_values=0)
    
    
    
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, 
                                                         window  = "hamming" , 
                                                         nperseg = int(FFT_WINDOW_SIZE/1000 * sample_rate) , 
                                                         noverlap = FFT_OVERLAP*int(FFT_WINDOW_SIZE/1000 * sample_rate))
    
    #normalize the spectrogram
    row_sums = spectrogram.sum(axis = 1)
    row_mins = spectrogram.min(axis = 1)
    row_max = spectrogram.max(axis = 1)
    norm_spectrogram = (spectrogram - row_mins[:,np.newaxis])/(row_max[:,np.newaxis] - row_mins[:,np.newaxis])
    
    #select the most active 4s window
    amp_sums = norm_spectrogram.sum(axis=0)
    n_windows = int((WINDOW_LEN*1000/FFT_WINDOW_SIZE - 1)/(1 - FFT_OVERLAP) + 1)
    
    activity = []
    for i in range(0,len(amp_sums)-n_windows+1):
        activity.append(sum(amp_sums[i:i+n_windows]))
    start = np.argmax(activity)
    end = start + n_windows
    norm_spectrogram = norm_spectrogram[:,start:end]
    times = times[start:end]
    times = times - times[0]
    return (times,frequencies,norm_spectrogram)
        
    

