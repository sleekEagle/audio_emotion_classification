#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:01:47 2018

@author: sleek_eagle
"""

import subprocess
import cv2
import math
import os

import voice

raw_data = "store/row/data/path"
extracted_data = "path/for/extracted/data"


def get_files(path):
    files = []
    for dirName, subdirList, fileList in os.walk(path):
        for fname in fileList:
            if (fname[-3:]) != 'avi':
                continue
            full_name = dirName + "/" + fname
            emotion = fname[0:2]
            files.append([full_name,emotion])
    return files

def extract_data(files):
    n = 0
    for file in files:
        new_file_name = file[1] + "_" + str(n)
        
        audio_dir = extracted + "/" + file[1] + "/audio/"
        audio_file = audio_dir +  new_file_name + ".wav" 
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)  
        
        
        spectro_dir = extracted + "/" + file[1] + "/spectrogram/"
        spectro_path =  spectro_dir +  new_file_name
        if not os.path.exists(spectro_dir):
            os.makedirs(spectro_dir) 
        
        image_dir = extracted + "/" + file[1] + "/image/"
        image_path = image_dir +  new_file_name
        if not os.path.exists(image_dir):
            os.makedirs(image_dir) 
        
        
        image_array_dir = extracted + "/" + file[1] + "/image_array/"
        image_array_path =  image_array_dir +  new_file_name
        if not os.path.exists(image_array_dir):
            os.makedirs(image_array_dir)
    
        #save sudio file
        command = "ffmpeg -i " + file[0] + " -ab 160k -ac 2 -ar 44100 -vn "  + audio_file
        subprocess.call(command, shell=True)
        
        #create spectrgram 
        times,frequencies,norm_spectrogram = voice.get_spectrogram(audio_file)
        #plt.pcolormesh(times, frequencies, norm_spectrogram)
        start,end = voice.get_effective_time(norm_spectrogram)
        start_time = voice.spectro_time_to_seconds(start)
        end_time = voice.spectro_time_to_seconds(end)
        np.save(spectro_path,norm_spectrogram)
        
        #store images between the required times
        time_array = np.linspace(start_time,end_time,num=5)
        for time in time_array:
            image_file = image_path  +"_" + str(round(time,3)) + ".jpg"
            command = "ffmpeg -i " + file[0] + " -ss " + convert_s_to_HMS(time) + " -vframes 1 "  + image_file
            subprocess.call(command, shell=True)
             #scale images and store as np arrays
            img = cv2.imread(image_file)
            res = cv2.resize(img,(48,48), interpolation = cv2.INTER_CUBIC)
            np.save((image_array_path + "_" + str(round(time,3))),res)
    
        
        n+=1
    
def convert_s_to_HMS(seconds):
    s = "00:00:"
    s+=str(round(seconds,3))
    return s