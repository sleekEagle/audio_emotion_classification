#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:55:30 2019

@author: sleek-eagle
"""
import os
import wave
import numpy as np
import NNs
import keras
import os
import numbers
import re
from keras.utils.vis_utils import plot_model


NUM_CLASSES = 6
extracted_data = "path/for/extracted/data"

def get_wav_paths_and_emo():
    files = []
    for dirName, subdirList, fileList in os.walk(extracted_data):
        spt = dirName.split('/')
        dName = spt[len(spt)-1]
        #print(dName)    
        if(dName == 'audio'):
            for fname in fileList:
                print(fname)
                if (fname[-3:]) != 'wav':
                    continue
                full_name = dirName + "/" + fname
                spt =  full_name.split('/')
                emotion = spt[-3]
                vid_num = spt[-1][0:-4].split('_')[1]
                files.append([full_name,emotion,vid_num])
    return files
                
            
def is_emo_name(s):
    if ((s == 'an') | (s == 'di') | (s == 'fe') | (s == 'ha') | (s == 'sa') | (s == 'su')):
        return True
    return False

def get_features_class(files):
    feature_class = []
    for file in files:
        emotion = file[1]
        path = file[0]
        
        y,sr = librosa.load(path)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr)
        features = np.vstack([mfcc, mfcc_delta,chromagram])
        avg_features = np.average(features,axis=1)
        feature_class.append(np.append(avg_features,get_emo_num(emotion)))
        print(file)
        
    np_array = np.asarray(feature_class)
    return np_array
  
def get_emo_num(emotion):
    if emotion == 'an':
        return 0
    if emotion == 'di':
        return 1
    if emotion == 'fe':
        return 2
    if emotion == 'ha':
        return 3
    if emotion == 'sa':
        return 4
    if emotion == 'su':
        return 5
    return -1
    
def to_onehot(num):
    out = np.empty([0,NUM_CLASSES])
    for x in np.nditer(num):
        onehot = np.zeros(NUM_CLASSES)
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out


def normalize(v):
    min_v=v
    min_v = v.min(axis=0).reshape((1,v.shape[1]))
    min_v = np.repeat(min_v,v.shape[0],axis=0)
    range_v = np.ptp(v,axis = 0).reshape((1,v.shape[1]))
    range_v = np.repeat(range_v,v.shape[0],axis=0)
    norm = (v - min_v)/(range_v + 0.00001)
    return norm

#conffile is the configuration file for open smile which contain the features set to be calculated
conffile = 'configuration/file/for/openSMILE'
#outfile is a temp file to store the features temporarily
outfile = 'temp/out/file/for/features/any/location/this/is/temp'
       
def get_smile_features(files):
    feature_class = []
    num_features = []
    num_cleaned = []
    count = 0
    for file in files:
        count+=1
        #if count > 5 :
            #break
        print(count)
        command = 'SMILExtract -C '
        command += (conffile + '-O  ' + outfile)
        command += " -I "
        emotion = file[1]
        noise = "1"
        if(len(file) >=3):
            noise = file[2]
        path = file[0]
        command += path
        if os.path.exists(outfile):
            os.remove(outfile)
        os.system(command)
        a=open(outfile,'rb')
        lines = a.readlines()
        if lines:
            last_line = str(lines[-1])
            last_line = last_line[0:-3].split(',')[1:-1]
            num_features.append(len(last_line))
            cleaned = [float(i) for i in last_line]
            num_cleaned.append(len(cleaned))
            array = np.array(cleaned)
            app1 = np.append(array,get_emo_num(emotion))
            app2 = np.append(app1,float(noise))
            feature_class.append(app2)
        print(file)      
    np_array = np.asarray(feature_class)
    return np_array


#add noise to audio with sox

def add_noise():
    files = get_wav_paths_and_emo()
    path = "path/for/root/of/project/"
    out_path = path + "/noise-added"
    noise_path = path + "/noise"

    count = 0
    for dirName, subdirList, fileList in os.walk(noise_path):
        #print(fileList)
        for noise in fileList:
            #print("++++++++++++++++++")
            for dirName, subdirList, fileList_1 in os.walk(path + "/extracted"):
                spt = dirName.split('/')
                dName = spt[len(spt)-1]
                if(not is_emo_name(dName)):
                    continue
                for dirName, subdirList, sourceFiles in os.walk(path + "/extracted/"+dName + "/audio"):
                    #print(dirName)
                    spt_1 = dirName.split("/")
                    dName_1 = spt_1[len(spt_1)-1]
                    for source in sourceFiles:
                        noise_level = noise[6:-4]
                        out_name = noise_level + "_" + source
                        command = "sox -m " + noise_path + "/" + noise + " " + path + "/extracted/" + dName + "/audio/" + source + " " + out_path + "/" + out_name 
                        os.system(command)
                        count+=1
                        print(count)
                    
                    
def get_wav_paths_emo_noise():
    files = []
    path = "root/of/project/noise-added/"
    for dirName, subdirList, fileList in os.walk(path):
        for file in fileList:
            file_list = file[0:-4].split("_")
            noise = file_list[0]
            emotion = file_list[1]
            full_name = path+file
            files.append([full_name,emotion,noise])
    return files
            

    
#opensmile features 
def calc_opensmile_features():
    files = get_wav_paths_and_emo()
    video_num = np.array(files)[:,2].astype(int)
    
    features = get_smile_features(files)
    #np.save(file = '/home/sleek-eagle/research/Emotion_classification/code/RML/RML_audio_features_opensmile' , arr = features)
    
    
    #features = np.load('/home/sleek-eagle/research/Emotion_classification/code/RML/RML_audio_features_opensmile.npy')
    labels = features[:,-1]
    feature_values = features[:,0:-1]
    norm_feature_values = normalize(feature_values)
    #data = np.concatenate((labels.reshape((labels.shape[0],1)),video_num.reshape((video_num.shape[0],1)),norm_feature_values),axis=1)
    
    #np.save("/home/sleek-eagle/research/Emotion_classification/RML/combined_features/audio_level.npy",data)
    
    add_noise()
    files_noise = get_wav_paths_emo_noise()
    features_noise = get_smile_features(files_noise)
    #np.save(file = '/home/sleek-eagle/research/Emotion_classification/code/RML/RML_audio_features_noise_opensmile' , arr = features_noise)
    return features_noise

def seperate_noise_levels(features):
    l = []
    noise_levels = np.unique(features[:,features.shape[1]-1])
    for level in noise_levels:
        l.append(features[features[:,features.shape[1]-1] == level])
    return l
    
def randomize_list_array(noise_features):
    for array in noise_features:
        np.random.shuffle(array)
    return noise_features


#seperate the features (data) into train, test and validatiaon sets according to the ratio provided
def separate_list_traintestvali(noise_features,ratios):
    data = []
    for array in noise_features:
        train = []
        test = []
        vali = []
        
        features = array[:,0:-2]
        labels = array[:,-2]
        noise = array[:,-1]
        l=features.shape[0]
        
        features_train = features[0:int(l*ratios[0]),:]
        labels_train = labels[0:int(l*ratios[0])]
        noise_train = noise[0:int(l*ratios[0])]
        train.append([features_train,labels_train,noise_train])
        
        features_test = features[int(l*ratios[0]) : int(l*(ratios[0]+ratios[1])),:]
        labels_test = labels[int(l*ratios[0]) : int(l*(ratios[0] + ratios[1]))]
        noise_test = noise[int(l*ratios[0]) : int(l*(ratios[0] + ratios[1]))]
        test.append([features_test,labels_test,noise_test])

        
        features_vali = features[int(l*(ratios[0]+ratios[1])) : int(l*(ratios[0]+ratios[1]+ratios[2])),:]
        labels_vali = labels[int(l*(ratios[0]+ratios[1])) : int(l*(ratios[0] + ratios[1]+ratios[2]))]
        noise_vali = noise[int(l*(ratios[0]+ratios[1])) : int(l*(ratios[0] + ratios[1]+ratios[2]))]
        vali.append([features_vali,labels_vali,noise_vali])
        
        data.append([train,test,vali])
    return data

#returns train test and vali sets with specified noise levels
#specify noise levels as lists
#e.g [0.2,0.3,0.4] will give the data with coise levels 0.2,0.3 and 0.4
def get_data_noise(data,train_noise,test_noise,vali_noise):
    train = []
    test = []
    vali = []
    
    for d in data:
        if (d[0][0][2][0] in train_noise):
            train.append(d[0])
        if (d[1][0][2][0] in test_noise):
            test.append(d[0])
        if (d[2][0][2][0] in vali_noise):
            vali.append(d[0])
    return [train,test,vali]
        

def train_NN(train,test,vali):
    batch_size = 20
    epochs = 20
    train_labels = to_onehot(train[:,-2])
    test_labels = to_onehot(test[:,-2])
    vali_labels = to_onehot(vali[:,-2])
    
    model = NNs.get_audio_dnn_1(((train.shape[1]-2),))
    model_train = model.fit(train[:,0:-2],train_labels,batch_size=batch_size,epochs=epochs, validation_data = (vali[:,0:-2],vali_labels))

def trian_conf_NN(train,test,vali):
    batch_size = 20
    epochs = 20
    train_labels = to_onehot(train[:,-2])
    train_labels = np.append(train_labels,np.zeros((train_labels.shape[0],1)),axis=1)
    test_labels = to_onehot(test[:,-2])
    test_labels = np.append(test_labels,np.zeros((test_labels.shape[0],1)),axis=1)
    vali_labels = to_onehot(vali[:,-2])
    vali_labels = np.append(vali_labels,np.zeros((vali_labels.shape[0],1)),axis=1)
    
    model = NNs.get_audio_dnn_with_confidence(((train.shape[1]-2),))
    model.fit(train[:,0:-2],train_labels,batch_size=batch_size,epochs=epochs, validation_data = (vali[:,0:-2],vali_labels))
    
    return model
    
def get_accuracy(model,vali):
     vali_labels = to_onehot(vali[:,-2])
     pred = model.predict(vali[:,0:-2])
     pred_classes = pred[:,0:-1]
     max_values = np.amax(pred_classes,axis=1).reshape((-1,1))
     np.repeat(max_values,vali_labels.shape[1],axis=1)
     sub = pred_classes - max_values
     pred_classes = (sub >= 0).astype(int)
     diff = vali_labels - pred_classes
     non_zero = np.count_nonzero(diff,axis=1)
     correct_num =  (non_zero == 0).sum()
     total = non_zero.shape[0]
     accuracy = correct_num/total
     
     
     #accuracy for each class
     non_zero_clswise = np.count_nonzero(diff,axis=0)
     num_each_class = np.count_nonzero(vali_labels,axis=0)
     corrects = (~diff.any(axis=1)).astype(int)
     corrects = corrects.reshape((corrects.shape[0],1))
     corrects = np.repeat(corrects ,vali_labels.shape[1],axis=1 )
     check =  np.multiply(vali_labels,corrects)
     correct_num = np.sum(check,axis=0)
     correct_per = correct_num/num_each_class
     
     

     
     return accuracy,correct_per
  
 
def get_pred(model,vali):
     vali_labels = to_onehot(vali[:,-2])
     pred = model.predict(vali[:,0:-2])
     pred_classes = pred[:,0:-1]
     max_values = np.amax(pred_classes,axis=1).reshape((-1,1))
     np.repeat(max_values,vali_labels.shape[1],axis=1)
     sub = pred_classes - max_values
     pred_classes = (sub >= 0).astype(int)
     pred_conf = pred[:,-1]
     return [pred_classes,pred_conf]
              
def mix_noise_data(data):
    features = []
    labels = []
    noise = []
    for d in data:
        features.append(d[0][0])
        labels.append(d[0][1])
        noise.append(d[0][2])
    feature_arr = np.array(features)
    feature_arr = feature_arr.reshape(-1,feature_arr.shape[2])
    label_arr = np.array(labels).reshape(-1,1)
    noise_arr = np.array(noise).reshape(-1,1)
    array = np.append(feature_arr,label_arr,axis=1)
    array = np.append(array,noise_arr,axis = 1)
    #randomize
    np.random.shuffle(array)
    return array


    
###################################################
#####train
#####################
    
features_noise = calc_opensmile_features()   
#seperate feautues into various noise levels 
noise_features = seperate_noise_levels(features_noise)
randomize_list_array(noise_features)
noise_data = separate_list_traintestvali(noise_features,[0.6,0.2,0.2])

#
[train,test,vali] = get_data_noise(noise_data,[1,2,3,0.1,0.01,0.2,0.3,0.03,0.4,0.5,0.05,0.6,0.7,0.07,0.8,0.9],[0.8],[0.01,0.03])
[train,test,vali] = [mix_noise_data(train),mix_noise_data(test),mix_noise_data(vali)]
model = trian_conf_NN(train,test,vali)

accuracy = get_accuracy(model,vali)
[pred_classes,pred_conf] = get_pred(model,vali)
conf_mean = np.mean(pred_conf)

print(accuracy)
print(conf_mean)

plot_model(model, to_file = 'conf_model.png', show_shapes=True, show_layer_names=True)
##################################################################


    
    
    


