# classify spoken audio into emotion states with deep neural networks

This is source code for classifying amotions from audio data. 
This uses deep neural networks to do this. Using python Keras is used to build models and for prediction.

Brief description of files

## NNs.py
Thhis defines the structure of deep neural networks in Keras

## video.py
This extracts audio from video files containing various emotions.

## audio.py
- This takes the audio files generated bu video.py and 
- Calculates audio features with OpenSMILE. 
- Trains DNN model from models defined by NNs.py
- Gives classification resulrs

This code builds a NN with confidence prediction branch. For more details read my article on the subject on 
https://sleekeagle.github.io/
This method is based on the paper 
### Learning Confidence for Out-of-Distribution Detection in Neural Networks
by Terrance DeVries and Graham W. Taylor

Take a look at network.pdf file to see the network structure. For those who are familier with Keras, 
this is the summary of the model

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
data (InputLayer)               (None, 384)          0                                            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 384)          147840      data[0][0]                       
__________________________________________________________________________________________________
dense_1/bn (BatchNormalization) (None, 384)          1536        dense_1[0][0]                    
__________________________________________________________________________________________________
bn_1/relu (Activation)          (None, 384)          0           dense_1/bn[0][0]                 
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 20)           7700        bn_1/relu[0][0]                  
__________________________________________________________________________________________________
conf_dense_1 (Dense)            (None, 384)          147840      data[0][0]                       
__________________________________________________________________________________________________
dense_2/bn (BatchNormalization) (None, 20)           80          dense_2[0][0]                    
__________________________________________________________________________________________________
conf_dense_1/bn (BatchNormaliza (None, 384)          1536        conf_dense_1[0][0]               
__________________________________________________________________________________________________
bn_2/relu (Activation)          (None, 20)           0           dense_2/bn[0][0]                 
__________________________________________________________________________________________________
conf_bn_1/relu (Activation)     (None, 384)          0           conf_dense_1/bn[0][0]            
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 6)            126         bn_2/relu[0][0]                  
__________________________________________________________________________________________________
conf_dense_2 (Dense)            (None, 20)           7700        conf_bn_1/relu[0][0]             
__________________________________________________________________________________________________
dense_3/bn (BatchNormalization) (None, 6)            24          dense_3[0][0]                    
__________________________________________________________________________________________________
conf_dense_2/bn (BatchNormaliza (None, 20)           80          conf_dense_2[0][0]               
__________________________________________________________________________________________________
bn_3/relu (Activation)          (None, 6)            0           dense_3/bn[0][0]                 
__________________________________________________________________________________________________
conf_bn_2/relu (Activation)     (None, 20)           0           conf_dense_2/bn[0][0]            
__________________________________________________________________________________________________
relu_3/softmax (Activation)     (None, 6)            0           bn_3/relu[0][0]                  
__________________________________________________________________________________________________
confidence (Dense)              (None, 1)            21          conf_bn_2/relu[0][0]             
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 7)            0           relu_3/softmax[0][0]             
                                                                 confidence[0][0]                 
==================================================================================================
Total params: 314,483
Trainable params: 312,855
Non-trainable params: 1,628

