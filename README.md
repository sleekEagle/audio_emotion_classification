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
