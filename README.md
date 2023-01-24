# Fingerprint-Recognition
The repository contains files for fingerprint matching using both a Convolutional Neural Network and Minutiae extraction. 
The Jupyter Notebooks contain the code used to create and train the CNN and all the scripts for the image processing and minutiae extraction.  
The .py files contain the code for the application of the fingerprint matching using the images from each user.  
Below are all the .py and .yml files extensively described.  
- train_CNN.py : 
- register.py : 
- orientation_and_cropping.py : 
- image_enhancement.py : 
- minutiae_extraction.py : 
- evaluation_testing.py : 
- main.py : 
- Fingerprint_env.yml : All the necessary packages and dependancies are in here. The environment needs to be installed for the easier use of the scripts, especially for the training of the CNN.  
## Using the CNN method :  
If you do not want to train the network from scratch there is an already trained model which you can access and download [here](insert link). You can also downoad the untrained model to compare the difference in results.  
All the code was written in Python 3.9.  
  
Credits: 
- [Minutiae extraction code](https://github.com/Utkarsh-Deshmukh/Fingerprint-Feature-Extraction) from Utkarsh-Deshmukh.
- [Image enhancement code](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python/blob/develop/src/FingerprintImageEnhancer.py) again from Utkarsh-Deshmukh.
