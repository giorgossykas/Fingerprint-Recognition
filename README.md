# Fingerprint-Recognition
The repository contains files for fingerprint matching using both a Convolutional Neural Network and Minutiae extraction.
The Jupyter Notebooks contain the code used to create and train the CNN and all the scripts for the image processing and minutiae extraction.  
The .py files contain the code for the application of the fingerprint matching using the images from each user.  
Below are all the .py and .yml files extensively described.  
- train_CNN.py : Contains all the code to train the CNN. Can be run directly from the terminal on a GPU on a server.
- register.py : Function to save a new users' name and fingerprint embedding into the database. It is to be completed when an actual application will be deployed.
- orientation_and_cropping.py : Contains all the classes and functions for the first step of the image processing.
- image_enhancement.py : Contains all classes and functions to process the output image of "orientation_and_cropping.py" to make it look like a contact-based fingerprint.
- minutiae_extraction.py : Classes and functions to extract the minutiae features of a fingerprint(output of "image_enhancement.py") and compare them to another.
- evaluation_testing.py : Contains all the necessary methods to display and evaluate the results.
- main.py : This is the script that through the users' input can run everything, ROC, process display, registering and matching. The ROC curve will help to choose the  
  appropriate threshold for the matching comparisons.

- Fingerprint_env.yml contains all the necessary packages and dependancies to run the main.py.
- fingerprint_train.yml contains all the necessary packages and dependancies to train the network.  
- Images folder contains two test images to use and the outcome of a fingerprint.
- models folder is empty (contains a .gitignore file). The models that will be downloaded will be put here.

## Image processing.  
The workflow of the image processing is displayed below.  
![Screenshot_20230209_030411](https://user-images.githubusercontent.com/23582994/217821134-348a0f59-44bb-4468-95fb-b3c91b02b6c0.png)  
Then taking the cropped image:  
![merge_from_ofoct(1)](https://user-images.githubusercontent.com/23582994/217827675-03e99e54-76fd-4282-b80f-79e78e3b21d6.jpg)  

## Using the CNN method :  
If you do not want to train the network from scratch there is an already trained model which you can access and download [here](https://drive.google.com/drive/folders/1_SuyUE58SOivMXq-ux2pQu10yZ2ysoEZ?usp=sharing). You need to download both models in order for it to work. You can also use the untrained model alone to compare the difference in results.  
All the code was written in Python 3.9.  
  
Credits: 
- [Minutiae extraction code](https://github.com/Utkarsh-Deshmukh/Fingerprint-Feature-Extraction) from Utkarsh-Deshmukh.
- [Image enhancement code](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python/blob/develop/src/FingerprintImageEnhancer.py) again from Utkarsh-Deshmukh.
