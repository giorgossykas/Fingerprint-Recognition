# Fingerprint-Recognition
The repository contains files for fingerprint matching using both a Convolutional Neural Network and Minutiae extraction. 
The Jupyter Notebooks contain the code used to create and train the CNN and all the scripts for the image processing and minutiae extraction.  
The .py files contain the code for the application of the fingerprint matching using the images from each user.  
Below are all the .py and .yml files extensively described.  
- train_CNN.py : Contains all the code to train the CNN. Can be run directly from the terminal on a GPU on a server.
- register.py : Function to save a new users' name and fingerprint embedding into the database. It is to be completed when an actual application will be deployed.
- orientation_and_cropping.py : Contains all the classes and functions for the first step of the image processing.
- image_enhancement.py : Contains all classes and functions to process the output image of "orientation_and_cropping.py" to make it look like a contact-based fingerprint.
- minutiae_extraction.py : Classes and functions to extract the minutiae features of a fingerprint(output of "image_enhancement.py") and compare them to another. Tis script is   not used in the "main.py". This is because the results of the CNN were much better than tose of the minutiae. However using the class Evaluator from      "evaluation_and_testing.py" you can calculate the number of matched minutiae points between two images or even draw the ROC curve. Changing the parameter showResult to True in   the extract_minutiae_features method will display the minutiae.
- evaluation_testing.py : Contains all the necessary methods to display and evaluate the results.
- main.py : This is the script that through the users' input can run everything, ROC, process display, registering and matching.
- Fingerprint_env.yml : All the necessary packages and dependancies are in here. The environment needs to be installed for the easier use of the scripts, especially for the training of the CNN.  
## Using the CNN method :  
If you do not want to train the network from scratch there is an already trained model which you can access and download [here](insert link). You can also download the untrained model to compare the difference in results.  
All the code was written in Python 3.9.  
  
Credits: 
- [Minutiae extraction code](https://github.com/Utkarsh-Deshmukh/Fingerprint-Feature-Extraction) from Utkarsh-Deshmukh.
- [Image enhancement code](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python/blob/develop/src/FingerprintImageEnhancer.py) again from Utkarsh-Deshmukh.
