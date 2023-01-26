import time
import tensorflow as tf
import cv2 as cv
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import scipy
from scipy import signal
from scipy import ndimage
import math
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from sklearn.metrics import roc_curve, roc_auc_score
import tkinter as tk
from tkinter import ttk
plt.style.use('seaborn')

# Import all Classes and methods from the scripts
from evaluation_testing import *
from image_enhancement import *
from minutiae_extraction import *
from orientation_and_cropping import *
from register import *

'''
    When running main.py there are two options: either register a new user,
    or check the user fingerprint to see if it matches any from the database(user authentication).
'''

if __name__ == "__main__":

    result = Evaluator()

    answer = 0
    while (answer != 1)  and (answer != 2) and (answer != 3) and (answer != 4):
        answer = int(input("Type 1 for ROC, 2 for image processing display, 3 for database search, 4 for new user registration > "))
    if answer == 1:
        images = np.zeros((15, 350, 350))
        labels = []

        # Create object to preprocess the images
        aligner = OrientationCrop()
        enhancer = ProcessEnhance()

        # Find the directory in which the images are stored, my own or the poly dataset
        #dir = "C:/Users/giorg/OneDrive/Υπολογιστής/Python scripts/Fingerprint recognition notebook/Fingerprint Data/myFingerprintData/new"
        dir = os.getcwd()
        dir = os.path.join(dir, "Fingerprint Data\\myFingerprintData\\newdata")
        i = 0
        for image in os.listdir(dir): # I have 15 images, 5 fingers with 3 different fingerprint images for each.
            path = '.\\Fingerprint Data\\myFingerprintData\\new' + '\\' + image
            img = cv.imread(path)
            print(type(img))
            img = aligner.process(img)
            img = enhancer.process(img)
            images[i, :, :] = img
            labels.append(image.split('_')[0])
            i+=1

        # Now my images are stored in X, all with dims 350x350 and their labels are stored in Y
        # Their names were in the form of Li_2.jpg which stands for Left hand, index finger, second image
        # Next step is to transform every image into an embedding using the CNN  to display the ROC curve
        # First I load the embedding model
        embedding_model = tf.keras.load_model('C:/Users/giorg/OneDrive/Υπολογιστής/Python scripts/Fingerprint recognition notebook/untrained_model.h5')
        embedding_model = tf.keras.load_weights('C:/Users/giorg/OneDrive/Υπολογιστής/Python scripts/Fingerprint recognition notebook/CustomCNN_final/emb_model.h5')

        # CNN requires 356x328 images, and I have 350x350, so I must resize them first
        # and binarize them again because after the resizing they will become grayscale.
        X = np.empty((images.shape[0], 356, 328))
        for i in range(images.shape[0]):
            temp = cv.resize(images[i, :, :], (328, 356), interpolation=cv.INTER_CUBIC)
            nib_thr = threshold_niblack(temp, window_size=21, k=0.1)
            temp = temp > nib_thr
            temp = temp * 255
            X[i, :, :] = temp
        X = np.expand_dims(X, axis = -1)
        Y = labels

        # Now I will run the method "evaluate" to draw the ROC curve
        # If I omit the embedding model argument it will be calculated using the minutiae points
        result.evaluate(X, Y, targetFPR = 1e-03, embedding_model = embedding_model)

    elif answer == 2:
        print(2)
    elif answer == 3:
        print(3)
    elif answer == 4:
        print(4)