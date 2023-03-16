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
        dir = r"C:\Users\giorg\Desktop\Python scripts\Fingerprint recognition notebook\Fingerprint Data\myFingerprintData\new"
        #dir = os.getcwd()
        #dir = os.path.join(dir, "Fingerprint Data\\myFingerprintData\\newdata")
        i = 0
        for image in os.listdir(dir): # I have 15 images, 5 fingers with 3 different fingerprint images for each.
            path = dir + '\\' + image
            img = cv.imread(path)
            img = aligner.process(img)
            img = enhancer.process(img)
            images[i, :, :] = img
            labels.append(image.split('_')[0])
            i+=1

        choice = 0
        # Now choose the method to follow
        while (choice != 1) and (choice != 2):
            choice = int(input("Type 1 for CNN and 2 for Minutiae > "))
        if choice==1:
            # Now my images are stored in X, all with dims 350x350 and their labels are stored in Y
            # Their names were in the form of Li_2.jpg which stands for Left hand, index finger, second image
            # Next step is to transform every image into an embedding using the CNN  to display the ROC curve
            # First I load the embedding model
            embedding_model = tf.keras.models.load_model(r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\untrained_model.h5')
            embedding_model.load_weights(r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\emb_model.h5')

            # CNN requires 356x328 images, and I have 350x350, so I must resize them first
            # and binarize them again because after the resizing they will become grayscale.
            X = np.empty((images.shape[0], 356, 328))
            for i in range(images.shape[0]):
                temp = cv.resize(images[i, :, :], (328, 356), interpolation=cv.INTER_CUBIC)
                nib_thr = threshold_niblack(temp, window_size=21, k=0.1)
                temp = temp > nib_thr
                temp = temp * 255
                X[i, :, :] = temp
            cv.imwrite("fingerprint.jpg", X[0])
            X = np.expand_dims(X, axis = -1)
            Y = labels

            # Now I will run the method "evaluate" to draw the ROC curve
            # If I omit the embedding model argument it will be calculated using the minutiae points
            result.evaluate(X, Y, targetFPR = 1e-03, embedding_model = embedding_model)

        elif choice==2:
            X = images
            Y = labels
            result.evaluate(X, Y, targetFPR=1e-03)

    elif answer == 2:
        # Choose one picture to display the process before being fed into the network
        img = cv.imread(r"C:\Users\giorg\Desktop\Python scripts\Fingerprint recognition notebook\Fingerprint Data\myFingerprintData\new\L_p_2.jpg")

        choice = 0
        while (choice!=1) and (choice!=2):
            choice = int(input("Type 1 for image process display and 2 for Minutiae of fingerprint > "))

        # First align and crop the image
        aligner = OrientationCrop()
        enhancer = ProcessEnhance()

        if choice == 1:
            aligner.process_and_display(img)
            img = aligner.process(img)
            enhancer.process_and_display(img)

        elif choice == 2:
            img = aligner.process(img)
            img = enhancer.process(img)
            extract_minutiae_features(img, showResult=True)

    elif answer == 3:
        # First get the image of the user, here lets say it's the image L_i_1.jpg again
        img = cv.imread(r"C:\Users\giorg\Desktop\Python scripts\Fingerprint recognition notebook\Fingerprint Data\myFingerprintData\new\R_m_2.jpg")
        aligner = OrientationCrop()
        enhancer = ProcessEnhance()
        img = aligner.process(img)
        img = enhancer.process(img)
        # This the processed fingerprint image which will
        # now be resized to be fed into the network
        temp = cv.resize(img, (328, 356), interpolation=cv.INTER_CUBIC)
        nib_thr = threshold_niblack(temp, window_size=21, k=0.1)
        temp = temp > nib_thr
        temp = temp * 255
        img = temp
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        #print(img.shape)
        # Feeding the CNN - first load it
        embedding_model = tf.keras.models.load_model(r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\untrained_model.h5')
        embedding_model.load_weights(r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\emb_model.h5')
        embedding = embedding_model.predict(img)
        # Now the euclidean distance between this embedding and all the others
        # already stored in the database will be calculated. If the distance is
        # below a certain threshold, selected from the ROC curve, the corresponding
        # fingerprint will be considered a match.
    elif answer == 4:
        # Same as before (case 3) the embedding will be calculated and stored
        # in the database along with the name of the individual which is asked.
        # First get the image of the user, here lets say it's the image L_i_1.jpg again
        img = cv.imread(
            r"C:\Users\giorg\Desktop\Python scripts\Fingerprint recognition notebook\Fingerprint Data\myFingerprintData\new\L_i_1.jpg")
        aligner = OrientationCrop()
        enhancer = ProcessEnhance()
        img = aligner.process(img)
        img = enhancer.process(img)
        # This the processed fingerprint image which will
        # now be resized to be fed into the network
        temp = cv.resize(img, (328, 356), interpolation=cv.INTER_CUBIC)
        nib_thr = threshold_niblack(temp, window_size=21, k=0.1)
        temp = temp > nib_thr
        temp = temp * 255
        img = temp
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis = 0)

        # Feeding the CNN - first load it
        embedding_model = tf.keras.models.load_model(
            r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\untrained_model.h5')
        embedding_model.load_weights(
            r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\emb_model.h5')
        embedding = embedding_model.predict(img)
        print(embedding)
        fullname = input("Enter your Fullname: ")
        # Now save both the name and the embedding