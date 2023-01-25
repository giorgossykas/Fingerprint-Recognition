import time
#import tensorflow as tf
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

if name == "__main__":
    # Run app