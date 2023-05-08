import tensorflow as tf
from skimage.filters import threshold_niblack

# Import all Classes and methods from the scripts
from image_enhancement import *
from orientation_and_cropping import *

'''
register.py will perform the user registration.
First it will take a name and an image (of a finger)
as input and it will store them in the database
'''

def register_user(img, name):
    # Here I use an image from my local directory and I input the name.
    # In the application both will be the inputs to function from the device.
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
    img = np.expand_dims(img, axis=0)

    # Feeding the CNN - first load it (change the directory)
    embedding_model = tf.keras.models.load_model(
        r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\untrained_model.h5')
    embedding_model.load_weights(
        r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\emb_model.h5')
    embedding = embedding_model.predict(img)
    fullname = input("Enter your Fullname: ")
    # Now save both the name and the embedding