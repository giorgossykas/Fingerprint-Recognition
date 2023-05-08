import tensorflow as tf
from skimage.filters import threshold_niblack

# Import all Classes and methods from the scripts
from image_enhancement import *
from orientation_and_cropping import *

'''
search.py will take an image as input and search for a potential match.
'''


def search_user(img):
    # First get the image of the user(here it's from a local directory)
    img = cv.imread(
        r"C:\Users\giorg\Desktop\Python scripts\Fingerprint recognition notebook\Fingerprint Data\myFingerprintData\new\R_m_2.jpg")
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
    # Feeding the CNN - first load it(here it is my local directory, change it)
    embedding_model = tf.keras.models.load_model(
        r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\untrained_model.h5')
    embedding_model.load_weights(r'C:\Users\giorg\Desktop\Python scripts\Fingerprint-Recognition\models\emb_model.h5')
    embedding = embedding_model.predict(img)
    # Now the euclidean distance between this embedding and all the others
    # already stored in the database will be calculated. If the distance is
    # below a certain threshold, selected from the ROC curve, the corresponding
    # fingerprint will be considered a match. Otherwise, not recognized.
