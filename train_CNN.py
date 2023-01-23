import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import random
import os
import glob
import random
from datetime import datetime
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Input, concatenate
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, MaxPooling2D, Concatenate, Lambda, Flatten, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.regularizers import l2
import math
import json
from tensorflow.python.client import device_lib

####################################################################################
# Load the processed images and labels saved as an .npz file
loaded = np.load('fingers_poly.npz')
x_train, y_train, x_test, y_test = loaded['x_train'], loaded['y_train'], loaded['x_test'], loaded['y_test']

x_train = x_train / 255.
x_test = x_test / 255.

print(f'x_train:', x_train.shape, 'y_train:', y_train.shape)
print(f'x_test:', x_test.shape, 'y_test:', y_test.shape)

####################################################################################
batch_size = 32
num_hard = 16


def create_batch(batch_size=batch_size, split='train'):
    x_anchors = np.zeros((batch_size, x_train.shape[2], x_train.shape[3]))
    x_positives = np.zeros((batch_size, x_train.shape[2], x_train.shape[3]))
    x_negatives = np.zeros((batch_size, x_train.shape[2], x_train.shape[3]))

    if split == "train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test

    for i in range(0, batch_size):
        # Find an anchor, a positive (augmentation from the same finger) and a negative example.
        # anchor
        rand_finger = random.randint(0, data.shape[1] - 1)  # Choose a finger randomly
        x_anchor = data[0, rand_finger, :, :]

        # positive
        rand_aug = random.randint(1, 5)  # Choose a random augmentation of it
        while np.all(data[rand_aug, rand_finger, :, :] == np.zeros(shape=(x_train.shape[2], x_train.shape[3]))):
            rand_aug = random.randint(1, 5)
        x_positive = data[rand_aug, rand_finger, :, :]

        # negative
        rand_neg = random.randint(0, data.shape[1] - 1)  # Choose a random negative finger DIFFERENT than the anchor
        while rand_neg == rand_finger:
            rand_neg = random.randint(0, data.shape[1] - 1)
        x_negative = data[0, rand_neg, :, :]

        x_anchors[i, :, :] = x_anchor
        x_positives[i, :, :] = x_positive
        x_negatives[i, :, :] = x_negative

    return [x_anchors, x_positives, x_negatives]


##################################################################################

def create_hard_batch(batch_size, num_hard, split='train'):
    x_anchors = np.zeros((batch_size, x_train.shape[2], x_train.shape[3]))  ### All three are of size (32,100,160)
    x_positives = np.zeros((batch_size, x_train.shape[2], x_train.shape[3]))
    x_negatives = np.zeros((batch_size, x_train.shape[2], x_train.shape[3]))

    if split == "train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test

    # Generate num_hard number of hard examples
    hard_batches = []
    batch_losses = []

    rand_batches = []

    # Get some random batches
    for i in range(0, batch_size):
        hard_batches.append(create_batch(1, split))  # Returns only one triplet

        A_emb = embedding_model.predict(np.expand_dims(hard_batches[i][0], axis=-1))
        P_emb = embedding_model.predict(np.expand_dims(hard_batches[i][1], axis=-1))
        N_emb = embedding_model.predict(np.expand_dims(hard_batches[i][2], axis=-1))

        # Compute distance for each selected batch
        batch_losses.append(np.sum(np.square(A_emb - P_emb), axis=1) - np.sum(np.square(A_emb - N_emb), axis=1))

    # Sort batch_loss by distance, highest first, and keep num_hard of them

    # Semi-hard require dist(A,N) > dist(A,P) AND still positive loss so
    # I choose the hard batches based on batch loss ascending order. The line
    # below creates zip/pairs of batch_losses, hard_batches and returns ONLY
    # the hard batches (_,x) having sorted in ascending order the batch loss (x[0]).

    hard_batch_selections = [x for _, x in sorted(zip(batch_losses, hard_batches), key=lambda x: x[0])]
    hard_batches = hard_batch_selections[:num_hard]

    # Get batch_size - num_hard number of random examples
    num_rand = batch_size - num_hard
    for i in range(0, num_rand):
        rand_batch = create_batch(1, split)
        rand_batches.append(rand_batch)

    selections = hard_batches + rand_batches

    for i in range(0, len(selections)):
        x_anchors[i, :, :] = selections[i][0]
        x_positives[i, :, :] = selections[i][1]
        x_negatives[i, :, :] = selections[i][2]

    x_anchors = np.expand_dims(x_anchors, axis=-1)
    x_positives = np.expand_dims(x_positives, axis=-1)
    x_negatives = np.expand_dims(x_negatives, axis=-1)

    return [x_anchors, x_positives, x_negatives]


#################################################################################

def create_embedding_model(emb_size):
    embedding_model = tf.keras.models.Sequential()

    embedding_model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', kernel_initializer='he_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001),
                               input_shape=(x_train.shape[2], x_train.shape[3], 1)))

    embedding_model.add(BatchNormalization())

    embedding_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output: (178,164,32)

    embedding_model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', kernel_initializer='he_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    embedding_model.add(BatchNormalization())

    embedding_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output: (89,82,64)

    embedding_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', kernel_initializer='he_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    embedding_model.add(BatchNormalization())

    embedding_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output: (44,41,128)

    embedding_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               activation='relu', kernel_initializer='he_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    embedding_model.add(BatchNormalization())

    embedding_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # output: (22,20,256)

    embedding_model.add(Dropout(rate=0.3))

    embedding_model.add(Flatten())  # output: 22*20*256

    embedding_model.add(Dense(units=512, activation='relu', kernel_initializer='he_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    embedding_model.add(Dense(units=256, activation='relu', kernel_initializer='he_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    # output layer with embedding_size no of units
    embedding_model.add(Dense(units=emb_size, activation=None, kernel_initializer='he_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)))

    # Force the embedding to live on the d-dimentional hypershpere
    embedding_model.add(Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1)))

    embedding_model.summary()

    return embedding_model


#################################################################################

def create_SNN(embedding_model):
    input_anchor = Input(shape=(x_train.shape[2], x_train.shape[3],))
    input_positive = Input(shape=(x_train.shape[2], x_train.shape[3],))
    input_negative = Input(shape=(x_train.shape[2], x_train.shape[3],))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    siamese_net = tf.keras.models.Model(inputs=[input_anchor, input_positive, input_negative],
                                        outputs=output)

    siamese_net.summary()

    return siamese_net


#################################################################################

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    tr_loss = tf.maximum(positive_dist - negative_dist + alpha, 0.)
    return tr_loss


#################################################################################

def data_generator(batch_size=batch_size, num_hard=num_hard, split='train'):
    leles = 0
    while True:
        x = create_hard_batch(batch_size, num_hard)
        y = np.zeros((batch_size, 3 * emb_size))
        leles += 1
        if leles % 6 == 0:
            print(f'', int(leles / 6 * 20), '%')
        if leles == 30:
            leles = 0
        yield x, y


#################################################################################

batch_size = 32
epochs = 100
steps_per_epoch = 30  # int(x_train.shape[0]*x_train.shape[1]/batch_size)
val_steps = 10  # int(x_test.shape[0]*x_test.shape[1]/batch_size)
alpha = 0.3
num_hard = 16  # Number of semi-hard triplet examples in the batch
lr = 0.0001
optimiser = 'Adam'
emb_size = 128

with tf.device('GPU'):
    # Create the embedding model
    print('Generating embedding model... \n')
    embedding_model = create_embedding_model(emb_size)

    print('\n Generating SNN... \n')
    # Create the SNN
    siamese_net = create_SNN(embedding_model)
    # Compile the SNN
    optimiser_obj = Adam(lr=lr)
    siamese_net.compile(loss=triplet_loss, optimizer=optimiser_obj)

    # Store visualizations of the embeddings using PCA for display
    # Create represantations of the embedding space via PCA
    # embeddings_before_train = embedding_model.predict(x_test[0, :20, :, :])
    # He has loaded_emb_model instead of embedding_model because he loads the model from the logs
    # pca = PCA(n_components=2)
    # decomposed_embeddings_before = pca.fit_transform(embeddings_before_train)

#################################################################################

# Set up logging directory

name = "CustomCNN_final"
logdir = os.path.join(r'/home/giorgossykas', name) # Change directory

if not os.path.exists(logdir):
    os.mkdir(logdir)

# Callbacks:
# Create the TensorBoard callback
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=0,
    batch_size=batch_size,
    write_graph=True,
    write_grads=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0
)

# Training logger
csv_log = os.path.join(logdir, 'training.csv')
csv_logger = CSVLogger(csv_log, separator=',', append=True)

# Only save the best model weights based on the val_loss
checkpoint = ModelCheckpoint(os.path.join(logdir, 'snn_model-{epoch:02d}-{val_loss:.2f}.h5'),
                             monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='auto')


# Save the embedding model weights based on the main model's val loss
# This is needed to recreate the emebedding model should we wish to visualise
# the latent space at the saved epoch

class SaveEmbeddingModelWeights(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.Inf
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("SaveEmbeddingModelWeights requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.best:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.verbose == 1:
                print("Saving embedding model weights at %s" % filepath)
            embedding_model.save_weights(filepath, overwrite=True)
            self.best = current

            # Delete the last best emb_model and snn_model
            delete_older_model_files(filepath)


# Save the embedding model weights if you save a new snn best model based on the model checkpoint above
emb_weight_saver = SaveEmbeddingModelWeights(os.path.join(logdir, 'emb_model-{epoch:02d}.h5'))

callbacks = [tensorboard, csv_logger, checkpoint, emb_weight_saver]

# Save model configs to JSON
model_json = siamese_net.to_json()
with open(os.path.join(logdir, "siamese_config.json"), "w") as json_file:
    json_file.write(model_json)
    json_file.close()

model_json = embedding_model.to_json()
with open(os.path.join(logdir, "embedding_config.json"), "w") as json_file:
    json_file.write(model_json)
    json_file.close()

hyperparams = {'batch_size': batch_size,
               'epochs': epochs,
               'steps_per_epoch': steps_per_epoch,
               'val_steps': val_steps,
               'alpha': alpha,
               'num_hard': num_hard,
               'optimiser': optimiser,
               'lr': lr,
               'emb_size': emb_size
               }

with open(os.path.join(logdir, "hyperparams.json"), "w") as json_file:
    json.dump(hyperparams, json_file)

# Set the model to TB
tensorboard.set_model(siamese_net)


def delete_older_model_files(filepath):
    model_dir = filepath.split("emb_model")[0]

    # Get model files
    model_files = os.listdir(model_dir)
    # Get only the emb_model files
    emb_model_files = [file for file in model_files if "emb_model" in file]
    # Get the epoch nums of the emb_model_files
    emb_model_files_epoch_nums = [file.split("-")[1].split(".h5")[0] for file in emb_model_files]

    # Find all the snn model files
    snn_model_files = [file for file in model_files if "snn_model" in file]

    # Sort, get highest epoch num
    emb_model_files_epoch_nums.sort()
    highest_epoch_num = emb_model_files_epoch_nums[-1]

    # Filter the emb_model and snn_model file lists to remove the highest epoch number ones
    emb_model_files_without_highest = [file for file in emb_model_files if highest_epoch_num not in file]
    snn_model_files_without_highest = [file for file in snn_model_files if highest_epoch_num not in file]

    # Delete the non-highest model files from the subdir
    if len(emb_model_files_without_highest) != 0:
        print("Deleting previous best model file:", emb_model_files_without_highest)
        for model_file_list in [emb_model_files_without_highest, snn_model_files_without_highest]:
            for file in model_file_list:
                os.remove(os.path.join(model_dir, file))


#################################################################################

# siamese_net is already compiled

siamese_history = siamese_net.fit(
    data_generator(batch_size, num_hard),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=2,
    callbacks=callbacks,
    workers=0,
    validation_data=data_generator(batch_size, num_hard, split='test'),
    validation_steps=val_steps)

print('-------------------------------------------')
print('Training complete')