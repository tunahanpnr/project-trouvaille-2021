import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random
import keras.backend as K
from pydub import AudioSegment

os.makedirs('./content/spectrograms3sec')
os.makedirs('./content/spectrograms3sec/train')
os.makedirs('./content/spectrograms3sec/test')

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
  path_audio = os.path.join('./content/audio3sec',f'{g}')
  os.makedirs(path_audio)
  path_train = os.path.join('./content/spectrograms3sec/train',f'{g}')
  path_test = os.path.join('./content/spectrograms3sec/test',f'{g}')
  os. makedirs(path_train)
  os. makedirs(path_test)

i = 0
for g in genres:
    j = 0
    print(f"{g}")
    for filename in os.listdir(os.path.join('./data/genres_original/', f"{g}")):

        song = os.path.join(f'./data/genres_original/{g}', f'{filename}')
        j = j + 1
        for w in range(0, 10):
            i = i + 1
            # print(i)
            t1 = 3 * (w) * 1000
            t2 = 3 * (w + 1) * 1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(f'./content/audio3sec/{g}/{g + str(j) + str(w)}.wav', format="wav")

for g in genres:
    j = 0
    print(g)
    for filename in os.listdir(os.path.join('./content/audio3sec', f"{g}")):
        song = os.path.join(f'./content/audio3sec/{g}', f'{filename}')
        j = j + 1

        y, sr = librosa.load(song, duration=3)
        # print(sr)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
        plt.savefig(f'./content/spectrograms3sec/train/{g}/{g + str(j)}.png')

directory = "./content/spectrograms3sec/train/"
for g in genres:
    filenames = os.listdir(os.path.join(directory, f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:100]

    for f in test_files:
        shutil.move(directory + f"{g}" + "/" + f, "/content/spectrograms3sec/test/" + f"{g}")

train_dir = "./content/spectrograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(288, 432), color_mode="rgba",
                                                    class_mode='categorical', batch_size=128)

validation_dir = "./content/spectrograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1. / 255)
vali_generator = vali_datagen.flow_from_directory(validation_dir, target_size=(288, 432), color_mode='rgba',
                                                  class_mode='categorical', batch_size=128)


def GenreModel(input_shape=(288, 432, 4), classes=9):
    X_input = Input(input_shape)

    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Flatten()(X)

    X = Dropout(rate=0.3)

    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs=X_input, outputs=X, name='GenreModel')

    return model


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


model = GenreModel(input_shape=(288, 432, 4), classes=9)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', get_f1])

model.fit_generator(train_generator, epochs=70, validation_data=vali_generator)
