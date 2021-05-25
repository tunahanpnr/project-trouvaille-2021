import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
# Preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Keras
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from src.models import run_exps


class Trouvaille:
    def __init__(self):
        self.model = models.Sequential()
        self.header = prep_header()

    def create_dataset(self):
        file = open('data.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(self.header)

        genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        for g in genres:
            for filename in os.listdir(f'./content/audio3sec/{g}'):
                songname = f'./content/audio3sec/{g}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=30)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rmse = librosa.feature.rms(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rmse)} {np.var(rmse)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)}'
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {g}'
                file = open('data.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

    def preprocessing_dataset(self):
        data = pd.read_csv('data.csv')
        data.head()

        # Dropping unneccesary columns
        data = data.drop(['filename'], axis=1)
        data.head()

        genre_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(genre_list)

        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

        pca = PCA(n_components=7)
        pca.fit(X)
        print(pca.explained_variance_ratio_)

        X, y = unison_shuffled_copies(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def create_model(self):
        self.model.add(layers.Dense(512, activation='relu', input_shape=(self.X_train.shape[1],)))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def training_model(self):
        self.model.fit(self.X_train,
                       self.y_train,
                       epochs=30,
                       batch_size=128)

        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('test_acc: ', test_acc)

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("default_model.h5")

    def predict(self):
        predictions = self.model.predict(self.X_test)
        high = np.argmax(predictions[0])
        print(high)

    def run(self):
        # print("create_dataset()")
        # self.create_dataset()

        print("preprocessing_dataset()")
        self.preprocessing_dataset()

        run_exps(self.X_train, self.y_train, self.X_test, self.y_test)


        # print("create_model()")
        # self.create_model()
        #
        # print("training_model()")
        # self.training_model()
        #
        # print("predict()")
        # self.predict()



def prep_header():
    header = 'filename chroma_stft_mean chroma_stft_var rmse_mean rmse_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean pectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    return header.split()


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
