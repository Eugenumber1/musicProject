import pandas as pd
from sklearn.preprocessing import LabelEncoder

import data_preprocessing
from tensorflow import keras
from knn_analyzer import accuracy_estimator
import numpy as np

# this method will build an artificial neural network
def ann_make(X_train, y_train):
    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(units=128, input_shape=(None, 11), activation='relu'))
    ann.add(keras.layers.Dense(units=512, activation='relu'))
    ann.add(keras.layers.Dense(units=10, activation='softmax'))
    ann.add(keras.layers.Flatten())
    ann.summary()
    ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, batch_size=32, epochs=100)
    return ann

# this method turns the y labels into numerical values from categorical
# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

def run(path):
    #df = pd.read_csv(path)
    #df.info()
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess(path, standard_scaler=True)
    #print(X_train)
    #print(y_train)
    #X_train = X_train.reshape(1, -1)
    #X_test = X_test.reshape(1, -1)
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    y_train = y_train_enc.reshape(-1, 1)
    y_test = y_test_enc.reshape(-1, 1)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    #print(X_train)
    #print(y_train)
    ann = ann_make(X_train, y_train)
    predictions = ann.predict(X_test)
    #print(predictions)
    #print(accuracy_estimator(y_test, predictions))
    #ann.evaluate()


run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')

