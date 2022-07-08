import data_preprocessing
from tensorflow import keras
from knn_analyzer import accuracy_estimator

# this method will build an artificial neural network
def ann_make(layers, X_train, y_train):
    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(units=16, input_shape=(1, ) ,activation='relu'))
    while layers > 0:
        ann.add(keras.layers.Dense(units=32, activation='relu'))
        layers = layers - 1
    ann.add(keras.layers.Dense(units=10, activation='softmax'))
    ann.summary()
    ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, batch_size=32, epochs=100)
    return ann

def run(path):
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess(path, standard_scaler=True)
    ann = ann_make(1, X_train, y_train)
    predictions = ann.predict(X_test)
    print(accuracy_estimator(y_test, predictions))


run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')

