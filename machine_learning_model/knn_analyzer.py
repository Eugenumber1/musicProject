from sklearn import neighbors
import numpy as np
import pandas as pd
from data_preprocessing import *

# this method is performs training of classification with KNN model
def knn_classifier(X_train, y_train):
    classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
    classifier.fit(X_train, y_train)
    return classifier

# this method does the prediction of the X_test
def knn_predicter(classifier, X_test):
    predictions = classifier.predict(X_test)
    return predictions

# this method returns the confusion matrix and accuracy
def accuracy_estimator(y_test, predictions):
    return (sklearn.metrics.confusion_matrix(y_test, predictions)), sklearn.metrics.accuracy_score(y_test, predictions)

def run(path):
    X_train, X_test, y_train, y_test = preprocess(path, standard_scaler=True)
    print(accuracy_estimator(y_test, knn_predicter(knn_classifier(X_train, y_train), X_test)))


run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')

