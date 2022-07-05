import pandas as pd
import numpy as np
import sklearn
import knn_analyzer
import data_preprocessing

# this method gets the training data and trains the classifier, returns the classifier as well
def svm_train(X_train, y_train):
    clf = sklearn.svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    return clf


# in this method the model predicts the classes from the test data
def svm_predict(clf, X_test):
    predictions = clf.predict(X_test)
    return predictions

# this method just runs the execution
def run(path):
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess(path)
    print(knn_analyzer.accuracy_estimator(y_test, svm_predict(svm_train(X_train, y_train), X_test)))

run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')

