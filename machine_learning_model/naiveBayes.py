import sklearn.naive_bayes
import pandas as pd
import numpy as np
import data_preprocessing
import knn_analyzer

# this method will train the Naive bayes classifier
def bayes_train(X_train, y_train):
    classifier = sklearn.naive_bayes.GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

# this method will make the list of predictions
def bayes_predict(classifier, X_test):
    predictions = classifier.predict(X_test)
    return predictions

# this method will just run all methods one in one and add the evaluation method from the knn python file
def run(path):
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess(path)
    print(knn_analyzer.accuracy_estimator(y_test, bayes_predict(bayes_train(X_train, y_train), X_test)))

run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')