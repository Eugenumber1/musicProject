import sklearn.linear_model as lm
import sklearn.ensemble
import numpy as np
import pandas as pd
import data_preprocessing
import knn_analyzer
import dimensionality_reduction as dr

# this method will train the decision tree classification on the data
# Note: Decision tree doesn't require data scaling, so we will update data_preprocessing.py
def logistic_regression_train(X_train, y_train):
    classifier = lm.LogisticRegression()
    classifier.fit(X_train, y_train)
    return classifier

# this method returns the predictions list

def logistic_regression_predict(classifier, X_test):
    return classifier.predict(X_test)

# run method runs everything
def run(path):
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess(path, not_decision_tree=False, standard_scaler=False)
    print(knn_analyzer.accuracy_estimator(y_test, logistic_regression_predict(logistic_regression_train(dr.lda(X_train, X_test, y_train)[0], y_train),
                                                                              dr.lda(X_train, X_test, y_train)[1])))


run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')
