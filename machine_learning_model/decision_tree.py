import sklearn.tree
import sklearn.ensemble
import numpy as np
import pandas as pd
import data_preprocessing
import knn_analyzer

# this method will train the decision tree classification on the data
# Note: Decision tree doesn't require data scaling, so we will update data_preprocessing.py
def decision_tree_train(X_train, y_train):
    classifier = sklearn.tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

# this method will use the random forest classifier, which is the collection of the
# different decision trees by its nature
def random_forest_train(X_train, y_train):
    classifier = sklearn.ensemble.RandomForestClassifier(criterion='gini', random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

# this method returns the predictions list

def decision_tree_predict(classifier, X_test):
    return classifier.predict(X_test)

# run method runs everything
def run(path):
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess(path, not_decision_tree=False)
    print(knn_analyzer.accuracy_estimator(y_test, decision_tree_predict(random_forest_train(X_train, y_train), X_test)))


run('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')