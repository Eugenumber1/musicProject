import numpy as np
import pandas as pd
import sklearn.model_selection


# this method splits the variables in the data into dependent and independent ones
# X - independent variables, or features
# y - what we want to predict, music genre in our case
# we take name and track id out of the context as those have no relation to the independent variables
def split_variables(pathname):
    dataset = pd.read_csv(pathname)
    X = dataset.iloc[:,2 :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

# this method splits the data into the train and test data samples
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


# this method will do the data scalling (choosing between standardization and normalisation)
# for classification we need to scale only the numerical data, which is - independent variables
# the dependent variable is a class(string) so we can't scale it
# decision - to use standard scaler at first and evaluate the accuracy of the algo later
def scaler(X_train, X_test, standard_scaler=True):
    if standard_scaler is True:
        scaler = sklearn.preprocessing.StandardScaler()
    else:
        scaler = sklearn.preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


#print(split_variables('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv'))


# this method return fully preprocessed data, split into train and test sets, also scaled
def preprocess(path, not_decision_tree=True, standard_scaler=True):
    X, y = split_variables(path)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    if not_decision_tree == True:
        X_train, X_test = scaler(X_train, X_test, standard_scaler)
    return X_train, X_test, y_train, y_test

