import numpy as np
import pandas as pd
import sklearn.model_selection


# this method splits the variables in the data into dependent and independent ones
# X - independent variables, or features
# y - what we want to predict, music genre in our case
def split_variables(pathname):
    dataset = pd.read_csv(pathname)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

# this method splits the data into the train and test data samples
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test



#print(split_variables('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv'))

print(split_train_test(split_variables('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')[0],
                       split_variables('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')[1]))
