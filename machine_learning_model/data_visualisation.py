import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# this method loads the data and shows it on scatter plot
def scatter_plot(path):
    dataframe = pd.read_csv(path)
    #print(dataframe.head(999))
    plt.scatter(dataframe['spectral rollof variance'][(dataframe.genre == 'pop') | (dataframe.genre == 'pop')],
                dataframe['chroma frequncy mean'][(dataframe.genre == 'pop') | (dataframe.genre == 'pop')],
                marker='D',
                color='red',
                label='Pop')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'blues'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'blues'],
                marker='o',
                color='blue',
                label='Blues')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'jazz'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'jazz'],
                marker='o',
                color='grey',
                label='Jazz')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'rock'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'rock'],
                marker='o',
                color='orange',
                label='Rock')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'reggae'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'reggae'],
                marker='o',
                color='pink',
                label='Regggae')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'metal'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'metal'],
                marker='o',
                color='yellow',
                label='Metal')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'disco'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'disco'],
                marker='o',
                color='purple',
                label='Disco')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'classical'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'classical'],
                marker='o',
                color='green',
                label='Classical')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'hiphop'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'hiphop'],
                marker='o',
                color='black',
                label='Hiphop')
    plt.scatter(dataframe['spectral rollof variance'][dataframe.genre == 'country'],
                dataframe['chroma frequncy mean'][dataframe.genre == 'country'],
                marker='o',
                color='brown',
                label='Country')
    plt.xlabel('spectral rollof variance')
    plt.ylabel('chroma frequncy mean')
    plt.legend()
    plt.show()


scatter_plot('/Users/zhenyabudnyk/PycharmProjects/musicProject/features.csv')
