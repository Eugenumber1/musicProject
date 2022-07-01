import csv
import os

from features_extraction import *
import processing
import re
from track import Track

HEADER = ['track_id', 'name', 'zero crossings', 'spectral centroid variance', 'spectral centroid mean', 'spectral rollof variance', 'spectral rollof mean',
          'mel coefficient variance', 'mel coefficient mean', 'chroma frequency variance', 'chroma frequncy mean', 'spectral bandwidth variance',
          'spectral bandwidth mean', 'genre']

track1 = (processing.PATH, 'alternative rock')
track2 = (processing.PATH2, 'tech house')
track3 = (processing.PATH3, 'alternative rock')
track4 = (processing.PATH4, 'rap')
tracks = (track1, track2, track3, track4)
track5 = Track(path='/Users/zhenyabudnyk/PycharmProjects/musicProject/sox_transformer/genres_original/hiphop/hiphop.00002.wav')

# this method is doing parsing through the folders of the genres folder and through their music files
def parser(dir_path):
    tracks = list()
    for folder in os.listdir('/Users/zhenyabudnyk/PycharmProjects/musicProject/sox_transformer/genres_original'):
        for song in os.listdir('/Users/zhenyabudnyk/PycharmProjects/musicProject/sox_transformer/genres_original'+'/'+ folder):
            new_track = Track('/Users/zhenyabudnyk/PycharmProjects/musicProject/sox_transformer/genres_original'+'/'+ folder + '/' + song)
            new_track.setGenre(folder)
            tracks.append(new_track)
    return tracks





# method which generates the list of data
# the data list will be used to make csv files later
def data_collector(track):
    data = list()
    data.append(processing.name_retriever(track[0]))
    zc = zero_crossing(track[0])
    data.append(zc)
    print(f'here is {zc}')
    scv, scm = spectral_centroid(track[0])
    data.append(scv)
    data.append(scm)
    print(f'here is {scv}')
    print(f'here is {scm}')
    srv, srm = spectral_rollof(track[0])
    data.append(srv)
    data.append(srm)
    print(f'here is {srv}')
    print(f'here is {srm}')
    mcv, mcm = mel_coef(track[0])
    data.append(mcv)
    data.append(mcm)
    print(f'here is {mcv}')
    print(f'here is {mcm}')
    cfv, cfm = chroma_freq(track[0])
    data.append(cfv)
    data.append(cfm)
    print(f'here is {cfv}')
    print(f'here is {cfm}')
    sbv, sbm = spectral_bandwidth(track[0])
    data.append(sbv)
    data.append(sbm)
    print(f'here is {sbv}')
    print(f'here is {sbm}')
    data.append(track[1])
    return data

# this method inputs list of tracks and returns makes the csv file with all the features
# features are not scaled yet
# adds the track id to the csv file
def writer(tracks):
    with open('features.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i in tracks:
            data = data_collector(i)
            data.insert(0, tracks.index(i)+1)
            print(data)
            writer.writerow(data)

#writer(tracks)

for i in parser('/Users/zhenyabudnyk/PycharmProjects/musicProject/sox_transformer/genres_original'):
    print(i.name)
