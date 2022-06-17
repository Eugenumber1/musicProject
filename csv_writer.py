import csv
from features_extraction import *
import processing

HEADER = ['name', 'zero crossings', 'spectral centroid variance', 'spectral centroid mean', 'spectral rollof variance', 'spectral rollof mean',
          'mel coefficient variance', 'mel coefficient mean', 'chroma frequency variance', 'chroma frequncy mean', 'spectral bandwidth variance',
          'spectral bandwidth mean', 'genre']

track1 = (processing.PATH, 'alternative rock')
track2 = (processing.PATH2, 'tech house')
track3 = (processing.PATH3, 'alternative rock')
track4 = (processing.PATH4, 'rap')
tracks = (track1, track2, track3, track4)

# method which generates the list of data
# the data list will be used to make csv files later

def data_collector(track):
    data = list()
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

data_collector(tracks[0])
