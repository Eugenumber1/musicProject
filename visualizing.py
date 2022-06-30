
import matplotlib.pyplot as plt
import librosa.display
import processing
import sklearn
import threading

#this function shows the picture of the wave of the sound
# arguments = tuple of x and sr
def show_wave(sound_file):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(sound_file[0], sr=sound_file[1])
    plt.show()

#show_wave(FILE)

#this function makes the picture of the spectrogram, time to log
def show_spectrum(sound_file):
    X = librosa.stft(sound_file[0]) # make the signal in the time-frequency domain by computing DFT(descrete fourier transforms)
    Xdb = librosa.amplitude_to_db(abs(X)) # convert amplitude to db scaled spectrogram
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sound_file[1], x_axis='time', y_axis='log') # show spectrogram
    plt.colorbar()
    plt.show()


def show_zero_crossing(name: str):
    x, sr = librosa.load(name)
    #plt.figure(figsize=(14,5))
    #librosa.display.waveshow(x, sr=sr)
    n0 = 0
    n1 = 100000
    plt.figure(figsize=(14,5))
    plt.plot(x[n0:n1])
    plt.grid()
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_zero_crossings.png')
    plt.close()

def show_chroma_freq(name: str):
    x, sr = librosa.load(name)
    hop_len = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr,
                                             hop_length=hop_len)  # compute chromagram from waveform or power spectogram
    plt.figure(figsize=(15,5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_len, cmap='coolwarm')
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_chroma_freq.png')
    plt.close()



def show_spectral_centroid(name):
    x, sr = processing.imp_sound(name)
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    #print(spectral_centroids.shape)
    # computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # plotting the spectral centroid along the waveform
    librosa.display.waveshow(x, sr=sr, alpha=0.4)
    plt.plot(t, processing.normalize(spectral_centroids), color='r')
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_spectral_centroid.png')
    plt.close()

def show_spectral_rollof(name):
    x, sr = processing.imp_sound(name)
    spectral_rollof = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    librosa.display.waveshow(x, sr=sr, alpha=0.4)
    frames = range(len(spectral_rollof))
    t = librosa.frames_to_time(frames)
    plt.plot(t, processing.normalize(spectral_rollof), color='r')
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_spectral_rollof.png')
    plt.close()

def show_mel_coef(name):
    x, sr = processing.imp_sound(name)
    #visualizing.show_wave((x, sr))
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1) #standardization of the variables
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_mel_coef.png')
    plt.close()


def show_spectral_bandwidth(name):
    y, sr = processing.imp_sound(name)
    band = librosa.feature.spectral_bandwidth(y, sr)
    times = librosa.times_like(band)
    plt.semilogy(times, band[0], label='Spectral bandwidth')
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_spect_band.png')
    plt.close()


# this method makes pictures of all diagrams and saves them

def visualize(name):
    show_zero_crossing(name)
    show_chroma_freq(name)
    show_spectral_centroid(name)
    show_spectral_rollof(name)
    show_mel_coef(name)
    show_spectral_bandwidth(name)


visualize(processing.PATH)










