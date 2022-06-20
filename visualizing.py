
import matplotlib.pyplot as plt
import librosa.display
import processing



FILE = processing.imp_sound()

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

#show_spectrum(FILE)

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

def vis_chroma_freq(name: str):
    x, sr = librosa.load(name)
    hop_len = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr,
                                             hop_length=hop_len)  # compute chromagram from waveform or power spectogram
    plt.figure(figsize=(15,5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_len, cmap='coolwarm')
    plt.savefig('/Users/zhenyabudnyk/PycharmProjects/musicProject/photos/' + processing.name_retriever(name)+'_chroma_freq.png')



def spectral_centroid(sound_file):
    x, sr = processing.imp_sound(sound_file)
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    #print(spectral_centroids.shape)
    # computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # plotting the spectral centroid along the waveform
    librosa.display.waveshow(x, sr=sr, alpha=0.4)
    plt.plot(t, processing.normalize(spectral_centroids), color='r')
    plt.savefig()
show_zero_crossing(processing.PATH)
vis_chroma_freq(processing.PATH)




