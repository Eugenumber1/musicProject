
import matplotlib.pyplot as plt
import librosa.display
import processing


FILE = processing.imp_sound()

#this function shows the picture of the wave of the sound
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

show_spectrum(FILE)

def show_zero_crossing(name: str):
    x, sr = librosa.load(name)
    #plt.figure(figsize=(14,5))
    #librosa.display.waveshow(x, sr=sr)
    n0 = 7000
    n1 = 9100
    plt.figure(figsize=(14,5))
    plt.plot(x[n0:n1])
    plt.grid()
    #plt.show()




