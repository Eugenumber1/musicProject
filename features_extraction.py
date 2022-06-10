import librosa
import processing
import visualizing
import matplotlib.pyplot as plt
import sklearn
# zero-crossing Rate
# to visualize the zero crossing
def zero_crossing(sound_file):
    x, sr = processing.imp_sound(sound_file)
    zero_crossings = librosa.zero_crossings(x, pad=False)
    visualizing.show_zero_crossing(sound_file)
    #print(zero_crossings)
    #print(sum(zero_crossings))
    return sum(zero_crossings)

# spectral centroid - indicates where the center of mass for a sound is located
# calculated as the weighted mean of the frequencies present in the sound
# So spectral centroid for blues song will lie somewhere near the middle of
# its spectrum while that for a metal song would be towards its end.
#TODO the function doesnt' retrurn any particular feature, just draws the graph
def spectral_centroid(sound_file):
    x, sr = processing.imp_sound(sound_file)
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    print(spectral_centroids.shape)
    # computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # plotting the spectral centroid along the waveform
    librosa.display.waveshow(x, sr=sr, alpha=0.4)
    plt.plot(t, processing.normalize(spectral_centroids), color='r')
    plt.show()


# it is the measure of the shape of the signal, it shows the frequency below which
# a specified percentage of the total spectral energy lies
#TODO define what will be returned as a feature
def spectral_rollof(sound_file):
    x, sr = processing.imp_sound(sound_file)
    spectral_rollof = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    librosa.display.waveshow(x, sr=sr, alpha=0.4)
    frames = range(len(spectral_rollof))
    t = librosa.frames_to_time(frames)
    plt.plot(t, processing.normalize(spectral_rollof), color='r')
    plt.show()

#small set of features that describe the overall shape of a spectral envelope
#TODO learn what is standardization doing here and find the return value
def mel_coef(sound_file):
    x, sr = processing.imp_sound(sound_file)
    visualizing.show_wave((x, sr))
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1) #standardization of the variables
    print(mfccs.mean(axis=1))
    print(mfccs.var(axis=1))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()


#chroma frequencies - entire spectrum is projected onto 12 bins representing 12 distinct semitones
#TODO choose what it returns
def chroma_freq(sound_file):
    x, sr = processing.imp_sound(sound_file)
    hop_len = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_len) # compute chromagram from waveform or power spectogram
    plt.figure(figsize=(15,5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_len, cmap='coolwarm')
    plt.show()








chroma_freq(processing.PATH3)
#spectral_rollof(processing.PATH)
#spectral_centroid(processing.PATH)
#mel_coef(processing.PATH3)
#zero_crossing(processing.PATH)
