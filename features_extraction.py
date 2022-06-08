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


spectral_centroid(processing.PATH)
#zero_crossing(processing.PATH)
