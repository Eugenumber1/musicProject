import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import sklearn

filename = librosa.example('nutcracker')

# load audio as waveform
#y, sr = librosa.load(filename)
PATH = "/Users/zhenyabudnyk/Downloads/636402__klankbeeld__estate-nl-1156am-220509-0343.wav"
PATH2 = "/Users/zhenyabudnyk/Downloads/Autotune - Blade Runner (Original) [Official Audio] (128 kbps).mp3"
PATH3 = '/Users/zhenyabudnyk/Downloads/Music-Genre-Classification-with-Python-master_simple_loop.wav'
#y, sr = librosa.load(PATH)
#print(sr)

#sr - smapling rate
#y - time series - NumPy floating point array one dimensional

#tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#output here is beat per minute
#beat_times = librosa.frames_to_time(beat_frames, sr=sr)

#beat times - array of timestamps corresponding to beat events

#print(y)
#print(sr)
#print(beat_times)
#print(type(y), type(sr))
#ipd.Audio(filename=PATH2)


#print(y.shape, sr)

#this function loads the audio file and returns the time-serises and sampling rate
def imp_sound(path=PATH):
    y, sr = librosa.load(path)
    return y, sr

#this function is writing the numpy array and sampling rate to the filename
def write_audio(name: str, y, sr):
   sf.write(name+'.wav', y, sr, subtype='PCM_24')

#this function creates the audio signal
#input variables sr - sample rate and time, how long the audio should be in seconnd
def create_audio(sr, time, name):
    t = np.linspace(0, time, int(time*sr)) # time var, linspace makes the array with start, end and number of steps generated
    print(t)
    x = 0.5*np.sin(2*np.pi*200*t) # pure sine wave at 220 Hz
    print(x)
    ipd.Audio(x, rate=sr) # load np array
    write_audio(name, x, sr)

#create_audio(22050, 5.0, 'meme6')


#normalising the spectral centroid visualization
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)








