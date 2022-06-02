import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt

filename = librosa.example('nutcracker')

# load audio as waveform
#y, sr = librosa.load(filename)
PATH = "/Users/zhenyabudnyk/Downloads/636402__klankbeeld__estate-nl-1156am-220509-0343.wav"
PATH2 = "/Users/zhenyabudnyk/Downloads/Autotune - Blade Runner (Original) [Official Audio] (128 kbps).mp3"
y, sr = librosa.load(PATH)

#sr - smapling rate
#y - time series - NumPy floating point array one dimensional

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#output here is beat per minute
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

#beat times - array of timestamps corresponding to beat events

#print(y)
#print(sr)
#print(beat_times)
#print(type(y), type(sr))
ipd.Audio(filename=PATH2)


#print(y.shape, sr)

def imp_sound(path=PATH):
    y, sr = librosa.load(path)
    return y, sr


