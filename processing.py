import librosa

filename = librosa.example('nutcracker')

# load audio as waveform
#y, sr = librosa.load(filename)
y, sr = librosa.load("/Users/zhenyabudnyk/Downloads/636402__klankbeeld__estate-nl-1156am-220509-0343.wav")

#sr - smapling rate
#y - time series - NumPy floating point array one dimensional

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#output here is beat per minute
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

#beat times - array of timestamps corresponding to beat events

print(y)
print(sr)
print(beat_times)

