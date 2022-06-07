import librosa
import processing
# zero-crossing Rate
# to visualize the zero crossing
def zero_crossing(sound_file):
    x, sr = processing.imp_sound(sound_file)
    zero_crossings = librosa.zero_crossings(x, pad=False)
    print(zero_crossings)
    print(sum(zero_crossings))
    return sum(zero_crossings)

zero_crossing(processing.PATH)
