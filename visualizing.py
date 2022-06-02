import matplotlib_inline
from jedi.api.refactoring import inline
import matplotlib.pyplot as plt
import librosa.display
import processing

plt.figure(figsize=(14,5))
librosa.display.waveshow(processing.imp_sound()[0], sr=processing.imp_sound()[1])
plt.show()