import numpy as np

import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording
sd.default.samplerate = fs

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

write('output.wav', fs, myrecording)  # Save as WAV file
