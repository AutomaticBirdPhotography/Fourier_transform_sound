# Inspired from https://docs.scipy.org/doc/scipy/tutorial/fft.html

from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

samplerate, data = wavfile.read('cell2.wav')
print(f"Samplerate = {samplerate}")

N = data.shape[0]
duration = N / samplerate
T = 1 / samplerate

k = np.linspace(0.0, duration, N, endpoint=False)

if len(data.shape) > 1:
    print("Data has multiple channels. Using left channel.")
    f = data[:,0]
else:
    f = data

# Rfft to simply get the real values (so we dont have to devide by two)
F = rfft(f)
modulF = np.abs(F)
frequency = rfftfreq(N, T)

# Mutliply by 1.0/N since: we actually must multiply by 1/N since the fft simply adds the numbers (so it increases by the number of points N. 2.0 since we only has the upper half
plt.plot(frequency, 2.0/N * modulF, label="Fast fourier transform")
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

