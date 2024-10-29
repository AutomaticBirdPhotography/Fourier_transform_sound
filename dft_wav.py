import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

samplerate, data = wavfile.read('cell2.wav')
# data.shape = [Nsamples, Nchannels] I dette eksemplet brukes monolyd, så Nchannels = 1
print(f"samplerate = {samplerate}Hz")

N = data.shape[0]
duration = N / samplerate
T = 1 / samplerate
print(f"duration = {duration}s")

if len(data.shape) > 1:
    print("Data has two channels. Using left channel.")
    f = data[:,0]
else:
    f = data

N = len(f)
W = np.exp(-1j * (2 * np.pi / N))

A = np.zeros((N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        A[k][n] = W**(k*n)

F = A @ f
modulF = np.zeros(N)   # Modulen til F. |F[k]|
for k, z in enumerate(F):
    modulF[k] = np.sqrt((z.real)**2 + (z.imag)**2)

# Plotter amplitude som funksjon av tid
time = np.linspace(0., duration, N)
plt.plot(time, f, label="Sound")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# Plotter amplitude som funksjon av frekvens
k = np.arange(N//2)
frequency = k / duration
plt.plot(frequency, 2.0/N * modulF[:N//2], label="Discrete fourier transform")
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
