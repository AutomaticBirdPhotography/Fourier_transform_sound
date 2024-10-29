import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft, get_window
from scipy.io import wavfile

samplerate, data = wavfile.read('scale.wav')
print(f"samplerate = {samplerate}Hz")

# Number of samples 
N = data.shape[0]
print(f"Number of samples = {N}")

# Make sure data is mono
if len(data.shape) > 1:
    print("Data has multiple channels. Using left channel.")
    data = data[:,0]

# The signal amplitude as a function of time
# Convert the signal to float and normalize
signal = data.astype(float)
signal /= np.max(np.abs(signal))

# STFT parameters
window_length = 7000 # Number of samples in each window
hop_size = window_length // 4 # 75% overlap
noverlap = window_length - hop_size # Number of samples overlapping between windows

# Use a Hann window
window = get_window('hann', window_length)


# Compute the STFT of the signal
frequencies, times, Zxx = stft(signal, fs=samplerate, window=window,
                               nperseg=window_length, noverlap=noverlap, nfft=window_length)
# Zxx is the STFT matrix

# Convert the magnitude to decibels
Zxx_magnitude = np.abs(Zxx)
Zxx_magnitude_db = 20 * np.log10(Zxx_magnitude + 1e-6)  # Add small value to avoid log(0)

# Plot the STFT
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, Zxx_magnitude_db, shading='gouraud', cmap='viridis')
plt.title('Short-Time Fourier Transform (STFT) Magnitude')
plt.text(0.05, 0.95, f'Window length: {window_length} samples', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude [dB]')
plt.ylim(0, 8000)  # Limit the frequency range for better visualization
plt.tight_layout()
plt.show()