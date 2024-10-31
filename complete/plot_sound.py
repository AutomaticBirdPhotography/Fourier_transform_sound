import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf  # To read the audio file


# Waveform Visualization
def plot_waveform(filename):
    # Load the audio file
    signal, sample_rate = sf.read(filename)

    # If stereo, take one channel
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Create time axis
    duration = len(signal) / sample_rate
    time = np.linspace(0, duration, len(signal))

    # Plot waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)
    plt.title("Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


# Usage example
# plot_waveform("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import soundfile as sf


# Frequency Spectrum Visualization
def plot_frequency_spectrum(filename):
    # Load the audio file
    signal, sample_rate = sf.read(filename)

    # If stereo, take one channel
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Number of samples
    N = len(signal)

    # Compute FFT
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)

    # Only take the positive frequencies
    idx = np.where(xf >= 0)
    xf = xf[idx]
    yf = np.abs(yf[idx])

    # Plot frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()


# Usage example
# plot_frequency_spectrum("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import soundfile as sf


# Spectrogram Visualization
def plot_spectrogram(filename):
    # Load the audio file
    signal, sample_rate = sf.read(filename)

    # If stereo, take one channel
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Compute STFT
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=1024)

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()


# Usage example
# plot_spectrogram("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Mel Spectrogram Visualization
def plot_mel_spectrogram(filename):
    # Load the audio file
    signal, sample_rate = librosa.load(filename, sr=None, mono=True)

    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot Mel spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S_dB, sr=sample_rate, hop_length=512, x_axis="time", y_axis="mel"
    )
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


# Usage example
# plot_mel_spectrogram("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Chromagram Visualization
def plot_chromagram(filename):
    # Load the audio file
    signal, sample_rate = librosa.load(filename, sr=None, mono=True)

    # Compute Chromagram
    chroma = librosa.feature.chroma_stft(
        y=signal, sr=sample_rate, n_fft=4096, hop_length=512
    )

    # Plot Chromagram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        chroma, x_axis="time", y_axis="chroma", hop_length=512, cmap="coolwarm"
    )
    plt.title("Chromagram")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# Usage example
# plot_chromagram("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import soundfile as sf
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


# Waterfall Spectrogram Visualization
def plot_waterfall_spectrogram(filename):
    # Load the audio file
    signal, sample_rate = sf.read(filename)

    # If stereo, take one channel
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Compute STFT
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=1024)

    # Convert magnitude to decibels
    Zxx_magnitude = np.abs(Zxx)
    Zxx_magnitude_db = 20 * np.log10(Zxx_magnitude + 1e-6)

    # Prepare data for plotting
    T, F = np.meshgrid(t, f)

    # Create the waterfall plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    ax.plot_surface(
        T, F, Zxx_magnitude_db, cmap="viridis", linewidth=0, antialiased=False
    )

    # Customize the axes
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_zlabel("Magnitude [dB]")
    ax.set_title("Waterfall Spectrogram")

    # Adjust the view angle
    ax.view_init(elev=45, azim=-120)

    plt.tight_layout()
    plt.show()


# Usage example
# plot_waterfall_spectrogram("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
import librosa


# Zero-Crossing Rate Visualization
def plot_zero_crossing_rate(filename):
    # Load the audio file
    signal, sample_rate = librosa.load(filename, sr=None, mono=True)

    # Compute Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512)

    # Create time axis
    frames = range(len(zcr[0]))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=512)

    # Plot Zero-Crossing Rate
    plt.figure(figsize=(12, 4))
    plt.plot(t, zcr[0])
    plt.title("Zero-Crossing Rate")
    plt.xlabel("Time [s]")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.show()


# Usage example
# plot_zero_crossing_rate("your_audio_file.wav")


import numpy as np
import matplotlib.pyplot as plt
import librosa


# Root Mean Square Energy Visualization
def plot_rms_energy(filename):
    # Load the audio file
    signal, sample_rate = librosa.load(filename, sr=None, mono=True)

    # Compute RMS Energy
    rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=512)

    # Create time axis
    frames = range(len(rms[0]))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=512)

    # Plot RMS Energy
    plt.figure(figsize=(12, 4))
    plt.plot(t, rms[0])
    plt.title("RMS Energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.show()


# Usage example
# plot_rms_energy("your_audio_file.wav")


def visualize_audio(filename):
    print("Choose a visualization:")
    print("1. Waveform")
    print("2. Frequency Spectrum")
    print("3. Spectrogram")
    print("4. Mel Spectrogram")
    print("5. Chromagram")
    print("6. Waterfall Spectrogram")
    print("7. Zero-Crossing Rate")
    print("8. RMS Energy")
    choice = input("Enter the number of your choice: ")

    if choice == "1":
        plot_waveform(filename)
    elif choice == "2":
        plot_frequency_spectrum(filename)
    elif choice == "3":
        plot_spectrogram(filename)
    elif choice == "4":
        plot_mel_spectrogram(filename)
    elif choice == "5":
        plot_chromagram(filename)
    elif choice == "6":
        plot_waterfall_spectrogram(filename)
    elif choice == "7":
        plot_zero_crossing_rate(filename)
    elif choice == "8":
        plot_rms_energy(filename)
    else:
        print("Invalid choice.")


# Usage example
filename = "signal.wav"
visualize_audio(filename)
