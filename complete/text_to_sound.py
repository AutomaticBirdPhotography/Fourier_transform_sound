# Objectives:
# 1. Convert ASCII characters to binary values represnted in bytes.
# 2. Represent each byte as a combination of 8 frequencies. Then make a signal out of these multi-friquency byte representations.
# 3. Save full signal to a .wav file
# 4. Plot the Time-Domain Signal
# 5. Plot Frequency-Domain (FFT) for Each Segment

import numpy as np  # Part 2
import soundfile as sf  # Part 3
import matplotlib.pyplot as plt  # Part 4 & 5


def text_to_sound(text = "hello", duration=0.1, base_frequency = 1000, sample_rate = 44100, filename="signal.wav"):
    segments = generate_signal_segments(
        text,
        sample_rate=sample_rate,
        base_freq=base_frequency,
        duration=duration,
    )

    # Save the full signal to a .wav file
    save_signal_to_file(segments, filename, sample_rate=44100)

    # # Plot time-domain signal
    # plot_time_domain_signal(segments)

    # # Plot frequency-domain signal for each segment
    # plot_frequency_domain_segments(segments)


# Part 1: Convert ASCII Characters to Binary Values Represented in Bytes


def ascii_to_binary(ascii_text):
    # Convert each character to 8-bit binary represention
    return "".join(format(ord(char), "08b") for char in ascii_text)


# Part 2: Represent Each Byte as a Combination of 8 Frequencies


def byte_to_frequencies(byte, base_freq=100, sample_rate=44100, duration=0.1):
    # Create a multi-frequency signal for each byte
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for i, bit in enumerate(byte):
        if bit == "1":
            freq = base_freq * (i + 1)
            sine_wave = np.sin(2 * np.pi * freq * t)
            signal += sine_wave
    return signal


def generate_signal_segments(
    ascii_text, sample_rate=44100, base_freq=100, duration=0.1
):
    binary_representation = ascii_to_binary(ascii_text)
    # print("Binary representation:", binary_representation)
    # print("Decimal representation:", int(binary_representation, 2))
    segments = [
        byte_to_frequencies(
            binary_representation[i : i + 8], base_freq, sample_rate, duration
        )
        for i in range(0, len(binary_representation), 8)
    ]

    return segments  # Return list of segments for each character


# Part 3: Save signal to file


def save_signal_to_file(segments, filename="signal.wav", sample_rate=44100):
    # Concatenate all segments into a single 1D array
    full_signal = np.concatenate(segments)

    # Ensure that the signal is within a typical range for .wav files
    # Normalize to the range -1 to 1
    full_signal /= np.max(np.abs(full_signal) + 1e-6)
    # Write full signal to a .wav file
    sf.write(filename, full_signal, sample_rate)

    print(f"Signal generated and saved in the file: '{filename}'")


# Part 6: Plot the Time-Domain Signal


def plot_time_domain_signal(segments, sample_rate=44100):
    # Concatenate all segments for full time-domain view
    full_signal = np.concatenate(segments)
    time = np.linspace(0, len(full_signal) / sample_rate, len(full_signal))

    plt.figure(figsize=(12, 4))
    plt.plot(time, full_signal)
    plt.title("Time-Domain Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


# Part 7: Plot Frequency-Domain (FFT) for Each Segment


def plot_frequency_domain_segments(segments, sample_rate=44100, base_freq=100):
    plt.figure(figsize=(12, 6))

    for idx, segment in enumerate(segments):
        # Compute FFT
        N = len(segment)
        fft_result = np.fft.fft(segment)
        frequencies = np.fft.fftfreq(N, d=1 / sample_rate)
        positive_frequencies = frequencies[: N // 2]
        positive_fft_result = np.abs(fft_result[: N // 2])

        # Plot the frequency magnitude
        plt.subplot(len(segments), 1, idx + 1)
        plt.plot(positive_frequencies, positive_fft_result)
        plt.title(f"Frequency-Domain for Segment {idx + 1} (Character {chr(65 + idx)})")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.xlim(0, base_freq * 8)  # Limit to expected frequency range

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    text = input("Enter the text to be converted to signal (default: hello): ")

    if text == "":
        text = "hello"

    duration_input = input("Enter the duration of each character in seconds (defualt: 0.1): ")
    
    if duration_input == "":
        duration_input = "0.1"

    base_frequency = input("Enter the base frequency (default: 1000): ")

    if base_frequency == "":
        base_frequency = "1000"

    bitrate = input("Enter the bitrate (default: 44100): ")

    if bitrate == "":
        bitrate = "44100"

    threshold = input("Enter the threshold (default: 0.5): ")

    if threshold == "":
        threshold = "0.5"

    filename = input("Enter the filename (default: signal.wav): ")

    if filename == "":
        filename = "signal.wav"

    text_to_sound(text, float(duration_input), int(base_frequency), int(bitrate), filename)
