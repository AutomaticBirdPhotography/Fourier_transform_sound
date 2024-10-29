# Objectives:
# 1. Convert ASCII characters to binary values represnted in bytes.
# 2. Represent each byte as a combination of 8 frequencies. Then make a signal out of these multi-friquency byte representations.
# 3. Use STFT to find the frequencies of of the signal.
# 4. Decompose the frequencies into bytes, bits.
# 5. Convert the bitstream into ASCII characters.


import numpy as np  # Part 2
import matplotlib.pyplot as plt  # Part 6 & 7


def main():
    # Original message
    text = "Hello, my name is Benjamin! I'm soon 20 years old."
    sample_rate = 44100
    base_frequency = 100
    duration = 1  # The duration threshold value is 0.01

    segments = generate_signal_segments(
        text,
        sample_rate=sample_rate,
        base_freq=base_frequency,
        duration=duration,
    )

    # Analyze and decode the message with FFT
    binary_message = analyze_and_decode_segments_with_fft(
        segments, sample_rate=sample_rate, base_freq=base_frequency, threshold=0.5
    )

    decoded_text = binary_to_ascii(binary_message)

    print(f"Decoded message: {decoded_text}")

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
    print("Binary representation:", binary_representation)
    print("Decimal representation:", int(binary_representation, 2))
    segments = [
        byte_to_frequencies(
            binary_representation[i : i + 8], base_freq, sample_rate, duration
        )
        for i in range(0, len(binary_representation), 8)
    ]
    # Normalize to the range -1 to 1
    segments /= np.max(np.abs(segments) + 1e-6)
    return segments  # Return list of segments for each character


# Part 3 & 4:
#   Use FFT to Find the Frequencies of the Signal
#   Decompose the Frequencies into Bytes and Bits


def analyze_and_decode_segments_with_fft(
    segments, sample_rate=44100, base_freq=100, threshold=0.5
):
    binary_message = ""

    for segment_index, segment in enumerate(segments):
        # Perform FFT on the entire signal
        N = len(segment)
        fft_result = np.fft.fft(segment)
        frequencies = np.fft.fftfreq(N, d=1 / sample_rate)

        # Keep only the positive frequencies
        positive_frequencies = frequencies[: N // 2]
        positive_fft_results = np.abs(
            fft_result[: N // 2]
        )  # Take magnitude of FFT result

        # Decode each frequency to determine the bits for this character
        byte = ""
        for i in range(8):
            target_freq = base_freq * (i + 1)
            closest_index = np.argmin(np.abs(positive_frequencies - target_freq))
            magnitude = positive_fft_results[closest_index]

            if magnitude > threshold:
                byte += "1"
            else:
                byte += "0"

            # print(
            #     f"Segment {segment_index}, Bit {i+1}, Frequency {target_freq} Hz, Magnitude: {magnitude}, Decoded bit: {byte[-1]}"
            # )
        binary_message += byte
        print(f"Decoded byte for segment {segment_index}: {byte}")

    return binary_message


# Part 5: Convert the Bitstream into ASCII Characters


def binary_to_ascii(binary_message):
    chars = [
        chr(int(binary_message[i : i + 8], 2)) for i in range(0, len(binary_message), 8)
    ]
    return "".join(chars)


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
    main()
