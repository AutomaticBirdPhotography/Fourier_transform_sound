# Objectives:
# 1. Use FFT to find the frequencies of of the signal.
# 2. Decompose the frequencies into bytes, bits.
# 3. Convert the bitstream into ASCII characters.

import numpy as np
import soundfile as sf

# Gloabl variable if a print is needed
PRINT = True

def waw_to_text(duration_input=0.1, base_frequency = 1000, threshold = 0.5, filename="signal.wav", print=True):
    PRINT = print

    # Load signal from the .wav file
    signal, sample_rate = load_signal_from_file(filename)

    # Segment the signal into parts corresdponding to each character
    duration = (
        duration_input  # Duration of each character in seconds (must match encoding duration)
    )
    segments = segment_signal(signal, sample_rate, duration)

    # Decode the segments
    binary_message = analyze_and_decode_segments_with_fft(
        segments,
        sample_rate=sample_rate,
        base_freq=base_frequency,
        threshold=threshold,
    )
    decoded_text = binary_to_ascii(binary_message)

    if PRINT:
        print(f"Decoded message: {decoded_text}")

    return decoded_text


def load_signal_from_file(filename="signal.wav"):
    signal, sample_rate = sf.read(filename)
    return signal, sample_rate


def segment_signal(signal, sample_rate, duration=0.1):
    # Calculate the number of samples per segment
    samples_per_segment = int(duration * sample_rate)
    # Split the signal into segments, each of `samples_per_segment` length
    segments = [
        signal[i : i + samples_per_segment]
        for i in range(0, len(signal), samples_per_segment)
        if len(signal[i : i + samples_per_segment]) == samples_per_segment
    ]
    return segments


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

            # print(
            #     f"Segment {segment_index}, Bit {i+1}, Target Frequency {target_freq} Hz, "
            #     f"Detected Frequency {positive_frequencies[closest_index]:.2f} Hz, "
            #     f"Magnitude {magnitude:.2f}"
            # )

            if magnitude > threshold:
                byte += "1"
            else:
                byte += "0"
        binary_message += byte
        # print(f"Decoded byte for segment {segment_index}: {byte}")

    return binary_message


# Part 5: Convert the Bitstream into ASCII Characters


def binary_to_ascii(binary_message):
    chars = [
        chr(int(binary_message[i : i + 8], 2)) for i in range(0, len(binary_message), 8)
    ]
    return "".join(chars)


if __name__ == "__main__":
    duration_input = input("Enter the duration of each character in seconds (defualt: 0.1): ")
    
    if duration_input == "":
        duration_input = "0.1"

    base_frequency = input("Enter the base frequency (default: 1000): ")

    if base_frequency == "":
        base_frequency = "1000"

    threshold = input("Enter the threshold (default: 0.5): ")

    if threshold == "":
        threshold = "0.5"

    filename = input("Enter the filename (default: signal.wav): ")

    if filename == "":
        filename = "signal.wav"
    
    waw_to_text(float(duration_input), int(base_frequency), float(threshold), filename)
