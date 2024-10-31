import text_to_sound
import sound_to_text
import numpy as np

# Goal: Determine the relationship between the duration of each character and the frequency of the base tone
# We will use the text "Hello, my name is BOB. I like fish!! :)" as an example
# We will vary the duration from 1 seconds to 1ms and the base frequency from 10 to 5000 Hz
# We will try eash duration with each base frequency to determine the lowest base frequency that can be used for each duration
# If a duration and base frequency works, this is testet by converting the text to sound and back to text and the text is the same as the original text
# The results will be stored in a dictionary where the key is the duration and the value is the lowest base frequency that can be used

text = "Hello, my name is BOB. I like fish!! :)"

# Duration range from 1 second to 1 ms, wiriten in seconds in steps of 10 ms
durations = []
durations.extend(np.round(np.arange(1, 0.05, -0.01), 2).tolist())
durations.extend(np.round(np.arange(0.05, 0.0001, -0.001), 3).tolist())

print("This durations wil be tested: ", durations)

# Base frequency range from 1 Hz to 5000 Hz in steps of 1 Hz
base_frequencie_steps = 1
base_frequencies = np.round(np.arange(1, 5000, base_frequencie_steps), 1).tolist()

# Store the results in a dictionary, where the key is the duration and the value is the lowest base frequency that can be used
results = {}

# Last ran base frequency that worked to use as offset for the next duration
base_freq_offset_index = 0

for duration in durations:
    for base_freq in base_frequencies:
        
        # Convert text to sound
        text_to_sound.text_to_sound(text, duration, base_freq, 44100, "test.wav")

        # Convert sound to text
        decoded_text = sound_to_text.waw_to_text(duration, base_freq, 0.5, "test.wav")

        if decoded_text == text:
            # Store the last ran frequency that worked
            print(f"\nWorked: Duration: {duration} s, Base Frequency: {base_freq} Hz")
            base_freq_offset_index = base_frequencies.index(base_freq)
            results[duration] = base_freq
            break
        else:
            print(f"\rDuration: {duration} s, Base Frequency: {base_freq} Hz did not work", end="")

def test_duration(durations, base_freq, text, print_text=False):
    global results

    for duration in durations:
        # Convert text to sound
        text_to_sound.text_to_sound(text, duration, base_freq, 44100, "test.wav", print=False)

        # Convert sound to text
        decoded_text = sound_to_text.waw_to_text(duration, base_freq, 0.5, "test.wav", print=False)

        if decoded_text == text:
            # Store the last ran frequency that worked
            print(f"\nWorked: Duration: {duration} s, Base Frequency: {base_freq} Hz")
            base_freq_offset_index = base_frequencies.index(base_freq)
            results[duration] = base_freq
            break
        else:
            print(f"\rDuration: {duration} s, Base Frequency: {base_freq} Hz did not work", end="")

    return
        
        
            
# Save the results to a file
with open("duration_frequency_results.txt", "w") as f:
    f.write("Duration, Base Frequency\n")
    for duration, base_freq in results.items():
        f.write(f"{duration} s, {base_freq} Hz\n")