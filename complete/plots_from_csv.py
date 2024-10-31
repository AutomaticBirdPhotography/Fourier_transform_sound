import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your CSV file
csv_file = '311024_duration_frequency_results.csv'  # Replace with your actual filename

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Clean up column names by stripping any extra whitespace
df.columns = df.columns.str.strip()

# Check if the CSV has the expected columns
if 'Duration' in df.columns and 'Base Frequency' in df.columns:
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Duration'], df['Base Frequency'], marker='o', linestyle='-', color='b')
    plt.xlabel('Duration (s)')
    plt.ylabel('Base Frequency (Hz)')
    plt.title('Duration vs. Base Frequency')
    plt.grid(True)
    plt.show()
else:
    print("CSV file does not contain the expected 'Duration' and 'Base Frequency' columns.")
