import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the JSON file
with open('results/token-frequencies.json', 'r') as file:
    data = json.load(file)

# Process each tokenizer
for tokenizer, frequencies in data.items():
    # Convert to DataFrame
    df = pd.DataFrame(list(frequencies.items()), columns=['Token ID', 'Frequency'])

    # Create ~30 equal-sized buckets based on frequency
    df['Bucket'] = pd.qcut(df['Frequency'], q=100, duplicates='drop')

    # Calculate average frequency per bucket
    bucket_avg = df.groupby('Bucket')['Frequency'].mean().sort_index(ascending=False)

    # Plotting
    sns.barplot(x=bucket_avg.index, y=bucket_avg.values)
    plt.xticks([])  # Disable x-axis labels
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Frequency Buckets')
    plt.ylabel('Average Frequency (Log Scale)')
    plt.title(f'Token Frequency Distribution for {tokenizer} (Log Scale)')
    plt.show()
