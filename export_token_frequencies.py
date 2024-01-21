import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams['text.usetex'] = True # TeX rendering

# Step 1: Load the data
with open('results/token-frequencies.json', 'r') as file:
    data = json.load(file)

# Prepare a list for each tokenizer's data
all_data = []

# Determine the number of buckets
num_buckets = 30

# Step 2: Process each tokenizer and calculate the average frequency per bucket
for tokenizer_name, frequencies in data.items():
    # Convert the frequencies to a DataFrame
    df = pd.DataFrame(list(frequencies.values()), index=frequencies.keys(), columns=['Frequency'])
    # Use qcut to create quantile-based buckets
    df['Bucket'], bins = pd.qcut(df['Frequency'], q=num_buckets, labels=range(num_buckets), retbins=True, duplicates='drop')
    # Calculate the mean frequency for each bucket
    bucket_means = df.groupby('Bucket')['Frequency'].mean()
    # Store the data
    all_data.append(bucket_means)

# Prepare a DataFrame for plotting
plot_data = {'Bucket': range(num_buckets)}

# Fill in the data for each tokenizer
for tokenizer_name, bucket_means in zip(data.keys(), all_data):
    plot_data[tokenizer_name] = bucket_means.reindex(range(num_buckets), fill_value=0).values

# Create a DataFrame for plotting
plot_df = pd.DataFrame(plot_data)

# Sort the DataFrame in descending order to reverse the curve
plot_df = plot_df.sort_values(by='Bucket', ascending=False)

# Melt the DataFrame to have long-form data
plot_df_long = pd.melt(plot_df, id_vars=['Bucket'], var_name='Tokenizer', value_name='Frequency')

# Set the aesthetic style of the plots
sns.set(style="whitegrid", palette="pastel", font_scale=1.2, rc={"font.family": "Times New Roman"})

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(12, 6))

# Define the width of the bars so that they all fit in a single bucket group
num_tokenizers = len(data.keys())
bar_width = 1 / num_tokenizers

# Plot bars for each tokenizer
for i, tokenizer in enumerate(data.keys()):
    # Filter the DataFrame for the current tokenizer
    df_tokenizer = plot_df_long[plot_df_long['Tokenizer'] == tokenizer]
    # Plot the bars
    ax.bar(df_tokenizer['Bucket'] + i * bar_width, df_tokenizer['Frequency'], width=bar_width, label=tokenizer)

# Set the x-axis labels, y-axis to log scale, labels, and title
ax.set_xticks(np.arange(num_buckets) + bar_width * (len(data.keys()) - 1) / 2)
ax.set_xticklabels([])
ax.set_yscale('log')
ax.set_xlabel('Frequency Buckets')
ax.set_ylabel('Average Frequency (Log Scale)')
ax.set_title('Token Frequency Distribution by Tokenizer (Log Scale)')

# Add the legend
ax.legend(title='Tokenizer')

# Show the plot
plt.tight_layout()
plt.grid(linestyle='dotted')
plt.savefig('results/token-frequencies.pdf', dpi=300)
