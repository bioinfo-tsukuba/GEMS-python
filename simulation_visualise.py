import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt




# Define the directory name
directory = Path('/Users/yuyaarai/Documents/Humanics/Project/GEMS-python/volatile_IPS_ten_2024-09-06_00_52_60a9ee7d-fee2-4244-b07e-f001f1006a99')

left = 0
right = 2
while right - left > 1:
    if (directory/f"step_{right}").exists():
        left = right
        right *= 2
    else:
        mid = (left + right) // 2
        if (directory/f"step_{mid}").exists():
            left = mid
        else:
            right = mid

maximum_step = left

directory = directory/f"step_{maximum_step}"/"experiments"
print(f"{directory=}")

# List all files in the experiments directory
csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize a figure for the plot
plt.figure(figsize=(10, 6))

# Loop through each CSV file and plot the data
for csv_file in csv_files:
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Plot time vs real_density if the 'real_density' column exists
    if 'real_density' in df.columns:
        plt.plot(df['time'], df['real_density'], label=os.path.basename(csv_file).split('_')[1])

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Real Density')
plt.title('Time vs Real Density for Various Experiments')

# Show the legend to differentiate between experiments
plt.legend(title="Experiment ID")

# Display the plot
plt.show()
