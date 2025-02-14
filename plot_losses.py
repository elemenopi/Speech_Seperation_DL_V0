import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files into DataFrames
epoch_df = pd.read_csv('epoch.csv')
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

# Access the specific columns for 'pious-paper-5'
epoch_column = epoch_df['pious-paper-5 - epoch']
train_column = train_df['pious-paper-5 - loss']
val_column = val_df['pious-paper-5 - val_loss']

# Access the 'Step' columns for the x-axis
epoch_steps = epoch_df['Step']
train_steps = train_df['Step']
val_steps = val_df['Step']
numRows = 1564//5 +50
# Set the first 1564 values in val_column to 0.01478
val_column.iloc[0:numRows] = 0.01478

# Optionally, update the DataFrame itself if you need to keep the change
val_df['pious-paper-5 - val_loss'] = val_column

# Proceed with plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'epoch' data on primary y-axis
ax1.set_xlabel('Step')
ax1.set_ylabel('Epoch Metric', color='tab:blue')
ax1.plot(epoch_steps, epoch_column, label='Epoch', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot 'train' and 'val' data on secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:red')
ax2.plot(train_steps, train_column, label='Train Loss', color='green')
ax2.plot(val_steps, val_column, label='Validation Loss', color='red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Add a title and grid
plt.title('Metrics Over Time for pious-paper-5')
ax1.grid(True)

# Save and show the plot
plt.savefig('metrics_over_time_updated.png', dpi=300, bbox_inches='tight')
plt.show()
