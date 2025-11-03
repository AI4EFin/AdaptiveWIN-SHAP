import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

model = "LSTM"
jump = 1
n_0 = 72
dataset_type = "line_std"
order = "50"
# root_dir = f"examples/results/{model}/{dataset_type}_{order}/Jump_{jump}_N0_{n_0}"
root_dir = f"examples/results/{model}/empirical/Jump_{jump}_N0_{n_0}"

# Find all matching CSV files
all_files = glob.glob(os.path.join(root_dir, "run_*.csv"))

dfs = []
for file in all_files:
    # Read only the column you care about
    df = pd.read_csv(file, usecols=["windows"])

    # Rename the column to something unique (e.g., filename without extension)
    name = os.path.splitext(os.path.basename(file))[0]
    df = df.rename(columns={"windows": f"windows_{name}"})

    dfs.append(df)

# Concatenate all on columns (axis=1)
df = pd.concat(dfs, axis=1)
df["windows_mean"] = df.mean(axis=1)

data_df = pd.read_csv(f"examples/datasets/simulated/{dataset_type}/{order}.csv")
plt.plot(data_df["Date"], data_df["N"], color='#3B75AF')
# plt.plot(jump*np.arange(len(df['windows_mean']))[20:], rolling[20:], color='red')

plt.xlabel('Time Step')
plt.ylabel('Value')
# plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig(f"{root_dir}/dataset.png", format='png', dpi=600, transparent=True)
plt.show()

rolling = df["windows_mean"].rolling(window=20).mean()
plt.plot(jump*np.arange(len(df['windows_mean']))[50:], df['windows_mean'][50:], color='#3B75AF', label="Window Size")
plt.plot(jump*np.arange(len(df['windows_mean']))[50:], rolling[50:], color='red', label="Window Size Moving Average (10)")

plt.xlabel('Time Step')
plt.ylabel('Window Size')
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(f"{root_dir}/window_size.png", format='png', dpi=600, transparent=True)
plt.show()