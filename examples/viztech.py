import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = "LSTM"
jump = 1
n_0 = 100
dataset_type = "line_std"
order = "10"
root_dir = f"examples/results/{model}/{dataset_type}_{order}/Jump_{jump}_N0_{n_0}"

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

# rolling = df["windows_mean"].rolling(window=3).mean()
plt.plot(jump*np.arange(len(df['windows_mean']))[20:], df['windows_mean'][20:], color='blue')
# plt.plot(jump*np.arange(len(df['windows_mean']))[20:], rolling[20:], color='red')

plt.xlabel('Time Step')
plt.ylabel('Window Size')
# plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig(f"{root_dir}/window_size.png", format='png', dpi=600, transparent=True)
plt.show()