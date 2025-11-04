import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

def parse_df_timeindex(df):
    df = df.sort_index()
    # if not isinstance(price_df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True).map(lambda x: x.tz_convert('Europe/Bucharest'))
    return df

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


start = pd.Timestamp("2021-05-01 00:00:00", tz="Europe/Bucharest")
end = pd.Timestamp("2021-08-01 00:00:00", tz="Europe/Bucharest")

# data_df = pd.read_csv(f"examples/datasets/simulated/{dataset_type}/{order}.csv")
data_df = pd.read_csv(f"examples/datasets/empirical/dataset.csv", parse_dates=True, index_col=0)
data_df = parse_df_timeindex(data_df)
data_df = data_df.loc[start:end]
# data_df.index = pd.to_datetime(df.index)

# formatted_labels = data_df.index.strftime('%-d.%-m')
plt.figure(figsize=(10,6))
plt.plot(data_df.index, data_df["Price Day Ahead"], color='#3B75AF')
# plt.plot(jump*np.arange(len(df['windows_mean']))[20:], rolling[20:], color='red')

plt.xlabel('Time Step')
plt.ylabel('Price Day Ahead')
# plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig(f"{root_dir}/dataset.png", format='png', dpi=600, transparent=True)
plt.show()

rolling = df["windows_mean"].rolling(window=20).mean()
plt.figure(figsize=(10,6))
plt.plot(data_df.index[50:], df['windows_mean'][50:], color='#3B75AF', label="Window Size")
plt.plot(data_df.index[50:], rolling[50:], color='red', label="Window Size Moving Average (20)")

plt.xlabel('Time Step')
plt.ylabel('Window Size')
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(f"{root_dir}/window_size.png", format='png', dpi=600, transparent=True)
plt.show()