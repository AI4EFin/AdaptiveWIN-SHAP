from pandas.plotting import andrews_curves
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

model = "LSTM"
jump = 1
n_0 = 72
LSTM_SEQ_LEN = 24
root_dir = f"examples/results/{model}/empirical/Jump_{jump}_N0_{n_0}"

start = pd.Timestamp("2021-05-01 00:00:00", tz="Europe/Bucharest")
end = pd.Timestamp("2021-08-01 00:00:00", tz="Europe/Bucharest")
data_df = pd.read_csv(f"examples/datasets/empirical/dataset.csv", parse_dates=True, index_col=0)
def parse_df_timeindex(df):
    df = df.sort_index()
    # if not isinstance(price_df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True).map(lambda x: x.tz_convert('Europe/Bucharest'))
    return df
data_df = parse_df_timeindex(data_df)
data_df = data_df.loc[start:end][LSTM_SEQ_LEN:]

df = pd.read_csv(f"{root_dir}/results.csv")
df["Date"] = data_df.index
df["hour"] = df["Date"].dt.hour.astype("category")

# Grab all SHAP columns
shap_cols = [c for c in df.columns if c.startswith("shap_")]
new_labels = [f"$\phi_{{t-{i+1}}}$" for i in range(LSTM_SEQ_LEN)]

# OPTIONAL: if there are many shap_* columns, pick the top-N by variability to reduce clutter
# Comment this block out if you want to plot all shap_* lines.
# N = 10  # change as needed
# if len(shap_cols) > N:
#     shap_cols = (
#         df[shap_cols]
#         .std()
#         .sort_values(ascending=False)
#         .head(N)
#         .index
#         .tolist()
#     )

# Plot: one line per shap_* column over time
fig, ax = plt.subplots(figsize=(15, 9))
for col, label in zip(shap_cols, new_labels):
    plt.plot(df["Date"], df[col], label=label, alpha=0.9)

plt.ylim(0, 0.25)
plt.title("SHAP contributions over time")
plt.xlabel("Time")
plt.ylabel("SHAP value")
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=12, frameon=False)
plt.tight_layout()
plt.savefig(f"{root_dir}/shap_viztech.png", format='png', dpi=600, transparent=True)
plt.show()

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df[shap_cols])
df_scaled = pd.DataFrame(X, columns=shap_cols, index=df.index)
df_scaled["hour"] = df["hour"]

# Downsample (optional, helps readability)
df_plot = (
    df_scaled.groupby("hour", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), 100), random_state=42))
)

# Use a bright, high-contrast colormap
palette = sns.color_palette("tab10", n_colors=len(df_plot["hour"].unique()))

plt.figure(figsize=(12, 6))
andrews_curves(
    df_plot,
    "hour",
    color=palette,
    alpha=0.7,
    samples=200
)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, new_labels, title="Hour", bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=10, frameon=False)
plt.grid(False)
plt.title("Andrews Curves of SHAP features by Hour of Day", fontsize=14)
plt.xlabel("t (synthetic feature coordinate)")
plt.ylabel("f(t) (composite of SHAP values)")
plt.tight_layout()
plt.savefig(f"{root_dir}/shap_viztech_ac.png", format='png', dpi=600, transparent=True)
plt.show()


# (Optional) keep top-N most variable features to avoid spaghetti
# N = 12
# shap_cols = (
#     df[shap_cols].std().sort_values(ascending=False).head(N).index.tolist()
# )

# --- Weekly aggregation (grid per week) ---
# Take weekly averages (Mondays). You can pick .mean(), .median(), .last(), etc.
weekly = (
    df.set_index("Date")[shap_cols]
      .resample("W-MON")
      .mean()
)

# If there are many weeks, sample evenly to keep the plot readable
max_weeks = 10  # adjust as you like
if len(weekly) > max_weeks:
    idx = np.linspace(0, len(weekly) - 1, max_weeks).round().astype(int)
    weekly_sel = weekly.iloc[idx]
else:
    weekly_sel = weekly

# Nicely formatted axis labels = week start dates
axis_labels = [d.strftime("%Y-%m-%d") for d in weekly_sel.index]

# --- Build the parallel-coords DataFrame ---
# Rows = feature (class), Columns = selected weekly dates
plot_df = weekly_sel.T.copy()            # shape: (features x weeks)
plot_df.columns = axis_labels
plot_df["feature"] = new_labels       # class column = feature name
plot_df = plot_df.reset_index(drop=True) # tidy

# (Optional) scaling for readability (comment out if you want raw SHAP values)
scaler = StandardScaler()
plot_df[axis_labels] = scaler.fit_transform(plot_df[axis_labels])

# --- Plot ---
fig, ax = plt.subplots(figsize=(15, 9))

# Distinct, readable palette for many classes
colors = plt.cm.tab20.colors if plot_df.shape[0] <= 20 else plt.cm.gist_ncar(
    np.linspace(0, 1, plot_df.shape[0])
)

parallel_coordinates(
    plot_df,
    class_column="feature",
    cols=axis_labels,
    color=colors,
    alpha=0.85
)
plt.grid(False)
ax.set_title("Parallel Coordinates of SHAP features across weekly dates", fontsize=14)
ax.set_xlabel("Week (Date)")
ax.set_ylabel("SHAP value")

# If legend gets huge, you can move it outside or limit features
leg = ax.legend(title="Feature", fontsize=9, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=8, frameon=False)
fig.tight_layout()
plt.savefig(f"{root_dir}/shap_viztech_pcp.png", format='png', dpi=600, transparent=True)
plt.show()

