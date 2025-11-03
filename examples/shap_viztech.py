import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = "LSTM"
jump = 1
n_0 = 72

root_dir = f"examples/results/{model}/empirical/Jump_{jump}_N0_{n_0}"

df = pd.read_csv(f"{root_dir}/results.csv")
print(df.head())

rolling_t1 = df["shap_lstm_t-1"].rolling(window=20).mean()
rolling_t2 = df["shap_lstm_t-2"].rolling(window=20).mean()

plt.plot(jump*np.arange(len(df['shap_lstm_t-10']))[50:], df['shap_lstm_t-10'][50:], color='#3B75AF', label="$|\phi_{t-1}$|")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-2']))[50:], df['shap_lstm_t-2'][50:], color='red', label="$|\phi_{t-2}$|")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-4']))[50:], df['shap_lstm_t-2'][50:], color='green', label="$|\phi_{t-4}$|")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-5']))[50:], df['shap_lstm_t-2'][50:], color='yellow', label="$|\phi_{t-5}$|")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-6']))[50:], df['shap_lstm_t-2'][50:], color='brown', label="$|\phi_{t-6}$|")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-7']))[50:], df['shap_lstm_t-2'][50:], color='red', label="$|\phi_{t-2}$|")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-8']))[50:], df['shap_lstm_t-2'][50:], color='red', label="$|\phi_{t-2}$|")

# plt.plot(jump*np.arange(len(df['shap_lstm_t-1']))[50:], rolling_t1[50:], color='blue', label="Window Size Moving Average (10)")
# plt.plot(jump*np.arange(len(df['shap_lstm_t-2']))[50:], rolling_t2[50:], color='red', label="Window Size Moving Average (10)")
# plt.ylim(0, 1)

plt.xlabel('Ti/me Step')
plt.ylabel('Window Size')
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(f"{root_dir}/window_size.png", format='png', dpi=600, transparent=True)
plt.show()
