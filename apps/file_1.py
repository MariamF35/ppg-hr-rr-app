# signal_analysis_basic.py
import csv
import matplotlib.pyplot as plt
import math

# === Helper: moving average filter ===
def moving_average(signal, window=5):
    filtered = []
    for i in range(len(signal)):
        start = max(0, i - window // 2)
        end = min(len(signal), i + window // 2 + 1)
        avg = sum(signal[start:end]) / (end - start)
        filtered.append(avg)
    return filtered

# === Helper: compute stats ===
def compute_stats(signal, name):
    mean_val = sum(signal) / len(signal)
    min_val = min(signal)
    max_val = max(signal)
    std_val = math.sqrt(sum((x - mean_val) ** 2 for x in signal) / len(signal))
    print(f"\n{name} Statistics:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  Std Dev: {std_val:.4f}")

# === Load CSV (Signals) ===
time, resp, pleth = [], [], []
with open("bidmc_01_Signals.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        time.append(float(row["Time [s]"]))
        resp.append(float(row["RESP"]))
        pleth.append(float(row["PLETH"]))

# === Clean Signals ===
resp_clean = moving_average(resp, window=5)
pleth_clean = moving_average(pleth, window=5)

# === Stats ===
compute_stats(resp_clean, "Respiration")
compute_stats(pleth_clean, "PPG")

# === Plot ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, resp, label="Raw Resp", alpha=0.5)
plt.plot(time, resp_clean, label="Clean Resp", linewidth=2)
plt.title("Respiration Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, pleth, label="Raw PPG", alpha=0.5)
plt.plot(time, pleth_clean, label="Clean PPG", linewidth=2)
plt.title("PPG Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

