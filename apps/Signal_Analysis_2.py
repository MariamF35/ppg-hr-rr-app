# signal_analysis_complete.py
import csv
import math
import matplotlib.pyplot as plt

# === Moving average filter ===
def moving_average(signal, window=5):
    filtered = []
    for i in range(len(signal)):
        start = max(0, i - window // 2)
        end = min(len(signal), i + window // 2 + 1)
        avg = sum(signal[start:end]) / (end - start)
        filtered.append(avg)
    return filtered

# === Peak detection ===
def detect_peaks(signal, threshold):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    return peaks

# === HRV (time domain) ===
def hrv_measures(peaks, time):
    rr_intervals = []
    for i in range(1, len(peaks)):
        rr = (time[peaks[i]] - time[peaks[i - 1]]) * 1000  # ms
        rr_intervals.append(rr)

    if not rr_intervals:
        return None

    mean_rr = sum(rr_intervals) / len(rr_intervals)
    sdnn = math.sqrt(sum((x - mean_rr) ** 2 for x in rr_intervals) / len(rr_intervals))
    rmssd = math.sqrt(sum((rr_intervals[i] - rr_intervals[i - 1]) ** 2 for i in range(1, len(rr_intervals))) / len(rr_intervals))
    pnn50 = sum(1 for i in range(1, len(rr_intervals)) if abs(rr_intervals[i] - rr_intervals[i - 1]) > 50) / len(rr_intervals) * 100

    return mean_rr, sdnn, rmssd, pnn50

# === FFT helper (no numpy) ===
def compute_fft(signal, fs):
    N = len(signal)
    re, im = [], []
    for k in range(N):
        real = 0
        imag = 0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            real += signal[n] * math.cos(angle)
            imag -= signal[n] * math.sin(angle)
        re.append(real)
        im.append(imag)
    mag = [math.sqrt(re[i]**2 + im[i]**2) for i in range(N)]
    freqs = [fs * k / N for k in range(N)]
    return freqs, mag

# === Respiration from PPG oscillations (envelope tracking) ===
def estimate_resp_rate(ppg_signal, time, fs):
    window = int(fs * 2)  # 2-second moving average for baseline
    baseline = moving_average(ppg_signal, window)
    resp_component = [ppg_signal[i] - baseline[i] for i in range(len(ppg_signal))]

    # Peak detection on respiration component
    threshold = sum(resp_component) / len(resp_component)
    resp_peaks = detect_peaks(resp_component, threshold)
    if len(resp_peaks) < 2:
        return None

    intervals = [(time[resp_peaks[i]] - time[resp_peaks[i - 1]]) for i in range(1, len(resp_peaks))]
    mean_interval = sum(intervals) / len(intervals)
    resp_rate = 60 / mean_interval  # breaths per min
    return resp_rate

# === Load CSVs ===
base = input("Enter base file name (Example: bidmc_01): ")
#c=0 ################## count
# Signals
time_sig, resp_sig, pleth_sig = [], [], []
with open(base + "_Signals.csv", "r") as f:
    #c=c+1
    reader = csv.DictReader(f)
    for row in reader:
       """ if c>6: #for sample
            break #for sample
            """
        time_sig.append(float(row["Time [s]"]))
        resp_sig.append(float(row[" RESP"]))
        pleth_sig.append(float(row[" PLETH"]))
        c+=1

f4=open(base+"_Fix.txt","r") #to fetch frequency
trial=f4.readlines()
f4.close()
fs = 1 / (time_sig[1] - time_sig[0])  # sampling frequency
print("frequency = ",trial[2].strip()[28::])
# Preprocess
pleth_clean = moving_average(pleth_sig, window=5)

# Peak detection for PPG
threshold = sum(pleth_clean) / len(pleth_clean)
peaks = detect_peaks(pleth_clean, threshold)

# === HRV Time-domain ===
metrics = hrv_measures(peaks, time_sig)
if metrics:
    mean_rr, sdnn, rmssd, pnn50 = metrics
    hr = 60000 / mean_rr
    print("\n=== HRV (Time-Domain) ===")
    print(f"Mean RR: {mean_rr:.2f} ms")
    print(f"SDNN: {sdnn:.2f} ms")
    print(f"RMSSD: {rmssd:.2f} ms")
    print(f"pNN50: {pnn50:.2f} %")
    print(f"Estimated HR: {hr:.2f} bpm")

# === Frequency-domain HRV ===
freqs, mag = compute_fft(pleth_clean, fs)

# Integrate power in LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz)
lf_power = sum(mag[i] for i in range(len(freqs)) if 0.04 <= freqs[i] <= 0.15)
hf_power = sum(mag[i] for i in range(len(freqs)) if 0.15 <= freqs[i] <= 0.40)
lf_hf_ratio = lf_power / hf_power if hf_power > 0 else None

print("\n=== Frequency-Domain HRV ===")
print(f"LF Power: {lf_power:.2f}")
print(f"HF Power: {hf_power:.2f}")
print(f"LF/HF Ratio: {lf_hf_ratio}")

# === Respiration Rate ===
resp_rate = estimate_resp_rate(pleth_clean, time_sig, fs)
if resp_rate:
    print(f"\nEstimated Respiration Rate: {resp_rate:.2f} breaths/min")

# === Plots ===
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time_sig, pleth_sig, label="Raw PPG", alpha=0.5)
plt.plot(time_sig, pleth_clean, label="Clean PPG")
plt.scatter([time_sig[i] for i in peaks], [pleth_clean[i] for i in peaks], color="red", label="Peaks")
plt.legend(); plt.title("PPG with Peaks")

plt.subplot(3, 1, 2)
plt.plot(freqs[:200], mag[:200])
plt.title("Frequency Spectrum (PPG)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.subplot(3, 1, 3)
plt.plot(time_sig, resp_sig, label="Respiration Signal")
plt.legend(); plt.title("Respiration")

plt.tight_layout()
plt.show()
