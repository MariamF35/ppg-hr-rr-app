# signal_analysis_userfriendly.py
import csv
import math
import matplotlib.pyplot as plt
import os

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

# === Respiration from PPG oscillations ===
def estimate_resp_rate(ppg_signal, time, fs):
    window = int(fs * 2)  # 2-second moving average for baseline
    baseline = moving_average(ppg_signal, window)
    resp_component = [ppg_signal[i] - baseline[i] for i in range(len(ppg_signal))]

    threshold = sum(resp_component) / len(resp_component)
    resp_peaks = detect_peaks(resp_component, threshold)
    if len(resp_peaks) < 2:
        return None

    intervals = [(time[resp_peaks[i]] - time[resp_peaks[i - 1]]) for i in range(1, len(resp_peaks))]
    mean_interval = sum(intervals) / len(intervals)
    resp_rate = 60 / mean_interval  # breaths per min
    return resp_rate

# === MAIN PROGRAM ===
def analyze_file(base, max_readings=6000, save_results=True):
    # Load Signals
    time_sig, resp_sig, pleth_sig = [], [], []
    with open(base + "_Signals.csv", "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_readings:
                break
            time_sig.append(float(row["Time [s]"]))
            resp_sig.append(float(row[" RESP"]))
            pleth_sig.append(float(row[" PLETH"]))

    fs = 1 / (time_sig[1] - time_sig[0])  # sampling frequency

    # Preprocess
    pleth_clean = moving_average(pleth_sig, window=5)

    # Peak detection
    threshold = sum(pleth_clean) / len(pleth_clean)
    peaks = detect_peaks(pleth_clean, threshold)

    # HRV time-domain
    results = {"File": base}
    metrics = hrv_measures(peaks, time_sig)
    if metrics:
        mean_rr, sdnn, rmssd, pnn50 = metrics
        hr = 60000 / mean_rr
        results.update({
            "Mean_RR_ms": round(mean_rr, 2),
            "SDNN_ms": round(sdnn, 2),
            "RMSSD_ms": round(rmssd, 2),
            "pNN50_%": round(pnn50, 2),
            "HR_bpm": round(hr, 2)
        })

    # Frequency-domain HRV
    freqs, mag = compute_fft(pleth_clean, fs)
    lf_power = sum(mag[i] for i in range(len(freqs)) if 0.04 <= freqs[i] <= 0.15)
    hf_power = sum(mag[i] for i in range(len(freqs)) if 0.15 <= freqs[i] <= 0.40)
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else None
    results.update({
        "LF_power": round(lf_power, 2),
        "HF_power": round(hf_power, 2),
        "LF/HF": round(lf_hf_ratio, 2) if lf_hf_ratio else None
    })

    # Respiration rate
    resp_rate = estimate_resp_rate(pleth_clean, time_sig, fs)
    if resp_rate:
        results["RespRate_bpm"] = round(resp_rate, 2)

    # === Save results to CSV ===
    if save_results:
        out_file = "analysis_results.csv"
        file_exists = os.path.isfile(out_file)
        with open(out_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

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

    return results


# === Run for one or multiple files ===
if __name__ == "__main__":
    print("Enter base file names separated by commas (e.g. bidmc_01,bidmc_02):")
    bases = input().split(",")
    max_readings = int(input("How many readings to load? (e.g. 6000): "))

    for base in bases:
        base = base.strip()
        if base:
            print(f"\nAnalyzing {base} ...")
            res = analyze_file(base, max_readings=max_readings)
            print("Results:", res)

    print("\nAll results saved to analysis_results.csv")
