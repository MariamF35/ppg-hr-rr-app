# import statements
import csv
import matplotlib.pyplot as plt
import math

# === moving average filter ===
def moving_average(signal, window=5):
    filtered = []
    for i in range(len(signal)):
        start = max(0, i - window // 2)
        end = min(len(signal), i + window // 2 + 1)
        avg = sum(signal[start:end]) / (end - start)
        filtered.append(avg)
    return filtered

# === compute stats ===
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
c=0 ###count for sample data
time, resp, pleth = [], [], []
#print("hello! program started!")
#print("time, resp,pleth = ",time,resp,pleth,end="\n")
with open("bidmc_01_Signals.csv", "r") as f:
    print("Opened Signals file")
    c+=1 #count for sample data 
    reader = csv.DictReader(f)
    for row in reader:
        if c>20:  #for sample data (change value...20 represents no. of rows considered for sample data
            break  #for sample data
        #print("   c=",c,"row,len(row) = ",row,len(row))
        time.append(row["Time [s]"])
        resp.append(float(row[" RESP"]))
        pleth.append(float(row[" PLETH"]))
        c+=1 #for sample data
        
#print("len(time)= ",len(time),"\nlen(resp)= ",len(resp),"len(pleth)= ",len(pleth),end="\n\n")
"""for j in resp:
    print("j in resp = ",j,"\t",type(j))
print()"""
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
