import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

SAMPLE_INTERVAL = 0.03  # 30ms

# === 读取数据 ===
df = pd.read_csv(r"C:\Users\13512\Desktop\MIT数据\raw_data\el150800460518_cycles_long.csv")

# === 存储每循环参数 ===
cycle_stats = []

# === 绘制准备 ===
plt.figure(figsize=(12, 10))

# --- (a) 容量随循环变化 ---
for cycle, cycle_df in df.groupby("cycle_index"):
    discharge_df = cycle_df[cycle_df["current"] < 0]
    if len(discharge_df) < 5:
        continue
    Qd = discharge_df["discharge_capacity"].max() - discharge_df["discharge_capacity"].min()
    cycle_stats.append({
        "cycle": cycle,
        "Qd": Qd,
        "T": discharge_df["temperature"].mean(),
        "td": len(discharge_df) * SAMPLE_INTERVAL / 60.0,  # min
        "Ro": discharge_df["internal_resistance"].mean() if "internal_resistance" in discharge_df.columns else np.nan
    })

    # 绘制容量曲线
    plt.subplot(2, 2, 1)
    plt.plot(cycle, Qd, "o", color=plt.cm.viridis(cycle / df["cycle_index"].max()))

plt.subplot(2, 2, 1)
plt.title("(a) Discharge Capacity vs Cycle")
plt.xlabel("Cycle Times")
plt.ylabel("Discharge Capacity (Ah)")

# --- (b) 温度 / 时间 / 内阻 ---
stats_df = pd.DataFrame(cycle_stats)
plt.subplot(4, 2, 2)
plt.plot(stats_df["cycle"], stats_df["T"], color="orange")
plt.ylabel("Temperature (°C)")

plt.subplot(4, 2, 4)
plt.plot(stats_df["cycle"], stats_df["td"], color="green")
plt.ylabel("Time (min)")

if "Ro" in stats_df:
    plt.subplot(4, 2, 6)
    plt.plot(stats_df["cycle"], stats_df["Ro"], color="blue")
    plt.ylabel("Resistance (Ω)")

# --- (c) dQ/dV 曲线 ---
plt.subplot(2, 2, 3)
for cycle, cycle_df in df.groupby("cycle_index"):
    discharge_df = cycle_df[cycle_df["current"] < 0]
    if len(discharge_df) < 5:
        continue
    v = discharge_df["voltage"].values
    q = discharge_df["discharge_capacity"].values
    # 重采样
    v_new = np.linspace(v.min(), v.max(), 300)
    f_q = interp1d(v, q, fill_value="extrapolate")
    q_new = f_q(v_new)
    dq_dv = np.gradient(q_new, v_new)
    plt.plot(v_new, dq_dv, color=plt.cm.plasma(cycle / df["cycle_index"].max()), alpha=0.4)

plt.title("(c) dQ/dV")
plt.xlabel("Voltage (V)")
plt.ylabel("dQ/dV (Ah/V)")

# --- (d) dV/dt 曲线 ---
plt.subplot(2, 2, 4)
for cycle, cycle_df in df.groupby("cycle_index"):
    discharge_df = cycle_df[cycle_df["current"] < 0]
    if len(discharge_df) < 5:
        continue
    v = discharge_df["voltage"].values
    time_s = np.arange(len(discharge_df)) * SAMPLE_INTERVAL
    dv_dt = np.gradient(v, time_s / 60.0)  # V/min
    plt.plot(time_s / 60.0, dv_dt, color=plt.cm.inferno(cycle / df["cycle_index"].max()), alpha=0.4)

plt.title("(d) dV/dt")
plt.xlabel("Time (min)")
plt.ylabel("dV/dt (V/min)")

plt.tight_layout()
plt.show()
