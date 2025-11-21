import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

SAMPLE_INTERVAL = 0.03  # 每点30ms


def preprocess_cycle(cycle_df, num_points=200):
    """处理单个循环并返回曲线数据"""
    discharge_df = cycle_df[cycle_df["current"] < 0].copy()
    if len(discharge_df) < 5:
        return None

    # 构造时间轴 (秒 → 分钟)
    n_points = len(discharge_df)
    time_s = np.arange(n_points) * SAMPLE_INTERVAL
    time_min = time_s / 60.0

    # 平滑
    discharge_df["voltage_smooth"] = savgol_filter(discharge_df["voltage"], 21, 3)
    discharge_df["capacity_smooth"] = savgol_filter(discharge_df["discharge_capacity"], 21, 3)
    discharge_df["temperature_smooth"] = savgol_filter(discharge_df["temperature"], 21, 3)

    v = discharge_df["voltage_smooth"].values
    q = discharge_df["capacity_smooth"].values
    t = discharge_df["temperature_smooth"].values

    # 重构 dQ/dV & dT/dV
    v_new = np.linspace(v.min(), v.max(), num_points)
    f_q = interp1d(v, q, kind="linear", fill_value="extrapolate")
    f_t = interp1d(v, t, kind="linear", fill_value="extrapolate")
    q_new = f_q(v_new)
    t_new = f_t(v_new)

    dq_dv = np.gradient(q_new, v_new)
    dt_dv = np.gradient(t_new, v_new)

    # dV/dt (用秒计算，再画图时横坐标转分钟)
    dv_dt = np.gradient(v, time_s)

    return v_new, dq_dv, dt_dv, time_min, dv_dt


def plot_multiple_cycles(df, cycle_list, num_points=200):
    """画多个循环的 dQ/dV, dT/dV, dV/dt 对比"""
    plt.figure(figsize=(15, 4))

    # dQ/dV
    plt.subplot(1, 3, 1)
    for cycle_id in cycle_list:
        cycle_df = df[df["cycle_index"] == cycle_id]
        curves = preprocess_cycle(cycle_df, num_points)
        if curves:
            v_new, dq_dv, _, _, _ = curves
            plt.plot(v_new, dq_dv, label=f"Cycle {cycle_id}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("dQ/dV")
    plt.title("dQ/dV Comparison")
    plt.legend()

    # dT/dV
    plt.subplot(1, 3, 2)
    for cycle_id in cycle_list:
        cycle_df = df[df["cycle_index"] == cycle_id]
        curves = preprocess_cycle(cycle_df, num_points)
        if curves:
            v_new, _, dt_dv, _, _ = curves
            plt.plot(v_new, dt_dv, label=f"Cycle {cycle_id}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("dT/dV")
    plt.title("dT/dV Comparison")
    plt.legend()

    # dV/dt
    plt.subplot(1, 3, 3)
    for cycle_id in cycle_list:
        cycle_df = df[df["cycle_index"] == cycle_id]
        curves = preprocess_cycle(cycle_df, num_points)
        if curves:
            _, _, _, time_min, dv_dt = curves
            plt.plot(time_min, dv_dt, label=f"Cycle {cycle_id}")
    plt.xlabel("Time (min)")
    plt.ylabel("dV/dt (V/s)")
    plt.title("dV/dt Comparison (0-20 min)")
    plt.xlim(0, 20)  # 限制横坐标 0~20 分钟
    plt.legend()

    plt.tight_layout()
    plt.show()


# === 使用示例 ===
df = pd.read_csv(r"C:\Users\13512\Desktop\MIT数据\raw_data\el150800460649_cycles_long.csv")

# 指定要画的循环编号
cycle_list = [1, 10, 100, 200, 300, 400]
plot_multiple_cycles(df, cycle_list)
