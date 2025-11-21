import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

SAMPLE_INTERVAL = 0.03  # 每点30ms


def preprocess_cycle(cycle_df, num_points=200):
    """提取单个循环的特征"""
    discharge_df = cycle_df[cycle_df["current"] < 0].copy()
    if len(discharge_df) < 5:
        return None

    # 构造时间轴
    n_points = len(discharge_df)
    discharge_df["time"] = np.arange(n_points) * SAMPLE_INTERVAL

    # 平滑
    discharge_df["voltage_smooth"] = savgol_filter(discharge_df["voltage"], 21, 3)
    discharge_df["capacity_smooth"] = savgol_filter(discharge_df["discharge_capacity"], 21, 3)
    discharge_df["temperature_smooth"] = savgol_filter(discharge_df["temperature"], 21, 3)

    v = discharge_df["voltage_smooth"].values
    q = discharge_df["capacity_smooth"].values
    t = discharge_df["temperature_smooth"].values
    time = discharge_df["time"].values

    # === 重构 dQ/dV 和 dT/dV ===
    v_new = np.linspace(v.min(), v.max(), num_points)
    f_q = interp1d(v, q, kind="linear", fill_value="extrapolate")
    f_t = interp1d(v, t, kind="linear", fill_value="extrapolate")
    q_new = f_q(v_new)
    t_new = f_t(v_new)

    dq_dv = np.gradient(q_new, v_new)
    dt_dv = np.gradient(t_new, v_new)

    # === dV/dt ===
    dv_dt = np.gradient(v, time)

    # === 统计特征（可用于模型输入） ===
    features = {
        "dQdV_peak_pos": v_new[np.argmax(dq_dv)],
        "dQdV_peak_val": np.max(dq_dv),
        "dQdV_area": np.trapz(dq_dv, v_new),

        "dTdV_peak_pos": v_new[np.argmax(dt_dv)],
        "dTdV_peak_val": np.max(dt_dv),
        "dTdV_area": np.trapz(dt_dv, v_new),

        "dVdt_mean": np.mean(dv_dt),
        "dVdt_std": np.std(dv_dt),
        "dVdt_min": np.min(dv_dt),
        "dVdt_max": np.max(dv_dt),
    }
    return features


def process_all_cycles(file_path, output_path="cycle_features.csv"):
    df = pd.read_csv(r"C:\Users\13512\Desktop\MIT数据\raw_data\el150800460649_cycles_long.csv")
    all_features = []

    for cycle, cycle_df in df.groupby("cycle_index"):
        feats = preprocess_cycle(cycle_df)
        if feats is not None:
            feats["cycle_index"] = cycle
            all_features.append(feats)

    features_df = pd.DataFrame(all_features).sort_values("cycle_index")
    features_df.to_csv(output_path, index=False)
    print(f"✅ 已保存特征到 {output_path}")
    return features_df


# === 运行示例 ===
features_df = process_all_cycles("el150800737381_cycles_long.csv")
print(features_df.head())
