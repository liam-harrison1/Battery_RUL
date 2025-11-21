import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis

SAMPLE_INTERVAL = 0.03  # 每点30ms


def compute_basic_stats(x):
    """返回一组数据的统计特征"""
    return {
        "max": np.max(x),
        "min": np.min(x),
        "mean": np.mean(x),
        "var": np.var(x),
        "skew": skew(x, bias=False),
        "kurt": kurtosis(x, bias=False),
        "quar1": np.quantile(x, 0.25),
        "medi": np.median(x),
        "quar3": np.quantile(x, 0.75),
        "iqr": np.quantile(x, 0.75) - np.quantile(x, 0.25)
    }


def preprocess_cycle(cycle_df, num_points=200):
    """
    输入: 单个循环的原始数据
    输出: 特征字典
    """
    discharge_df = cycle_df[cycle_df["current"] < 0].copy()
    if len(discharge_df) < 5:
        return None

    # 构造时间轴 (秒 → 分钟)
    n_points = len(discharge_df)
    time_s = np.arange(n_points) * SAMPLE_INTERVAL
    time_min = time_s / 60.0

    # 平滑
    discharge_df["voltage_smooth"] = savgol_filter(discharge_df["voltage"], 21, 3)
    discharge_df["capacity_smooth"] = savgol_filter(discharge_df["capacity"], 21, 3)
    discharge_df["temperature_smooth"] = savgol_filter(discharge_df["temperature"], 21, 3)

    v = discharge_df["voltage_smooth"].values
    q = discharge_df["capacity_smooth"].values
    t = discharge_df["temperature_smooth"].values

    # 重构 dQ/dV 和 dT/dV
    v_new = np.linspace(v.min(), v.max(), num_points)
    f_q = interp1d(v, q, kind="linear", fill_value="extrapolate")
    f_t = interp1d(v, t, kind="linear", fill_value="extrapolate")
    q_new = f_q(v_new)
    t_new = f_t(v_new)

    dq_dv = np.gradient(q_new, v_new)
    dt_dv = np.gradient(t_new, v_new)

    # dV/dt
    dv_dt = np.gradient(v, time_s)

    # === 特征计算 ===
    features = {"cycle_index": cycle_df["cycle_index"].iloc[0]}

    # 直接特征
    features["Qd"] = np.max(q)  # 放电容量
    features["td"] = time_s[-1]  # 放电时间 (s)

    # 如果有阻抗测量列，比如 'resistance'
    if "resistance" in cycle_df.columns:
        features["Ro"] = np.mean(cycle_df["resistance"].values)

    # 统计特征
    for prefix, arr in {
        "Q": q,
        "t": time_s,
        "V": v,
        "dQV": dq_dv,
        "dTV": dt_dv,
        "dVt": dv_dt
    }.items():
        stats = compute_basic_stats(arr)
        for k, val in stats.items():
            features[f"{prefix}_{k}"] = val

    # 演化特征
    features["dQ_pos"] = v_new[np.argmax(dq_dv)]
    features["dQ_area"] = np.trapz(dq_dv, v_new)
    features["dT_pos"] = v_new[np.argmax(dt_dv)]
    features["dT_area"] = np.trapz(dt_dv, v_new)

    return features


def extract_features_from_csv(file_path, output_path="features_matrix.csv"):
    df = pd.read_csv(file_path)
    all_features = []
    for cycle, cycle_df in df.groupby("cycle_index"):
        feats = preprocess_cycle(cycle_df)
        if feats is not None:
            all_features.append(feats)
    features_df = pd.DataFrame(all_features).sort_values("cycle_index")
    features_df.to_csv(output_path, index=False)
    print(f"✅ 特征矩阵已保存到 {output_path}, 共 {features_df.shape[0]} 个循环, {features_df.shape[1]} 个特征")
    return features_df


# === 使用示例 ===
features_df = extract_features_from_csv("el150800737381_cycles_long.csv")
print(features_df.head())
