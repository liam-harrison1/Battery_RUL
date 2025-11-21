import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis

SAMPLE_INTERVAL = 0.03  # 每点30ms
RESAMPLE_POINTS = 1000  # 论文用 1000 点重采样


def compute_stats(arr):
    """返回论文定义的10个统计量"""
    return {
        "max": np.max(arr),
        "min": np.min(arr),
        "mean": np.mean(arr),
        "var": np.var(arr),
        "skew": skew(arr, bias=False),
        "kurt": kurtosis(arr, bias=False),
        "quar1": np.quantile(arr, 0.25),
        "medi": np.median(arr),
        "quar3": np.quantile(arr, 0.75),
        "iqr": np.quantile(arr, 0.75) - np.quantile(arr, 0.25),
    }


def preprocess_cycle(cycle_df):
    """提取单个循环的79个特征"""
    discharge_df = cycle_df[cycle_df["current"] < 0].copy()
    if len(discharge_df) < 5:
        return None

    # 时间轴 (秒 → 分钟)
    n_points = len(discharge_df)
    time_s = np.arange(n_points) * SAMPLE_INTERVAL

    # 平滑
    discharge_df["V"] = savgol_filter(discharge_df["voltage"], 21, 3)
    discharge_df["Q"] = savgol_filter(discharge_df["discharge_capacity"], 21, 3)
    discharge_df["T"] = savgol_filter(discharge_df["temperature"], 21, 3)

    v = discharge_df["V"].values
    q = discharge_df["Q"].values
    t = discharge_df["T"].values

    # === 重采样 (论文里1000点均匀化) ===
    v_new = np.linspace(v.min(), v.max(), RESAMPLE_POINTS)
    f_q = interp1d(v, q, kind="linear", fill_value="extrapolate")
    f_t = interp1d(v, t, kind="linear", fill_value="extrapolate")
    q_new = f_q(v_new)
    t_new = f_t(v_new)

    dq_dv = np.gradient(q_new, v_new)
    dt_dv = np.gradient(t_new, v_new)
    dv_dt = np.gradient(v, time_s)

    features = {"cycle_index": cycle_df["cycle_index"].iloc[0]}

    # === 直接特征 ===
    features["Qd"] = q.max() - q.min()  # 放电容量
    features["td"] = time_s[-1]  # 放电时间
    if "internal_resistance" in cycle_df.columns:
        features["Ro"] = np.mean(cycle_df["internal_resistance"].values)
    else:
        features["Ro"] = np.nan
    # 如果有 phase 标记，可单独算 tcc, tcv
    features["tcc"], features["tcv"] = np.nan, np.nan

    # === 统计特征 ===
    seqs = {
        "Q": q,
        "V": v,
        "T": t,
        "t": time_s,
        "dQV": dq_dv,
        "dTV": dt_dv,
        "dVt": dv_dt
    }
    for prefix, arr in seqs.items():
        stats = compute_stats(arr)
        for k, val in stats.items():
            features[f"{prefix}_{k}"] = val

    # === 演化特征 ===
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
    print(f"✅ 输出 {features_df.shape[0]} 个循环, {features_df.shape[1]} 个特征 → {output_path}")
    return features_df


# === 使用示例 ===
features_df = extract_features_from_csv(r"C:\Users\13512\Desktop\MIT数据\raw_data\el150800460649_cycles_long.csv")
print(features_df.head())
