import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# 采样间隔(秒)
DT = 0.03

def safe_interp(y: pd.Series) -> np.ndarray:
    """线性插值填补 NaN/Inf；若全是非法值则用0."""
    x = y.astype(float).to_numpy(copy=True)
    n = len(x)
    if n == 0:
        return x
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return np.zeros(n, dtype=float)
    if mask.sum() == n:
        return x
    idx = np.arange(n)
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x

def safe_savgol(arr: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
    """稳健 Savitzky-Golay：自动缩窗、处理短序列/异常，失败则回退到滑动平均。"""
    n = len(arr)
    if n == 0:
        return arr
    # 动态窗口：不超过长度，且为奇数
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    wl = max(wl, 3)                     # 至少3
    if wl % 2 == 0:
        wl -= 1
    po = min(polyorder, wl - 1)
    try:
        return savgol_filter(arr, wl, po, mode='interp')
    except Exception:
        # 回退：中心滑动平均
        k = min(5, n)
        return pd.Series(arr).rolling(k, center=True, min_periods=1).mean().to_numpy()

def extract_features(file_path: str, save_dir: str, window_length: int = 21, polyorder: int = 3) -> pd.DataFrame:
    """按你数据的列名计算：Ro、充电时间、放电时间、放电容量；每循环一行。"""
    df = pd.read_csv(file_path)

    # 排序保证时间顺序（sample_idx 有就用它，没有也不影响）
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    features = []
    for cyc, g in df.groupby('cycle_index', sort=True):
        if len(g) < 5:   # 极短循环直接跳过
            continue

        # 分步
        charge = g[g['step_type'] == 'charge']
        discharge = g[g['step_type'] == 'discharge']

        # === 平滑：每个子序列单独做，且先插值 ===
        if len(g) >= 3:
            v = safe_interp(g['voltage'])
            v_s = safe_savgol(v, window_length, polyorder)
        else:
            v_s = g['voltage'].to_numpy()

        if len(charge) >= 3:
            qc = safe_interp(charge['charge_capacity'])
            qc_s = safe_savgol(qc, window_length, polyorder)
        else:
            qc_s = charge['charge_capacity'].to_numpy()

        if len(discharge) >= 3:
            qd = safe_interp(discharge['discharge_capacity'])
            qd_s = safe_savgol(qd, window_length, polyorder)
        else:
            qd_s = discharge['discharge_capacity'].to_numpy()

        # 特征
        t_charge = len(charge) * DT / 60.0
        t_discharge = len(discharge) * DT / 60.0
        Qd = float(qd_s.max() - qd_s.min()) if len(qd_s) else np.nan
        Ro = float(g['internal_resistance'].astype(float).replace([np.inf, -np.inf], np.nan).mean())

        features.append({
            'cycle_index': int(cyc),
            'Ro(Ohm)': Ro,
            'Qd(Ah)': Qd,
            't_charge(min)': t_charge,
            't_discharge(min)': t_discharge,
            'points_total': int(len(g)),
            'points_charge': int(len(charge)),
            'points_discharge': int(len(discharge)),
        })

    feat_df = pd.DataFrame(features).sort_values('cycle_index')
    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.basename(file_path).replace('.csv', '_features.csv')
    out_path = os.path.join(save_dir, out_name)
    feat_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'✅ 保存: {out_path}  (cycles={len(feat_df)})')
    return feat_df

def process_folder(data_dir: str, output_dir: str, window_length: int = 21, polyorder: int = 3):
    os.makedirs(output_dir, exist_ok=True)
    for fn in os.listdir(data_dir):
        if fn.lower().endswith('.csv'):
            print(f'处理 {fn} ...')
            extract_features(os.path.join(data_dir, fn), output_dir, window_length, polyorder)
    print('✅ 全部处理完成')

# ===== 在这里填你的路径 =====
if __name__ == '__main__':
    data_dir = r"C:\Users\13512\Desktop\MIT数据\raw_data"         # 输入数据文件夹
    output_dir = r"C:\Users\13512\Desktop\direct_features"   # 输出特征文件夹
    process_folder(data_dir, output_dir, window_length=21, polyorder=3)
