# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis

DT = 0.03  # 采样间隔(秒)，论文/数据说明一致

def safe_savgol(y, win=7, poly=3):
    n = len(y)
    if n < 3: return np.asarray(y, float)
    wl = min(win, n if n % 2 == 1 else n - 1)
    wl = max(wl, 3)
    if wl % 2 == 0: wl -= 1
    po = min(poly, wl - 1)
    try:
        return savgol_filter(y, wl, po)
    except Exception:
        k = min(5, n)
        return pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()

def dvdt_stats(arr: np.ndarray):
    x = np.asarray(arr, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(dvdt_max=np.nan, dvdt_min=np.nan, dvdt_mean=np.nan, dvdt_var=np.nan,
                    dvdt_skew=np.nan, dvdt_kurt=np.nan, dvdt_q1=np.nan,
                    dvdt_median=np.nan, dvdt_q3=np.nan, dvdt_iqr=np.nan)
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return dict(
        dvdt_max=float(np.max(x)),
        dvdt_min=float(np.min(x)),
        dvdt_mean=float(np.mean(x)),
        dvdt_var=float(np.var(x)),
        dvdt_skew=float(skew(x, bias=False)),
        dvdt_kurt=float(kurtosis(x, fisher=True, bias=False)),
        dvdt_q1=float(q1),
        dvdt_median=float(q2),
        dvdt_q3=float(q3),
        dvdt_iqr=float(q3 - q1)
    )

def extract_dvdt_stats_for_file(file_path: str, stats_out_dir: str):
    df = pd.read_csv(file_path)
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    rows = []
    for cyc, g in df.groupby('cycle_index'):
        dis = g[g['step_type'] == 'discharge']
        if len(dis) < 5:  # 太短就跳过
            continue
        V = dis['voltage'].astype(float).to_numpy()
        dVdt = np.gradient(V, DT)
        dVdt_s = safe_savgol(dVdt, 31, 3)
        stats = dvdt_stats(dVdt_s)
        stats['cycle_index'] = int(cyc)
        rows.append(stats)

    if not rows:
        print(f"⚠️ {os.path.basename(file_path)} 无有效放电循环，跳过。")
        return

    out_df = pd.DataFrame(rows).sort_values('cycle_index')
    os.makedirs(stats_out_dir, exist_ok=True)
    out_path = os.path.join(stats_out_dir, os.path.basename(file_path).replace('.csv', '_dvdt_stats.csv'))
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ dV/dt 统计已保存：{out_path}（{len(out_df)} 个循环）")

def process_folder(data_dir: str, stats_out_dir: str):
    os.makedirs(stats_out_dir, exist_ok=True)
    for fn in os.listdir(data_dir):
        if fn.lower().endswith('.csv'):
            extract_dvdt_stats_for_file(os.path.join(data_dir, fn), stats_out_dir)
    print("✅ 全部电池 dV/dt 统计特征提取完成。")

if __name__ == '__main__':
    data_dir = r"C:\Users\13512\Desktop\MIT数据\raw_data"           # 输入：原始电池 CSV 文件夹
    stats_out_dir = r"C:\Users\13512\Desktop\dVdt_statistic_features"# 输出：每电池 *_dvdt_stats.csv
    process_folder(data_dir, stats_out_dir)
