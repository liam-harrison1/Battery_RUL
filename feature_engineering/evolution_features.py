import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from numpy import trapz

def compute_dqdv_dtdv(discharge_df, n_points: int = 1000):
    d = discharge_df.dropna(subset=['voltage', 'discharge_capacity', 'temperature']).sort_values('voltage')
    if len(d) < 10:
        return np.nan, np.nan, np.nan, np.nan

    # 提取原始数据
    V = d['voltage'].to_numpy()
    Q = d['discharge_capacity'].to_numpy()
    T = d['temperature'].to_numpy()

    # 等间距电压重采样
    V_grid = np.linspace(V.min(), V.max(), n_points)

    # 插值
    fQ = interp1d(V, Q, kind='linear', fill_value='extrapolate')
    fT = interp1d(V, T, kind='linear', fill_value='extrapolate')
    Q_eq = fQ(V_grid)
    T_eq = fT(V_grid)

    # 数值求导
    dQdV = np.gradient(Q_eq, V_grid)
    dTdV = np.gradient(T_eq, V_grid)

    # 平滑
    win = 31 if len(V_grid) > 31 else max(5, len(V_grid)//2*2+1)
    dQdV_s = savgol_filter(dQdV, win, 3)
    dTdV_s = savgol_filter(dTdV, win, 3)

    # 峰值位置与面积
    peaks_q, _ = find_peaks(dQdV_s)
    peaks_t, _ = find_peaks(np.abs(dTdV_s))

    dQpos = V_grid[peaks_q[np.argmax(dQdV_s[peaks_q])]] if len(peaks_q) else np.nan
    dTpos = V_grid[peaks_t[np.argmax(np.abs(dTdV_s[peaks_t]))]] if len(peaks_t) else np.nan
    dQarea = trapz(dQdV_s, V_grid)
    dTarea = trapz(dTdV_s, V_grid)

    return dQpos, dQarea, dTpos, dTarea


def extract_dqdv_features(file_path: str, save_dir: str):
    df = pd.read_csv(file_path)
    # 确保升序排列
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    features = []

    # 只取放电循环
    for cyc, g in df.groupby('cycle_index'):
        discharge = g[g['step_type'] == 'discharge']
        if len(discharge) < 20:
            continue

        dQpos, dQarea, dTpos, dTarea = compute_dqdv_dtdv(discharge)
        features.append({
            'cycle_index': cyc,
            'dQpos(V)': dQpos,
            'dQarea': dQarea,
            'dTpos(V)': dTpos,
            'dTarea': dTarea
        })

    feat_df = pd.DataFrame(features)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(file_path).replace('.csv', '_dqdv_features.csv'))
    feat_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存特征: {out_path} (共 {len(feat_df)} 个循环)")
    return feat_df


def process_folder(data_dir: str, output_dir: str):
    """批量处理整个文件夹"""
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            print(f"正在处理 {file} ...")
            extract_dqdv_features(os.path.join(data_dir, file), output_dir)
    print("✅ 全部提取完成。")


# ========= 在这里输入路径 =========
if __name__ == "__main__":
    data_dir = r"C:\Users\13512\Desktop\MIT数据\raw_data"         # 输入数据文件夹路径
    output_dir = r"C:\Users\13512\Desktop\evolution_features"     # 输出文件夹路径
    process_folder(data_dir, output_dir)
