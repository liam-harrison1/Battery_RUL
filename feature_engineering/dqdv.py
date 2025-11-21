import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def reconstruct_dqdv_curve(discharge_df, n_points: int = 1000):
    """重构单循环的 dQ/dV 曲线"""
    d = discharge_df.dropna(subset=['voltage', 'discharge_capacity']).sort_values('voltage')
    if len(d) < 10:
        return None, None

    V = d['voltage'].to_numpy()
    Q = d['discharge_capacity'].to_numpy()

    # 等间距电压
    V_grid = np.linspace(V.min(), V.max(), n_points)
    fQ = interp1d(V, Q, kind='linear', fill_value='extrapolate')
    Q_eq = fQ(V_grid)

    # 求导并平滑
    dQdV = np.gradient(Q_eq, V_grid)
    win = 31 if len(V_grid) > 31 else max(5, len(V_grid)//2*2+1)
    dQdV_s = savgol_filter(dQdV, win, 3)
    return V_grid, dQdV_s


def extract_all_dqdv_curves(file_path: str, save_dir: str, n_points: int = 1000):
    """
    对单个电池文件重构所有循环的 dQ/dV 曲线并保存为 CSV。
    输出格式：每列是一个循环的 dQ/dV 曲线。
    """
    df = pd.read_csv(file_path)
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    V_grid_global = None
    curve_dict = {}

    for cyc, g in df.groupby('cycle_index'):
        discharge = g[g['step_type'] == 'discharge']
        if len(discharge) < 20:
            continue
        V_grid, dQdV_s = reconstruct_dqdv_curve(discharge, n_points=n_points)
        if V_grid is None:
            continue

        # 对齐所有循环的电压网格
        if V_grid_global is None:
            V_grid_global = V_grid
        curve_dict[f'cycle_{int(cyc)}'] = dQdV_s

    if not curve_dict:
        print(f"⚠️ {os.path.basename(file_path)} 没有有效循环。")
        return None

    out_df = pd.DataFrame({'voltage': V_grid_global})
    for k, v in curve_dict.items():
        out_df[k] = v

    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.basename(file_path).replace('.csv', '_dqdv_curves.csv')
    out_path = os.path.join(save_dir, out_name)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存 dQ/dV 曲线: {out_path} ({len(curve_dict)} 个循环)")
    return out_df


def process_folder(data_dir: str, output_dir: str):
    """批量处理整个电池数据文件夹"""
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            print(f"正在处理 {file} ...")
            extract_all_dqdv_curves(os.path.join(data_dir, file), output_dir)
    print("✅ 全部电池 dQ/dV 曲线输出完成。")


# ========= 主程序入口 =========
if __name__ == "__main__":
    data_dir = r"C:\Users\13512\Desktop\MIT数据\raw_data"            # 输入文件夹
    output_dir = r"C:\Users\13512\Desktop\dQdV"    # 输出文件夹
    process_folder(data_dir, output_dir)
