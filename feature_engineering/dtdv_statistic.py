import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import skew, kurtosis

# ---------- 核心：重构单循环 dT/dV 曲线 ----------
def reconstruct_dtdv_curve(discharge_df, n_points: int = 1000):
    """
    放电段按电压等间距重采样，插值 T(V)，再数值求导得到 dT/dV，并作 Savitzky–Golay 平滑。
    返回: V_grid, dTdV_s；若样本太少返回 (None, None)
    """
    cols_needed = ['voltage', 'temperature']
    d = discharge_df.dropna(subset=cols_needed).sort_values('voltage')
    if len(d) < 10:
        return None, None

    V = d['voltage'].to_numpy()
    T = d['temperature'].to_numpy()

    # 等间距电压重采样
    V_grid = np.linspace(V.min(), V.max(), n_points)
    fT = interp1d(V, T, kind='linear', fill_value='extrapolate')
    T_eq = fT(V_grid)

    # 数值求导 & 平滑
    dTdV = np.gradient(T_eq, V_grid)
    win = 31 if len(V_grid) > 31 else max(5, (len(V_grid)//2)*2 + 1)  # 奇数窗口
    dTdV_s = savgol_filter(dTdV, win, 3)
    return V_grid, dTdV_s

# ---------- 统计量（改为 dvdt_* 命名） ----------
def dvdt_series_stats(arr: np.ndarray):
    """对一条 dT/dV 曲线计算 10 个统计量；显式 dvdt_* 命名；自动忽略 NaN。"""
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(
            dvdt_max=np.nan, dvdt_min=np.nan, dvdt_mean=np.nan, dvdt_var=np.nan,
            dvdt_skew=np.nan, dvdt_kurt=np.nan, dvdt_q1=np.nan,
            dvdt_median=np.nan, dvdt_q3=np.nan, dvdt_iqr=np.nan
        )
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return dict(
        dvdt_max=float(np.max(x)),
        dvdt_min=float(np.min(x)),
        dvdt_mean=float(np.mean(x)),
        dvdt_var=float(np.var(x)),
        dvdt_skew=float(skew(x, bias=False)),
        dvdt_kurt=float(kurtosis(x, fisher=True, bias=False)),  # excess kurtosis
        dvdt_q1=float(q1),
        dvdt_median=float(q2),
        dvdt_q3=float(q3),
        dvdt_iqr=float(q3 - q1),
    )

# ---------- 从原始电池 CSV 生成 dT/dV 曲线 & 统计 ----------
def extract_dtdv_for_file(file_path: str, curves_out_dir: str, stats_out_dir: str, n_points: int = 1000):
    df = pd.read_csv(file_path)
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 逐循环重构曲线，汇总到一个“电压为行”的大表
    V_grid_global = None
    curve_cols = {}
    stats_rows = []

    for cyc, g in df.groupby('cycle_index'):
        discharge = g[g['step_type'] == 'discharge']
        if len(discharge) < 20:
            continue

        V_grid, dTdV_s = reconstruct_dtdv_curve(discharge, n_points=n_points)
        if V_grid is None:
            continue

        if V_grid_global is None:
            V_grid_global = V_grid

        # 曲线收集
        col_name = f'cycle_{int(cyc)}'
        curve_cols[col_name] = dTdV_s

        # 峰位置/峰面积（演化特征） + 10 统计量（dvdt_* 命名）
        peaks, _ = find_peaks(np.abs(dTdV_s))
        dTpos = float(V_grid[peaks[np.argmax(np.abs(dTdV_s[peaks]))]]) if len(peaks) else np.nan
        dTarea = float(np.trapz(dTdV_s, V_grid))
        s = dvdt_series_stats(dTdV_s)

        stats_rows.append({
            'cycle_index': int(cyc),
            'dTpos(V)': dTpos,        # 若也想统一前缀，可改成 'dtdv_peak_pos_v'
            'dTarea': dTarea,         # 同上，可改成 'dtdv_area'
            **s                       # 展开 dvdt_* 统计量
        })

    # 若无有效循环，退出
    if not curve_cols:
        print(f"⚠️ {os.path.basename(file_path)} 无有效放电循环，跳过。")
        return

    # 保存曲线 CSV（每列一个循环）
    os.makedirs(curves_out_dir, exist_ok=True)
    curves_df = pd.DataFrame({'voltage': V_grid_global})
    for k, v in curve_cols.items():
        curves_df[k] = v
    curves_out = os.path.join(curves_out_dir, os.path.basename(file_path).replace('.csv', '_dtdv_curves.csv'))
    curves_df.to_csv(curves_out, index=False, encoding='utf-8-sig')

    # 保存统计 CSV（每行一个循环）
    os.makedirs(stats_out_dir, exist_ok=True)
    stats_df = pd.DataFrame(stats_rows).sort_values('cycle_index')
    stats_out = os.path.join(stats_out_dir, os.path.basename(file_path).replace('.csv', '_dtdv_stats.csv'))
    stats_df.to_csv(stats_out, index=False, encoding='utf-8-sig')

    print(f"✅ {os.path.basename(file_path)} → 曲线: {curves_out}；统计: {stats_out}")

# ---------- 批量入口 ----------
def process_folder(data_dir: str, curves_out_dir: str, stats_out_dir: str, n_points: int = 1000):
    for fn in os.listdir(data_dir):
        if fn.lower().endswith('.csv'):
            fp = os.path.join(data_dir, fn)
            extract_dtdv_for_file(fp, curves_out_dir, stats_out_dir, n_points=n_points)
    print("✅ 全部电池 dT/dV 曲线与统计已生成。")

# ======== 路径在这里改 ========
if __name__ == '__main__':
    data_dir = r"C:\Users\13512\Desktop\MIT数据\raw_data"              # 输入：原始电池 CSV 文件夹
    curves_out_dir = r"C:\Users\13512\Desktop\dtdv曲线" # 输出：每电池 dT/dV 曲线
    stats_out_dir = r"C:\Users\13512\Desktop\dtdv_statistic_features"   # 输出：每电池 dT/dV 统计
    process_folder(data_dir, curves_out_dir, stats_out_dir, n_points=1000)
