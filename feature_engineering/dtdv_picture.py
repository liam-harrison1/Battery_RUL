# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ---------- 基础：稳健的 SavGol ----------
def safe_savgol(y: np.ndarray, win_default: int = 31, poly: int = 3) -> np.ndarray:
    n = len(y)
    if n < 3:
        return y
    wl = min(win_default, n if n % 2 == 1 else n - 1)  # ≤长度且奇数
    wl = max(wl, 3)
    if wl % 2 == 0:
        wl -= 1
    po = min(poly, wl - 1)
    try:
        return savgol_filter(y, wl, po)
    except Exception:
        # 回退到居中滑动平均
        k = min(5, n)
        return pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()

# ---------- 重构单循环 dT/dV ----------
def reconstruct_dtdv_curve(discharge_df: pd.DataFrame, n_points: int = 1000):
    """
    放电段按电压等间距重采样，再对 T(V) 求导得到 dT/dV，并平滑。
    返回: (V_grid, dTdV_s)；若数据不足返回 (None, None)
    """
    need = ['voltage', 'temperature']
    d = discharge_df.dropna(subset=need).copy()
    if d.empty or len(d) < 10:
        return None, None

    # 按电压排序并去重，避免插值异常
    d = d.sort_values('voltage')
    V = d['voltage'].to_numpy()
    T = d['temperature'].to_numpy()
    V, uniq_idx = np.unique(V, return_index=True)
    T = T[uniq_idx]
    if V.size < 10 or np.isclose(V.min(), V.max()):
        return None, None

    # 等间距电压网格 + 线性插值
    V_grid = np.linspace(V.min(), V.max(), n_points)
    fT = interp1d(V, T, kind='linear', fill_value='extrapolate', bounds_error=False)
    T_eq = fT(V_grid)

    # 数值求导 + 平滑
    dTdV = np.gradient(T_eq, V_grid)
    dTdV_s = safe_savgol(dTdV, 31, 3)
    return V_grid, dTdV_s

# ---------- 选择要画的循环（自动兜底） ----------
def choose_cycles_to_plot(df: pd.DataFrame, desired=None, k: int = 6):
    """
    优先使用 desired（如 [1,10,100,200,300,400]），
    若数据不含这些循环，则按分位数自动取 k 个代表循环。
    """
    cyc_all = np.array(sorted(df['cycle_index'].unique()))
    if cyc_all.size == 0:
        return []
    if desired:
        chosen = [int(cyc_all[np.argmin(np.abs(cyc_all - c))]) for c in desired]
        chosen = sorted(set(chosen))
        return [c for c in chosen if c in cyc_all]
    # 自动均匀取 k 个
    idx = np.unique(np.linspace(0, len(cyc_all) - 1, num=min(k, len(cyc_all))).astype(int))
    return [int(cyc_all[i]) for i in idx]

# ---------- 给单个文件画 dT/dV 对比图 ----------
def plot_dtdv_for_file(file_path: str, save_dir: str, cycles_to_plot=None, n_points: int = 1000):
    df = pd.read_csv(file_path)
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if cycles_to_plot is None:
        cycles_to_plot = choose_cycles_to_plot(df, desired=[1, 10, 100, 200, 300, 400], k=6)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, max(1, len(cycles_to_plot))))

    plotted = 0
    for color, cyc in zip(colors, cycles_to_plot):
        g = df[df['cycle_index'] == cyc]
        discharge = g[g['step_type'] == 'discharge']
        Vg, dTdV = reconstruct_dtdv_curve(discharge, n_points=n_points)
        if Vg is None:
            continue
        plt.plot(Vg, dTdV, label=f'Cycle {cyc}', color=color, linewidth=1.5)
        plotted += 1

    if plotted == 0:
        plt.close()
        print(f"⚠️ {os.path.basename(file_path)} 没有可绘制的放电循环。")
        return

    plt.xlabel('Voltage (V)')
    plt.ylabel('dT/dV (°C/V)')
    plt.title(f'dT/dV Curves: {os.path.basename(file_path)}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_png = os.path.join(save_dir, os.path.basename(file_path).replace('.csv', '_dtdv_plot.png'))
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ 已保存 {out_png}")

# ---------- 汇总图：把每个电池的“最后一个循环”的 dT/dV 叠加在一张图 ----------
def plot_summary_last_cycle(data_dir: str, save_path: str, n_points: int = 1000, alpha: float = 0.6, max_batteries: int = 50):
    plt.figure(figsize=(10, 7))
    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    files = files[:max_batteries]

    n_plotted = 0
    for fn in files:
        fp = os.path.join(data_dir, fn)
        df = pd.read_csv(fp)
        sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
        df = df.sort_values(sort_cols).reset_index(drop=True)

        if 'cycle_index' not in df.columns or 'step_type' not in df.columns:
            continue
        last_cyc = int(df['cycle_index'].max())
        dis = df[(df['cycle_index'] == last_cyc) & (df['step_type'] == 'discharge')]
        Vg, dTdV = reconstruct_dtdv_curve(dis, n_points=n_points)
        if Vg is None:
            continue

        label = os.path.splitext(fn)[0][:28]  # 标签太长就截断
        plt.plot(Vg, dTdV, linewidth=1.2, alpha=alpha, label=label)
        n_plotted += 1

    if n_plotted == 0:
        plt.close()
        print("⚠️ 汇总图：没有可绘制的电池。")
        return

    # 电池多时避免爆炸的图例
    if n_plotted <= 15:
        plt.legend(fontsize=8)
    else:
        plt.legend().set_visible(False)

    plt.xlabel('Voltage (V)')
    plt.ylabel('dT/dV (°C/V)')
    plt.title('Summary: Last-Cycle dT/dV of All Batteries')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 已保存汇总图 {save_path}（{n_plotted} 个电池）")

# ---------- 批量入口 ----------
def process_folder_plot_dtdv(data_dir: str, out_dir: str, cycles_to_plot=None, n_points: int = 1000):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(data_dir):
        if fn.lower().endswith('.csv'):
            plot_dtdv_for_file(os.path.join(data_dir, fn), out_dir, cycles_to_plot, n_points)
    # 汇总图
    summary_path = os.path.join(out_dir, 'all_batteries_dtdv_summary.png')
    plot_summary_last_cycle(data_dir, summary_path, n_points=n_points)

# ========= 主程序入口：把路径改成你的 =========
if __name__ == "__main__":
    data_dir = r"D:\battery_data"       # 输入：原始电池 CSV 文件夹
    out_dir  = r"D:\battery_plots_dtdv" # 输出：每电池 dT/dV 图 + 汇总图
    # 若想固定循环，取消下一行注释并设置你想要的序号；否则自动均匀抽取 6 个代表循环
    # cycles = [1, 10, 100, 200, 300, 400]
    cycles = None
    process_folder_plot_dtdv(data_dir, out_dir, cycles_to_plot=cycles, n_points=1000)
