import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def series_stats(arr: np.ndarray):
    """对一条 dQ/dV 曲线（一个循环）计算 10 个统计量；自动忽略 NaN。"""
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(xmax=np.nan, xmin=np.nan, xmean=np.nan, xvar=np.nan,
                    xskew=np.nan, xkurt=np.nan,
                    xquar1=np.nan, xmedi=np.nan, xquar3=np.nan, xiqr=np.nan)
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return dict(
        xmax  = float(np.max(x)),
        xmin  = float(np.min(x)),
        xmean = float(np.mean(x)),
        xvar  = float(np.var(x)),
        xskew = float(skew(x, bias=False)),
        xkurt = float(kurtosis(x, fisher=True, bias=False)),  # 与论文一致用峰度（减3）
        xquar1= float(q1),
        xmedi = float(q2),
        xquar3= float(q3),
        xiqr  = float(q3 - q1),
    )

def dqdv_stats_from_curves_file(curves_csv_path: str, save_dir: str):
    """
    输入：单个电池的 dQ/dV 曲线文件（上一阶段生成的 …_dqdv_curves.csv）
         该文件格式：第一列 voltage，其余列为 cycle_数字
    输出：该电池的每循环统计特征（…_dqdv_stats.csv）
    """
    df = pd.read_csv(curves_csv_path)
    # 识别循环列（以 'cycle_' 开头）
    cycle_cols = [c for c in df.columns if c.startswith('cycle_')]
    if not cycle_cols:
        print(f"⚠️ 没找到循环列：{os.path.basename(curves_csv_path)}")
        return None

    rows = []
    for col in sorted(cycle_cols, key=lambda s: int(s.split('_')[1])):
        stats = series_stats(df[col].values)
        cyc_idx = int(col.split('_')[1])
        rows.append({'cycle_index': cyc_idx, **stats})

    out_df = pd.DataFrame(rows).sort_values('cycle_index')
    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.basename(curves_csv_path).replace('_dqdv_curves.csv', '_dqdv_stats.csv')
    out_path = os.path.join(save_dir, out_name)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存 dQ/dV 统计特征：{out_path}（{len(out_df)} 个循环）")
    return out_df

def process_folder(curves_dir: str, output_dir: str):
    """
    批量处理一个文件夹内所有电池的 dQ/dV 曲线文件（*_dqdv_curves.csv）
    为每个电池输出对应的 *_dqdv_stats.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    for fn in os.listdir(curves_dir):
        if fn.endswith('_dqdv_curves.csv'):
            dqdv_stats_from_curves_file(os.path.join(curves_dir, fn), output_dir)
    print("✅ 全部电池 dQ/dV 统计特征提取完成。")

# ===== 在这里填写路径 =====
if __name__ == "__main__":
    curves_dir = r"C:\Users\13512\Desktop\dQdV"   # 输入：上一步输出的曲线目录
    output_dir = r"C:\Users\13512\Desktop\dqdv_statistic"   # 输出：统计特征目录
    process_folder(curves_dir, output_dir)
