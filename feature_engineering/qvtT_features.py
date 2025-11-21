# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

DT = 0.03  # 秒，采样间隔 30ms

# ---------- 基础统计 ----------
def series_stats(arr: np.ndarray, prefix: str) -> dict:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}{k}": np.nan for k in
                ("max","min","mean","var","skew","kurt","q1","median","q3","iqr")}
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return {
        f"{prefix}max": float(np.max(x)),
        f"{prefix}min": float(np.min(x)),
        f"{prefix}mean": float(np.mean(x)),
        f"{prefix}var": float(np.var(x)),
        f"{prefix}skew": float(skew(x, bias=False)) if x.size>2 else np.nan,
        f"{prefix}kurt": float(kurtosis(x, fisher=True, bias=False)) if x.size>3 else np.nan,  # excess
        f"{prefix}q1": float(q1),
        f"{prefix}median": float(q2),
        f"{prefix}q3": float(q3),
        f"{prefix}iqr": float(q3 - q1),
    }

# ---------- 单文件提取四类统计 ----------
def extract_qtvt_for_file(file_path: str,
                          out_q_dir: str, out_t_dir: str, out_v_dir: str, out_tseq_dir: str):
    df = pd.read_csv(file_path)
    # 排序，确保顺序稳定；有 sample_idx 就用
    sort_cols = ['cycle_index', 'sample_idx'] if 'sample_idx' in df.columns else ['cycle_index']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 逐循环，仅取放电段
    rows_q, rows_T, rows_V, rows_t = [], [], [], []
    for cyc, g in df.groupby('cycle_index'):
        dis = g[g['step_type'] == 'discharge']
        if len(dis) < 5:
            continue

        # Q/T/V 原始序列
        Q = dis['discharge_capacity'].to_numpy(float) if 'discharge_capacity' in dis.columns else None
        T = dis['temperature'].to_numpy(float) if 'temperature' in dis.columns else None
        V = dis['voltage'].to_numpy(float) if 'voltage' in dis.columns else None

        # t 时间序列（本循环放电段从 0 开始，步长 DT）
        n = len(dis)
        t = np.arange(n, dtype=float) * DT

        # 统计
        if Q is not None:
            sQ = series_stats(Q, prefix="qstat_")
            sQ['cycle_index'] = int(cyc)
            rows_q.append(sQ)
        if T is not None:
            sT = series_stats(T, prefix="tstat_")
            sT['cycle_index'] = int(cyc)
            rows_T.append(sT)
        if V is not None:
            sV = series_stats(V, prefix="vstat_")
            sV['cycle_index'] = int(cyc)
            rows_V.append(sV)

        st = series_stats(t, prefix="tseq_")
        st['cycle_index'] = int(cyc)
        rows_t.append(st)

    # 输出 4 个文件（有哪类就存哪类）
    base = os.path.basename(file_path).replace('.csv', '')
    if rows_q:
        os.makedirs(out_q_dir, exist_ok=True)
        pd.DataFrame(rows_q).sort_values('cycle_index').to_csv(
            os.path.join(out_q_dir, f"{base}_Qstat.csv"),
            index=False, encoding='utf-8-sig', float_format='%.10g'
        )
    if rows_T:
        os.makedirs(out_t_dir, exist_ok=True)
        pd.DataFrame(rows_T).sort_values('cycle_index').to_csv(
            os.path.join(out_t_dir, f"{base}_Tstat.csv"),
            index=False, encoding='utf-8-sig', float_format='%.10g'
        )
    if rows_V:
        os.makedirs(out_v_dir, exist_ok=True)
        pd.DataFrame(rows_V).sort_values('cycle_index').to_csv(
            os.path.join(out_v_dir, f"{base}_Vstat.csv"),
            index=False, encoding='utf-8-sig', float_format='%.10g'
        )
    if rows_t:
        os.makedirs(out_tseq_dir, exist_ok=True)
        pd.DataFrame(rows_t).sort_values('cycle_index').to_csv(
            os.path.join(out_tseq_dir, f"{base}_tstat.csv"),
            index=False, encoding='utf-8-sig', float_format='%.10g'
        )

    print(f"✅ {os.path.basename(file_path)} → Q/T/V/t 四类统计已输出。")

# ---------- 批量入口 ----------
def process_folder(data_dir: str,
                   out_q_dir: str, out_t_dir: str, out_v_dir: str, out_tseq_dir: str):
    for fn in os.listdir(data_dir):
        if fn.lower().endswith('.csv'):
            extract_qtvt_for_file(os.path.join(data_dir, fn),
                                  out_q_dir, out_t_dir, out_v_dir, out_tseq_dir)
    print("✅ 全部电池 Q/T/V/t 统计提取完成。")

# ======= 修改这里的路径再运行 =======
if __name__ == "__main__":
    data_dir   = r"C:\Users\13512\Desktop\MIT数据\raw_data"      # 原始电池 CSV 文件夹
    out_q_dir  = r"C:\Users\13512\Desktop\Qstat"      # 输出：*_Qstat.csv
    out_t_dir  = r"C:\Users\13512\Desktop\T_stat"      # 输出：*_Tstat.csv
    out_v_dir  = r"C:\Users\13512\Desktop\Vstat"      # 输出：*_Vstat.csv
    out_tseq_dir = r"C:\Users\13512\Desktop\tstat"   # 输出：*_tstat.csv
    process_folder(data_dir, out_q_dir, out_t_dir, out_v_dir, out_tseq_dir)
