# -*- coding: utf-8 -*-
import os, re, json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ========= 在这里改成你的特征文件所在目录（一个文件夹放所有电池的各类特征CSV） =========
BASE_DIR = r"C:\Users\13512\Desktop\features"  # 例如：里面有 el150800440551_cycles_long_dqdv_stats.csv 等

OUT_COMBINED_DIR = r"C:\Users\13512\Desktop\features_mixed"
OUT_NORM_DIR     = r"C:\Users\13512\Desktop\quanjuguiyihua"
SCALER_JSON      = os.path.join(OUT_COMBINED_DIR, "minmax_stats.json")

# 需要参与合并的类型（通过后缀识别）；曲线 CSV（*_dqdv_curves.csv）不会合并
SUFFIX2KIND = {
    "_dqdv_stats.csv": "dqdv",
    "_dtdv_stats.csv": "dtdv",
    "_dvdt_stats.csv": "dvdt",
    "_direct_stats.csv": "direct",
    "_features.csv": "direct",          # 你给的 direct 特征文件名可能是这个
    "_Qstat.csv": "qstat",
    "_Tstat.csv": "tstat",
    "_Vstat.csv": "vstat",
    "_tstat.csv": "tseq",
}
IGNORE_SUFFIX = ("_dqdv_curves.csv",)  # 不合并曲线矩阵

# ---------- 工具 ----------
def list_csv(dirp: str) -> List[str]:
    return [os.path.join(root, f)
            for root, _, files in os.walk(dirp)
            for f in files if f.lower().endswith(".csv")]

def extract_battery_id_from_name(path: str) -> str:
    """
    默认取 '_cycles' 或 '-cycles' 之前的部分作为电池编号；
    若不存在，则取最长的字母数字串（>=8）。
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    m = re.split(r'(_cycles|-cycles)', stem, maxsplit=1, flags=re.IGNORECASE)
    if len(m) >= 2 and m[0]:
        return m[0]
    blocks = re.findall(r'[A-Za-z0-9]+', stem)
    if blocks:
        blocks.sort(key=len, reverse=True)
        if len(blocks[0]) >= 8:
            return blocks[0]
    return stem

def detect_kind_by_suffix(path: str) -> Optional[str]:
    for suf, kind in SUFFIX2KIND.items():
        if path.endswith(suf):
            return kind
    for suf in IGNORE_SUFFIX:
        if path.endswith(suf):
            return "ignore"
    return None  # 不认识的文件

def load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ 读取失败: {path} -> {e}")
        return None

def ensure_cycle_index(df: pd.DataFrame, tag: str):
    if 'cycle_index' in df.columns:
        return df
    alt = [c for c in df.columns if c.lower() == 'cycle_index']
    if alt:
        return df.rename(columns={alt[0]: 'cycle_index'})
    raise ValueError(f"{tag}: 缺少 cycle_index 列")

def standardize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    列名标准化：
    - dqdv: dQpos(V)->dqdv_peak_pos_v, dQarea->dqdv_area; 其余数字列前缀 dqdv_
    - dtdv: dTpos(V)->dtdv_peak_pos_v, dTarea->dtdv_area; 若列是 dvdt_*（你之前那版），改成 dtdv_*
    - dvdt: 时间导数，保留 dvdt_* 前缀
    - q/t/v/tseq: 统一加 qstat_/tstat_/vstat_/tseq_ 前缀
    - direct: Ro/ Qd/ t_charge/ t_discharge 做统一命名，其余数字列加 direct_ 前缀
    """
    out = ensure_cycle_index(df.copy(), kind)

    if kind == "dqdv":
        out.rename(columns={'dQpos(V)':'dqdv_peak_pos_v','dQarea':'dqdv_area'}, inplace=True)
        for c in list(out.columns):
            if c in ('cycle_index','dqdv_peak_pos_v','dqdv_area'): continue
            if not c.startswith('dqdv_') and np.issubdtype(out[c].dtype, np.number):
                out.rename(columns={c: f'dqdv_{c}'}, inplace=True)

    elif kind == "dtdv":
        out.rename(columns={'dTpos(V)':'dtdv_peak_pos_v','dTarea':'dtdv_area'}, inplace=True)
        for c in list(out.columns):
            if c in ('cycle_index','dtdv_peak_pos_v','dtdv_area'): continue
            if c.startswith('dvdt_'):
                out.rename(columns={c: c.replace('dvdt_', 'dtdv_')}, inplace=True)
            elif not c.startswith('dtdv_') and np.issubdtype(out[c].dtype, np.number):
                out.rename(columns={c: f'dtdv_{c}'}, inplace=True)

    elif kind == "dvdt":
        for c in list(out.columns):
            if c=='cycle_index': continue
            if not c.startswith('dvdt_') and np.issubdtype(out[c].dtype, np.number):
                out.rename(columns={c: f'dvdt_{c}'}, inplace=True)

    elif kind in ("qstat","tstat","vstat","tseq"):
        prefix = {"qstat":"qstat_","tstat":"tstat_","vstat":"vstat_","tseq":"tseq_"}[kind]
        for c in list(out.columns):
            if c=='cycle_index': continue
            if not c.startswith(prefix) and np.issubdtype(out[c].dtype, np.number):
                out.rename(columns={c: f'{prefix}{c}'}, inplace=True)

    elif kind == "direct":
        out.rename(columns={
            'Ro(Ohm)':'ro_ohm',
            'Ro':'ro_ohm',
            'Qd(Ah)':'qd_ah',
            'Qd':'qd_ah',
            't_charge(min)':'t_charge_min',
            't_discharge(min)':'t_discharge_min'
        }, inplace=True)
        for c in list(out.columns):
            if c=='cycle_index' or c in ('ro_ohm','qd_ah','t_charge_min','t_discharge_min'): continue
            if not c.startswith('direct_') and np.issubdtype(out[c].dtype, np.number):
                out.rename(columns={c: f'direct_{c}'}, inplace=True)

    # 只保留数值列 + cycle_index
    keep = ['cycle_index'] + [c for c in out.columns if c!='cycle_index' and np.issubdtype(out[c].dtype, np.number)]
    return out[keep]

def merge_by_cycle(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    combined = None
    for df in dfs:
        if df is None or df.empty: continue
        if combined is None:
            combined = df.copy()
        else:
            dup = set(combined.columns).intersection(set(df.columns)) - {'cycle_index'}
            if dup:
                df = df.rename(columns={c: f"{c}__{len(combined.columns)}" for c in dup})
            combined = combined.merge(df, on='cycle_index', how='left')
    if combined is None:
        combined = pd.DataFrame(columns=['cycle_index'])
    return combined.sort_values('cycle_index').reset_index(drop=True)

# ---------- 主流程：按文件夹中的“电池编号”匹配并合并 ----------
def build_combined_from_folder():
    os.makedirs(OUT_COMBINED_DIR, exist_ok=True)
    all_csv = list_csv(BASE_DIR)
    # 收集每个电池编号下的文件
    bucket: Dict[str, Dict[str, str]] = {}
    for fp in all_csv:
        # 忽略曲线矩阵
        if fp.endswith(IGNORE_SUFFIX):
            continue
        kind = detect_kind_by_suffix(fp)
        if kind is None or kind == "ignore":
            continue
        bid = extract_battery_id_from_name(fp)
        bucket.setdefault(bid, {})
        # 若某种类型重复，选择更短文件名（更可能是匹配的主文件）
        prev = bucket[bid].get(kind)
        if prev is None or len(os.path.basename(fp)) < len(os.path.basename(prev)):
            bucket[bid][kind] = fp

    print(f"发现 {len(bucket)} 个电池编号。")

    # 逐电池合并
    for bid, files in bucket.items():
        dfs = []
        missing = []
        for kind in ("dqdv","dtdv","dvdt","direct","qstat","tstat","vstat","tseq"):
            fp = files.get(kind)
            if not fp:
                missing.append(kind); continue
            df = load_csv(fp)
            if df is None or df.empty:
                missing.append(kind); continue
            try:
                dfs.append(standardize_columns(df, kind))
            except Exception as e:
                print(f"⚠️ {bid} -> {kind} 标准化失败：{e}")
        combined = merge_by_cycle(dfs)

        # 可选：去掉全空/常数列（无信息）
        num_cols = [c for c in combined.columns if c!='cycle_index' and np.issubdtype(combined[c].dtype, np.number)]
        drop_cols=[]
        for c in num_cols:
            v = combined[c].to_numpy()
            if np.all(~np.isfinite(v)) or (np.nanmax(v)==np.nanmin(v)):
                drop_cols.append(c)
        if drop_cols:
            combined = combined.drop(columns=drop_cols)

        out_path = os.path.join(OUT_COMBINED_DIR, f"{bid}_combined_features.csv")
        combined.to_csv(out_path, index=False, encoding='utf-8-sig', float_format='%.10g')
        print(f"✅ {bid} 合并完成 -> {out_path}（缺失: {','.join(missing) if missing else '无'}；特征数={len(combined.columns)-1}）")

# ---------- 全局 Min–Max ----------
def fit_minmax_and_save(src_dir: str, scaler_path: str):
    mins, maxs = {}, {}
    for fp in list_csv(src_dir):
        df = load_csv(fp);
        if df is None: continue
        for c in df.columns:
            if c=='cycle_index' or not np.issubdtype(df[c].dtype, np.number): continue
            s = df[c].dropna()
            if s.empty: continue
            mn, mx = float(s.min()), float(s.max())
            mins[c] = mn if c not in mins else min(mins[c], mn)
            maxs[c] = mx if c not in maxs else max(maxs[c], mx)
    scaler={}
    for c in mins:
        a,b = mins[c], maxs[c]
        if not np.isfinite(a) or not np.isfinite(b) or a==b:
            scaler[c] = {"min": float(a if np.isfinite(a) else 0.0), "max": float((a if np.isfinite(a) else 0.0)+1.0)}
        else:
            scaler[c] = {"min": float(a), "max": float(b)}
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path,"w",encoding="utf-8") as f:
        json.dump(scaler,f,ensure_ascii=False,indent=2)
    print(f"✅ 保存全局 Min–Max：{scaler_path}（{len(scaler)} 列）")

def apply_minmax_dir(src_dir: str, dst_dir: str, scaler_path: str):
    with open(scaler_path,"r",encoding="utf-8") as f:
        scaler = json.load(f)
    os.makedirs(dst_dir, exist_ok=True)
    for fp in list_csv(src_dir):
        df = load_csv(fp)
        if df is None: continue
        out = df.copy()
        for c in out.columns:
            if c=='cycle_index' or not np.issubdtype(out[c].dtype, np.number): continue
            if c not in scaler: continue
            a,b = scaler[c]["min"], scaler[c]["max"]
            denom = (b-a) if (b-a)!=0 else 1.0
            out[c] = (out[c]-a)/denom
        out_path = os.path.join(dst_dir, os.path.basename(fp).replace(".csv","_norm.csv"))
        out.to_csv(out_path, index=False, encoding='utf-8-sig', float_format='%.10g')
        print(f"✅ 归一化：{out_path}")

# ---------- 入口 ----------
if __name__ == "__main__":
    # 1) 按“文件夹里的电池编号”匹配合并
    build_combined_from_folder()
    # 2) 拟合全局 Min–Max（用全部合并文件）
    fit_minmax_and_save(OUT_COMBINED_DIR, SCALER_JSON)
    # 3) 应用到每个合并文件
    apply_minmax_dir(OUT_COMBINED_DIR, OUT_NORM_DIR, SCALER_JSON)
