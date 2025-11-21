# -*- coding: utf-8 -*-
"""
入模前准备 + 两级筛选（Mean-Deviation + Null-Importance + Spearman）
- 输入：一个文件夹下的 *_combined_features.csv（每电池/每循环一行）
- 输出：tabular_train/val/test.csv（含 battery_id, cycle_index, 选中特征, y）
      以及：minmax_scaler.json、features_selected.json、
            null_importance_report.csv、spearman_report.csv

✅ 本版修复：
- LightGBM 4.x 移除 verbose_eval：改用 callbacks=[lgb.log_evaluation(period=0)]
- 参数 verbose → verbosity
"""

import os, re, json, random, gc
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import spearmanr
import lightgbm as lgb

# ========== 配置 ==========
COMBINED_DIR = r"C:\Users\13512\Desktop\features_mixed"  # 改成你的 *_combined_features.csv 所在目录
OUT_DIR      = r"C:\Users\13512\Desktop\final_dataset_10features"           # 输出目录
TARGET_MODE  = "rul"    # 'rul' 或 'capacity_ratio'
SPLIT_RATIO  = (0.7, 0.15, 0.15)  # Train/Val/Test 按“电池”划分
SEED         = 42

# 均值偏差初筛阈值（在 MinMax 之后）
MD_THRESHOLD = 1e-8

# Null-importance 参数（可先小一点加速试跑）
N_REAL_RUNS  = 20     # 真标签跑几次
N_NULL_RUNS  = 20     # 乱序标签跑几次
NUM_BOOST_ROUND = 600 # 每次 LightGBM 迭代轮数

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=1.0,
    max_depth=-1,
    verbosity=-1,        # ✅ 4.x 用 verbosity
    num_threads=0        # 0=用尽可能多的核；如需限核可改成具体数字
)

# Spearman：保留 |rho| 排名前 q 分位以上
SPEARMAN_KEEP_QUANTILE = 0


# ========== 工具函数 ==========
def list_csv(dirp: str) -> List[str]:
    return [os.path.join(dirp, f) for f in os.listdir(dirp) if f.lower().endswith(".csv")]

def extract_battery_id_from_combined(path: str) -> str:
    # e.g. EL150800460436_combined_features.csv -> EL150800460436
    return os.path.basename(path).replace("_combined_features.csv", "")

def load_all_combined(combined_dir: str) -> pd.DataFrame:
    rows = []
    for fp in list_csv(combined_dir):
        if not fp.endswith("_combined_features.csv"):
            continue
        bid = extract_battery_id_from_combined(fp)
        df = pd.read_csv(fp)
        if 'cycle_index' not in df.columns:
            raise ValueError(f"{fp} 缺少 cycle_index 列")
        df.insert(0, 'battery_id', bid)
        rows.append(df)
    if not rows:
        raise RuntimeError("未在 COMBINED_DIR 中找到 *_combined_features.csv")
    all_df = pd.concat(rows, axis=0, ignore_index=True)
    # 只保留数值列 + 主键
    keep = ['battery_id', 'cycle_index'] + [
        c for c in all_df.columns
        if c not in ('battery_id', 'cycle_index') and np.issubdtype(all_df[c].dtype, np.number)
    ]
    all_df = all_df[keep].sort_values(['battery_id','cycle_index']).reset_index(drop=True)
    return all_df

def train_val_test_split_grouped(battery_ids: List[str], ratios=(0.7,0.15,0.15), seed=42):
    rng = random.Random(seed)
    ids = sorted(battery_ids)
    rng.shuffle(ids)
    n = len(ids); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    train_ids = ids[:n_tr]
    val_ids   = ids[n_tr:n_tr+n_va]
    test_ids  = ids[n_tr+n_va:]
    return train_ids, val_ids, test_ids

def compute_target(df_one_batt: pd.DataFrame, mode: str) -> pd.Series:
    if mode == 'rul':
        m = int(df_one_batt['cycle_index'].max())
        return m - df_one_batt['cycle_index']
    elif mode == 'capacity_ratio':
        cand = [c for c in df_one_batt.columns if c.lower() in ('qd_ah','direct_qd_ah')]
        if not cand:
            raise ValueError("未找到 qd_ah/direct_qd_ah 列，无法计算 capacity_ratio。可改 TARGET_MODE='rul'")
        q = pd.to_numeric(df_one_batt[cand[0]], errors='coerce')
        q0 = float(q.iloc[0]) if np.isfinite(q.iloc[0]) else float(np.nanmedian(q))
        if not np.isfinite(q0) or q0 == 0:
            raise ValueError("初始容量无效，无法计算 capacity_ratio")
        return q / q0
    else:
        raise ValueError("TARGET_MODE 仅支持 'rul' 或 'capacity_ratio'")

def fit_minmax(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str,float]]:
    stats = {}
    for c in feature_cols:
        s = pd.to_numeric(train_df[c], errors='coerce')
        mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            stats[c] = {"min": float(mn if np.isfinite(mn) else 0.0),
                        "max": float((mn if np.isfinite(mn) else 0.0) + 1.0)}
        else:
            stats[c] = {"min": mn, "max": mx}
    return stats

def apply_minmax(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, Dict[str,float]]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        a, b = stats[c]["min"], stats[c]["max"]
        denom = (b-a) if (b-a) != 0 else 1.0
        out[c] = (pd.to_numeric(out[c], errors='coerce') - a) / denom
    return out

def mean_deviation_filter(df_train: pd.DataFrame, feature_cols: List[str], thr: float) -> List[str]:
    keep = []
    for c in feature_cols:
        s = pd.to_numeric(df_train[c], errors='coerce')
        md = float((s - s.mean()).abs().mean())
        if np.isfinite(md) and md > thr:
            keep.append(c)
    return keep

# ====== LightGBM 4.x：无 verbose_eval，用 callbacks 控制日志 ======
def lgb_feature_importance(train_X: pd.DataFrame, train_y: np.ndarray, params: dict, seed: int) -> pd.Series:
    dtrain = lgb.Dataset(train_X, label=train_y, free_raw_data=True)
    local = params.copy(); local["seed"] = seed
    model = lgb.train(
        local,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(period=0)]  # 关闭训练日志
    )
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=train_X.columns)
    total = imp.sum()
    return (imp/total) if total > 0 else imp

def null_importance_selection(train_X: pd.DataFrame, train_y: np.ndarray,
                              params: dict, n_real: int, n_null: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # 真标签多次
    real_imps = []
    for i in range(n_real):
        imp = lgb_feature_importance(train_X, train_y, params, seed + i)
        real_imps.append(imp)
    real_df = pd.concat(real_imps, axis=1).fillna(0.0)
    real_mean = real_df.mean(axis=1)

    # 乱序标签多次
    null_imps = []
    for j in range(n_null):
        y_shuf = train_y.copy(); rng.shuffle(y_shuf)
        imp = lgb_feature_importance(train_X, y_shuf, params, seed + 1000 + j)
        null_imps.append(imp)
    null_df = pd.concat(null_imps, axis=1).fillna(0.0)
    null_mean = null_df.mean(axis=1)

    eps = 1e-9
    imp_ratio = (real_mean + eps) / (null_mean + eps)
    # p_value：有多少 null 均值 >= 真均值
    p_value = (null_df.T >= real_mean.values).mean(axis=0)

    out = pd.DataFrame({
        "feature": real_mean.index,
        "real_mean": real_mean.values,
        "null_mean": null_mean.reindex(real_mean.index).values,
        "imp_ratio": imp_ratio.values,
        "p_value": p_value.values
    }).sort_values("imp_ratio", ascending=False).reset_index(drop=True)
    return out

def spearman_ranking(train_df: pd.DataFrame, feature_cols: List[str], y_col: str) -> pd.DataFrame:
    rhos = []
    y = train_df[y_col].values
    for c in feature_cols:
        x = pd.to_numeric(train_df[c], errors='coerce').values
        if np.all(~np.isfinite(x)) or np.nanstd(x) == 0:
            rho = 0.0
        else:
            rho, _ = spearmanr(x, y, nan_policy='omit')
            if not np.isfinite(rho): rho = 0.0
        rhos.append((c, abs(float(rho))))
    out = pd.DataFrame(rhos, columns=["feature","abs_spearman"]).sort_values("abs_spearman", ascending=False)
    return out


# ========== 主流程 ==========
def main():
    random.seed(SEED); np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 读取全部电池的合并特征
    all_df = load_all_combined(COMBINED_DIR)
    feature_cols_all = [c for c in all_df.columns if c not in ('battery_id','cycle_index')]
    print(f"载入完成：{len(all_df['battery_id'].unique())} 个电池，{len(all_df)} 条循环；候选特征 {len(feature_cols_all)} 列")

    # 2) 计算标签 y（按电池）
    parts = []
    for bid, g in all_df.groupby('battery_id'):
        g = g.sort_values('cycle_index').reset_index(drop=True).copy()
        g['y'] = compute_target(g, TARGET_MODE)
        parts.append(g)
    all_df = pd.concat(parts, axis=0, ignore_index=True)

    # 3) 按电池分组切分
    train_ids, val_ids, test_ids = train_val_test_split_grouped(sorted(all_df['battery_id'].unique()), SPLIT_RATIO, SEED)
    df_train = all_df[all_df['battery_id'].isin(train_ids)].reset_index(drop=True)
    df_val   = all_df[all_df['battery_id'].isin(val_ids)].reset_index(drop=True)
    df_test  = all_df[all_df['battery_id'].isin(test_ids)].reset_index(drop=True)
    print(f"划分：train={len(train_ids)} 电池, val={len(val_ids)}, test={len(test_ids)}")

    # 4) 仅在训练集拟合 Min–Max 并应用
    feature_cols_numeric = [c for c in feature_cols_all if np.issubdtype(all_df[c].dtype, np.number)]
    scaler = fit_minmax(df_train, feature_cols_numeric)
    with open(os.path.join(OUT_DIR, "minmax_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    df_train_n = apply_minmax(df_train, feature_cols_numeric, scaler)
    df_val_n   = apply_minmax(df_val,   feature_cols_numeric, scaler)
    df_test_n  = apply_minmax(df_test,  feature_cols_numeric, scaler)

    # 5) 均值偏差初筛（训练集）
    md_keep = mean_deviation_filter(df_train_n, feature_cols_numeric, MD_THRESHOLD)
    print(f"MD 初筛：保留 {len(md_keep)}/{len(feature_cols_numeric)} 列")

    # 6) Null-Importance（训练集）
    train_X = df_train_n[md_keep].copy()
    train_y = df_train_n['y'].values.astype(float)
    ni_table = null_importance_selection(train_X, train_y, LGB_PARAMS, N_REAL_RUNS, N_NULL_RUNS, SEED)
    # 规则：imp_ratio > 1 且 p_value <= 0.2
    ni_keep = ni_table[(ni_table['imp_ratio'] > 0.4) & (ni_table['p_value'] <= 0.7)]['feature'].tolist()
    print(f"Null-Importance 通过：{len(ni_keep)} 列")

    # 7) Spearman 验证（训练集）
    sp_table = spearman_ranking(df_train_n, ni_keep, 'y')
    if len(sp_table):
        cut = sp_table['abs_spearman'].quantile(SPEARMAN_KEEP_QUANTILE)
        sp_keep = sp_table[sp_table['abs_spearman'] >= cut]['feature'].tolist()
    else:
        cut = 1.0
        sp_keep = []
    print(f"Spearman 通过：{len(sp_keep)} 列 (阈值={cut:.4f})")

    selected_features = sp_keep
    with open(os.path.join(OUT_DIR, "features_selected.json"), "w", encoding="utf-8") as f:
        json.dump({"selected_features": selected_features}, f, ensure_ascii=False, indent=2)

    # 8) 导出 Tabular（仅保留选中特征）
    def export(df_in: pd.DataFrame, name: str):
        cols = ['battery_id','cycle_index'] + selected_features + ['y']
        out = df_in[cols].copy()
        out.to_csv(os.path.join(OUT_DIR, f"tabular_{name}.csv"),
                   index=False, encoding="utf-8-sig", float_format="%.10g")
        print(f"✅ 导出 {name}: {out.shape} -> {os.path.join(OUT_DIR, f'tabular_{name}.csv')}")

    export(df_train_n, "train")
    export(df_val_n,   "val")
    export(df_test_n,  "test")

    # 9) 保存筛选报告
    ni_table.to_csv(os.path.join(OUT_DIR, "null_importance_report.csv"), index=False, encoding="utf-8-sig")
    sp_table.to_csv(os.path.join(OUT_DIR, "spearman_report.csv"), index=False, encoding="utf-8-sig")

    print("🎯 完成：特征白名单与归一化数据已就绪，可直接入模。")

if __name__ == "__main__":
    main()
