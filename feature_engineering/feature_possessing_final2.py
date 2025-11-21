# -*- coding: utf-8 -*-
"""
入模前准备 + 每电池独立筛选 + 频次统计（Mean-Deviation + Null-Importance + Spearman）
- 输入：COMBINED_DIR 下的 *_combined_features.csv（每电池/每循环一行）
- 过程：
  1) 仅在训练集上拟合 MinMax；对 train/val/test 全量应用；
  2) 对训练集每个 battery_id 独立执行三步筛选（MD→NI→Spearman），产出每电池特征白名单；
  3) 对所有训练电池的白名单做“出现频次”统计；若出现并列，用各电池 Spearman 的 |rho| 平均值打破；
  4) 选出论文式“全局最终特征” Top-K（默认 16）；
  5) 仅保留这些全局最终特征，导出 tabular_train/val/test.csv；
  6) 输出与论文风格一致的图：
     - FigA：特征出现频次条形图（Top-N，黑白风格，Times New Roman）；
     - FigB：电池×特征 0/1 选择矩阵（灰度热图）。
- 输出：
  OUT_DIR/
    ├── tabular_train.csv / tabular_val.csv / tabular_test.csv
    ├── minmax_scaler.json
    ├── features_selected_per_battery.json
    ├── features_final_global.json
    ├── reports/
    │     ├── {battery_id}_null_importance.csv
    │     └── {battery_id}_spearman.csv
    └── figs/
          ├── feature_frequency_bar.png
          └── selection_matrix_heatmap.png

⚠️ 依赖：pandas, numpy, scipy, lightgbm, matplotlib
LightGBM 4.x 注意：无 verbose_eval，需用 callbacks 控制日志显示。
"""

import os, re, json, random, gc
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
import matplotlib as mpl
mpl.use("Agg")  # 后端无显示环境也能保存图
import matplotlib.pyplot as plt

# ================== 可配置区域 ==================
COMBINED_DIR = r"C:\Users\13512\Desktop\features_mixed"   # 输入目录
OUT_DIR      = r"C:\Users\13512\Desktop\final_dataset_pro"  # 输出目录

SEED         = 42

# 均值偏差初筛阈值（在 MinMax 之后）
MD_THRESHOLD = 1e-8

# Null-importance 参数（建议比旧脚本更严格，贴近论文）
N_REAL_RUNS      = 20     # 真标签跑几次
N_NULL_RUNS      = 20     # 乱序标签跑几次
NUM_BOOST_ROUND  = 600    # 每次 LightGBM 迭代轮数

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
    verbosity=-1,
    num_threads=0  # 0=尽可能多核；如需限核，改成具体数字
)

# Spearman：在每电池内的 NI 通过列里，按 |rho| 排序；可选 Top-K 或分位截断
SPEARMAN_KEEP_TOPK      = 999999      # 若想分位截断，设很大；
SPEARMAN_KEEP_QUANTILE  = 0.50        # 留中位数及以上（配合上面参数使用）

# 论文式“最终全局特征”个数
FINAL_TOP_K = 16

# 图表风格（尽可能贴近论文：黑白、细线、Times New Roman）
FIG_FONT_FAMILY = "Times New Roman"
FIG_DPI        = 300
FIG_W_SINGLE   = 3.4   # 单栏宽（英寸，~8.6cm）
FIG_H_BAR      = 2.2
FIG_H_HEAT     = 3.2

# =================================================

# ---------- 小工具 ----------
def list_csv(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.csv')]

def extract_battery_id_from_combined(fp: str) -> str:
    """从文件名提取 battery_id，例如 B0005_combined_features.csv → B0005"""
    base = os.path.basename(fp)
    m = re.match(r"(.+?)_combined_features\.csv$", base)
    return m.group(1) if m else os.path.splitext(base)[0]

# ---------- 数据读取 ----------
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

# ---------- 划分 ----------
def train_val_test_split_grouped(battery_ids: List[str], ratios=(0.7,0.15,0.15), seed=SEED):
    rng = random.Random(seed)
    ids = sorted(set(battery_ids))
    rng.shuffle(ids)
    n = len(ids); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    train_ids = ids[:n_tr]
    val_ids   = ids[n_tr:n_tr+n_va]
    test_ids  = ids[n_tr+n_va:]
    return train_ids, val_ids, test_ids

# ---------- y 目标 ----------
# 目标列自动/可配置生成，避免缺失导致报错
# 选择一种：
#   TARGET_MODE = 'auto'              -> 自动检测，优先已有 y/SOH/RUL；否则从容量列估算 SOH
#   TARGET_MODE = 'column'            -> 直接使用你指定的列名 TARGET_COLUMN
#   TARGET_MODE = 'soh_from_capacity' -> 从容量列估算 SOH（cap / 每电池首三循环中位容量）
#   TARGET_MODE = 'rul_from_eol'      -> 估算 RUL（需要 EOL 周期列或基于 SOH<阈值推断）
TARGET_MODE = 'auto'
TARGET_COLUMN = 'y'  # 当 TARGET_MODE='column' 时生效
CAPACITY_COL_CANDIDATES = [
    'Capacity','capacity','Q_d','Qd','Q_discharge','discharge_capacity',
    'discharge_cap','Q','Ah','mAh','Capacity (mAh)','Discharge_Capacity'
]
EOL_CYCLE_COL = None      # 如果你有每电池 EOL 周期列，填列名；否则留 None
EOL_SOH_THRESHOLD = 0.8   # 当基于 SOH 推断 EOL 时使用


def _find_first_existing(df: pd.DataFrame, names: list) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _guess_capacity_col(df: pd.DataFrame) -> str | None:
    # 在候选+包含关键字的列里选方差最大的数值列
    cand = [c for c in df.columns if c in CAPACITY_COL_CANDIDATES]
    if not cand:
        # 宽松关键字
        for c in df.columns:
            low = c.lower()
            if any(k in low for k in ['cap','qd','discharge']):
                cand.append(c)
    cand = [c for c in cand if np.issubdtype(df[c].dtype, np.number)]
    if not cand:
        return None
    var = {c: np.nanvar(pd.to_numeric(df[c], errors='coerce')) for c in cand}
    return max(var, key=var.get)


def _soh_from_capacity(df: pd.DataFrame, cap_col: str) -> pd.Series:
    # 每电池首三循环中位容量作为基线
    def _per_batt(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values('cycle_index')
        base = pd.to_numeric(g[cap_col], errors='coerce').head(3).median()
        soh = pd.to_numeric(g[cap_col], errors='coerce') / (base if base and np.isfinite(base) and base!=0 else 1.0)
        return soh.clip(lower=0.0, upper=1.2)
    return df.groupby('battery_id', group_keys=False).apply(_per_batt)


def _rul_from_eol(df: pd.DataFrame, cap_col: str | None) -> pd.Series:
    if EOL_CYCLE_COL and EOL_CYCLE_COL in df.columns:
        eol_map = df.groupby('battery_id')[EOL_CYCLE_COL].max().to_dict()
    else:
        # 通过 SOH 推断 EOL
        if cap_col is None:
            cap_col = _guess_capacity_col(df)
        if cap_col is None:
            raise ValueError("无法根据容量推断 EOL：找不到容量相关列。请设置 EOL_CYCLE_COL 或提供容量列/改用 SOH。")
        soh = _soh_from_capacity(df, cap_col)
        # 每电池第一个低于阈值的循环作为 EOL
        eol_map = {}
        for bid, g in df.assign(_soh=soh).groupby('battery_id'):
            below = g[g['_soh'] < EOL_SOH_THRESHOLD]['cycle_index']
            eol = int(below.iloc[0]) if len(below) else int(g['cycle_index'].max())
            eol_map[bid] = eol
    # RUL = EOL - 当前 cycle_index
    rul = df.apply(lambda r: max(0, int(eol_map.get(r['battery_id'], r['cycle_index'])) - int(r['cycle_index'])), axis=1)
    return pd.to_numeric(rul, errors='coerce')


def compute_target(df: pd.DataFrame) -> pd.Series:
    # 1) 直接使用已有列
    if TARGET_MODE == 'column':
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"目标列 {TARGET_COLUMN} 不存在，请检查 TARGET_COLUMN 或切换 TARGET_MODE")
        return pd.to_numeric(df[TARGET_COLUMN], errors='coerce')

    if TARGET_MODE == 'auto':
        for name in ['y','SOH','soh','RUL','rul']:
            if name in df.columns:
                return pd.to_numeric(df[name], errors='coerce')
        cap_col = _guess_capacity_col(df)
        if cap_col is not None:
            print(f"[info] AUTO: 依据容量列 '{cap_col}' 估算 SOH 作为 y")
            return _soh_from_capacity(df, cap_col)
        raise ValueError("AUTO 模式下：未找到 y/SOH/RUL，且无法识别容量列以估算 SOH。请设置 TARGET_MODE 和相关参数。")

    if TARGET_MODE == 'soh_from_capacity':
        cap_col = _find_first_existing(df, CAPACITY_COL_CANDIDATES) or _guess_capacity_col(df)
        if cap_col is None:
            raise ValueError("soh_from_capacity：找不到容量列（可在 CAPACITY_COL_CANDIDATES 中添加你数据的列名）")
        return _soh_from_capacity(df, cap_col)

    if TARGET_MODE == 'rul_from_eol':
        cap_col = _guess_capacity_col(df)
        return _rul_from_eol(df, cap_col)

    raise ValueError("未知 TARGET_MODE，请使用 'auto' / 'column' / 'soh_from_capacity' / 'rul_from_eol'")

# ---------- 归一化 ----------
def fit_minmax(df_train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str,float]]:
    stats = {}
    for c in feature_cols:
        s = pd.to_numeric(df_train[c], errors='coerce')
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

# ---------- MD 初筛 ----------
def mean_deviation_filter(df_train: pd.DataFrame, feature_cols: List[str], thr: float) -> List[str]:
    keep = []
    for c in feature_cols:
        s = pd.to_numeric(df_train[c], errors='coerce')
        md = float((s - s.mean()).abs().mean())
        if np.isfinite(md) and md > thr:
            keep.append(c)
    return keep

# ---------- Null-Importance ----------
def lgb_feature_importance(train_X: pd.DataFrame, train_y: np.ndarray, params: dict, seed: int) -> pd.Series:
    dtrain = lgb.Dataset(train_X, label=train_y, free_raw_data=True)
    local = params.copy(); local["seed"] = seed
    model = lgb.train(
        params=local,
        train_set=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=0)]  # 静默
    )
    imp = pd.Series(model.feature_importance(importance_type='gain'), index=train_X.columns)
    return imp

def null_importance_selection(train_X: pd.DataFrame, train_y: np.ndarray, params: dict,
                              n_real: int, n_null: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    real_imps = []
    for i in range(n_real):
        imp = lgb_feature_importance(train_X, train_y, params, seed + i)
        real_imps.append(imp)
    real = pd.concat(real_imps, axis=1).fillna(0.0)
    real_mean = real.mean(axis=1)

    null_imps = []
    for j in range(n_null):
        y_perm = train_y.copy()
        rng.shuffle(y_perm)
        imp = lgb_feature_importance(train_X, y_perm, params, seed + 100 + j)
        null_imps.append(imp)
    null = pd.concat(null_imps, axis=1).fillna(0.0)
    null_mean = null.mean(axis=1)

    # 统计量：imp_ratio + 近似 p-value
    eps = 1e-12
    imp_ratio = (real_mean + eps) / (null_mean + eps)
    # p_value ≈ null > real 的频率
    p_counts = (null.values > real_mean.values.reshape(-1,1)).sum(axis=1)
    p_value = (p_counts + 1.0) / (null.shape[1] + 1.0)

    out = pd.DataFrame({
        "feature": real_mean.index,
        "real_mean": real_mean.values,
        "null_mean": null_mean.reindex(real_mean.index).values,
        "imp_ratio": imp_ratio.values,
        "p_value": p_value.values
    }).sort_values("imp_ratio", ascending=False).reset_index(drop=True)
    return out

# ---------- Spearman ----------
def spearman_ranking(train_df: pd.DataFrame, feature_cols: List[str], y_col: str) -> pd.DataFrame:
    rhos = []
    y = train_df[y_col].values
    for c in feature_cols:
        x = pd.to_numeric(train_df[c], errors='coerce').values
        if np.all(~np.isfinite(x)) or np.nanstd(x) == 0:
            rho = 0.0
        else:
            rho, _ = spearmanr(x, y, nan_policy='omit')
            if not np.isfinite(rho):
                rho = 0.0
        rhos.append((c, abs(float(rho))))
    out = pd.DataFrame(rhos, columns=["feature","abs_spearman"]) \
            .sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    return out

# ---------- 每电池筛选主函数 ----------
def select_features_for_one_battery(df_batt_n: pd.DataFrame, feature_cols_numeric: List[str], y_col: str = "y") -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    # 1) MD 初筛（该电池归一化数据）
    md_keep = mean_deviation_filter(df_batt_n, feature_cols_numeric, MD_THRESHOLD)

    # 2) Null-importance（该电池自己的样本与标签）——严格阈值贴近论文
    train_X = df_batt_n[md_keep].copy()
    train_y = pd.to_numeric(df_batt_n[y_col], errors='coerce').values.astype(float)
    if train_X.shape[1] == 0:
        return [], pd.DataFrame(columns=["feature","imp_ratio","p_value"]), pd.DataFrame(columns=["feature","abs_spearman"])

    ni_table = null_importance_selection(train_X, train_y, LGB_PARAMS,
                                         N_REAL_RUNS, N_NULL_RUNS, SEED)
    ni_keep = ni_table[(ni_table['imp_ratio'] > 1.0) & (ni_table['p_value'] <= 0.2)]['feature'].tolist()
    if not ni_keep:
        # 兜底：至少给出若干候选，保证后续 Spearman 可执行
        ni_keep = ni_table.head(min(10, len(ni_table)))['feature'].tolist()

    # 3) Spearman 排序/截断
    sp_table = spearman_ranking(df_batt_n, ni_keep, y_col)
    if len(sp_table):
        if SPEARMAN_KEEP_TOPK < len(sp_table):
            sp_keep = sp_table.head(SPEARMAN_KEEP_TOPK)['feature'].tolist()
        else:
            q = SPEARMAN_KEEP_QUANTILE
            cut = sp_table['abs_spearman'].quantile(q)
            sp_keep = sp_table[sp_table['abs_spearman'] >= cut]['feature'].tolist()
    else:
        sp_keep = ni_keep

    return sp_keep, ni_table, sp_table

# ---------- 图表风格 ----------
def setup_paper_style():
    mpl.rcParams.update({
        'font.family': FIG_FONT_FAMILY,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'savefig.dpi': FIG_DPI,
        'pdf.fonttype': 42,   # TrueType，避免嵌入 Type3
        'ps.fonttype': 42
    })

# ---------- 画图 ----------
def plot_feature_frequency(freq_series: pd.Series, out_path: str, top_n: int = 30):
    setup_paper_style()
    s = freq_series.sort_values(ascending=False).head(top_n)
    fig = plt.figure(figsize=(FIG_W_SINGLE, FIG_H_BAR))
    ax = fig.add_subplot(111)
    bars = ax.bar(range(len(s)), s.values, color='white', edgecolor='black', linewidth=0.8)
    # 斜体/竖排标签：旋转 60°，紧凑排版
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=60, ha='right')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature selection frequency (train batteries)')
    # 上边框去掉颜色以简洁
    for spine in ['top']:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_selection_matrix(binary_df: pd.DataFrame, out_path: str):
    setup_paper_style()
    fig = plt.figure(figsize=(FIG_W_SINGLE, FIG_H_HEAT))
    ax = fig.add_subplot(111)
    # 灰度热图：1=黑，0=白
    mat = binary_df.values.astype(float)
    im = ax.imshow(mat, cmap='Greys', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_yticks(range(binary_df.shape[0]))
    ax.set_yticklabels(binary_df.index)
    ax.set_xticks(range(binary_df.shape[1]))
    ax.set_xticklabels(binary_df.columns, rotation=60, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Batteries (train)')
    ax.set_title('Per-battery selected features (binary)')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ---------- 主流程 ----------
def main():
    random.seed(SEED); np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'figs'), exist_ok=True)

    # 1) 读入合并特征
    all_df = load_all_combined(COMBINED_DIR)

    # 2) 目标列（若已有 y 列直接使用；否则在 compute_target 中自定义）
    all_df = all_df.copy()
    all_df['y'] = compute_target(all_df)

    # 3) 电池分层划分
    battery_ids = all_df['battery_id'].unique().tolist()
    train_ids, val_ids, test_ids = train_val_test_split_grouped(battery_ids)

    df_train = all_df[all_df['battery_id'].isin(train_ids)].reset_index(drop=True)
    df_val   = all_df[all_df['battery_id'].isin(val_ids)].reset_index(drop=True)
    df_test  = all_df[all_df['battery_id'].isin(test_ids)].reset_index(drop=True)

    feature_cols_numeric = [c for c in all_df.columns if c not in ['battery_id','cycle_index','y']]

    # 4) 仅用训练集拟合 MinMax，并应用到全量
    scaler = fit_minmax(df_train, feature_cols_numeric)
    with open(os.path.join(OUT_DIR, "minmax_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    df_train_n = apply_minmax(df_train, feature_cols_numeric, scaler)
    df_val_n   = apply_minmax(df_val,   feature_cols_numeric, scaler)
    df_test_n  = apply_minmax(df_test,  feature_cols_numeric, scaler)

    # 5) 每电池独立筛选（仅训练集）
    per_batt_selected: Dict[str, List[str]] = {}
    per_batt_spearman: Dict[str, pd.DataFrame] = {}

    for bid, g_train in df_train_n.groupby('battery_id'):
        feats, ni_tab, sp_tab = select_features_for_one_battery(
            g_train, feature_cols_numeric, y_col='y')
        per_batt_selected[bid] = feats
        per_batt_spearman[bid] = sp_tab
        # 保存报告
        ni_tab.to_csv(os.path.join(OUT_DIR, 'reports', f'{bid}_null_importance.csv'), index=False, encoding='utf-8-sig')
        sp_tab.to_csv(os.path.join(OUT_DIR, 'reports', f'{bid}_spearman.csv'), index=False, encoding='utf-8-sig')

    with open(os.path.join(OUT_DIR, 'features_selected_per_battery.json'), 'w', encoding='utf-8') as f:
        json.dump(per_batt_selected, f, ensure_ascii=False, indent=2)

    # 6) 频次统计 + 打破并列（全局 |rho| 的平均）
    # 频次
    freq = {}
    for bid, feats in per_batt_selected.items():
        for c in feats:
            freq[c] = freq.get(c, 0) + 1
    freq_ser = pd.Series(freq).sort_values(ascending=False)

    # 打破并列：各电池 Spearman 的 |rho| 平均值
    rho_mean = {}
    for c in freq_ser.index:
        vals = []
        for bid, sp_tab in per_batt_spearman.items():
            row = sp_tab[sp_tab['feature'] == c]
            if not row.empty:
                vals.append(float(row['abs_spearman'].iloc[0]))
        rho_mean[c] = float(np.mean(vals)) if len(vals) else 0.0
    rho_ser = pd.Series(rho_mean)

    # 最终排序：先按频次降序，再按 rho_mean 降序
    order_df = pd.DataFrame({'feature': freq_ser.index, 'freq': freq_ser.values})
    order_df['rho_mean'] = order_df['feature'].map(rho_ser)
    order_df = order_df.sort_values(['freq','rho_mean'], ascending=[False, False]).reset_index(drop=True)

    final_features = order_df['feature'].head(FINAL_TOP_K).tolist()
    with open(os.path.join(OUT_DIR, 'features_final_global.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'final_features': final_features,
            'frequency_table': order_df.to_dict(orient='list')
        }, f, ensure_ascii=False, indent=2)

    print(f"📌 最终全局特征 Top-{FINAL_TOP_K}: {final_features}")

    # 7) 导出统一列的 tabular_*.csv（论文式：只保留最终全局特征）
    def export(df_split_n: pd.DataFrame, name: str):
        cols = ['battery_id','cycle_index'] + final_features + ['y']
        # 对可能缺失的列补齐（没有的置为 NaN→0.0）
        tmp = df_split_n.copy()
        for c in final_features:
            if c not in tmp.columns:
                tmp[c] = np.nan
        out = tmp[cols].copy()
        out[final_features] = out[final_features].fillna(0.0)
        out.to_csv(os.path.join(OUT_DIR, f'tabular_{name}.csv'), index=False, encoding='utf-8-sig', float_format='%.10g')
        print(f"✅ 导出 {name}: {out.shape} -> {os.path.join(OUT_DIR, f'tabular_{name}.csv')}")

    export(df_train_n, 'train')
    export(df_val_n,   'val')
    export(df_test_n,  'test')

    # 8) 作图（论文风格）
    plot_feature_frequency(freq_ser, os.path.join(OUT_DIR, 'figs', 'feature_frequency_bar.png'), top_n=min(30, len(freq_ser)))

    # 选择矩阵（仅用最终特征）
    # 行：训练电池，列：final_features
    mat = []
    idx = []
    for bid in sorted(per_batt_selected.keys()):
        idx.append(bid)
        s = pd.Series(0, index=final_features, dtype=int)
        for c in per_batt_selected[bid]:
            if c in s.index:
                s[c] = 1
        mat.append(s)
    if len(mat):
        binary_df = pd.DataFrame(mat, index=idx, columns=final_features)
        plot_selection_matrix(binary_df, os.path.join(OUT_DIR, 'figs', 'selection_matrix_heatmap.png'))

    print("🎯 完成：每电池筛选 + 频次统计 + 论文风格图 与 数据集导出。")


if __name__ == "__main__":
    main()
