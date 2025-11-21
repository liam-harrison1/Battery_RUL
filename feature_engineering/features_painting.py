# -*- coding: utf-8 -*-
"""
可视化：
A. Mean Deviation（MD）分布 & TopN
B. Null-Importance：imp_ratio vs -log10(p) 散点 + TopN real/null 对比
C. Spearman：分布 & TopN
并导出综合得分榜（imp_ratio * |rho|）

把 BASE 改成你上一步 OUT_DIR 的路径（包含 tabular_train.csv、null_importance_report.csv、spearman_report.csv、features_selected.json）
"""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ==================== 路径与参数 ====================
BASE = r"C:\Users\13512\Desktop\final_dataset_7features"   # ← 改成你的 OUT_DIR
TOPN = 44                              # TopN 可视化个数
MD_THRESHOLD = 1e-6                   # 你筛选时用的阈值
NI_P_CUTOFF = 0.4                     # Null-importance p 值阈值
SP_KEEP_Q = 0.8                       # Spearman 选取的分位阈值（与筛选一致）

paths = {
    "train": os.path.join(BASE, "tabular_train.csv"),
    "ni":    os.path.join(BASE, "null_importance_report.csv"),
    "sp":    os.path.join(BASE, "spearman_report.csv"),
    "sel":   os.path.join(BASE, "features_selected.json"),
}
FIGDIR = os.path.join(BASE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

# ==================== A. Mean Deviation 可视化 ====================
df_train = pd.read_csv(paths["train"])
id_cols = ["battery_id","cycle_index","y"]
feat_cols = [c for c in df_train.columns if c not in id_cols]

# 重新计算 MD（训练集、已归一化）
md = (df_train[feat_cols] - df_train[feat_cols].mean()).abs().mean().sort_values(ascending=False)
md_df = md.reset_index(); md_df.columns = ["feature","mean_deviation"]
md_df.to_csv(os.path.join(FIGDIR, "mean_deviation_table.csv"), index=False, encoding="utf-8-sig")

# 直方图
plt.figure(figsize=(8,5))
plt.hist(md.values, bins=60, edgecolor='k', alpha=0.8)
plt.axvline(MD_THRESHOLD, color='r', linestyle='--', label=f"阈值={MD_THRESHOLD:g}")
plt.xlabel("Mean Deviation"); plt.ylabel("Count"); plt.title("MD 分布（训练集）"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "A1_MD_hist.png"), dpi=200)

# TopN 条形图
top_md = md_df.head(TOPN).iloc[::-1]
plt.figure(figsize=(10, 0.35*len(top_md)+2))
plt.barh(top_md["feature"], top_md["mean_deviation"])
plt.axvline(MD_THRESHOLD, color='r', linestyle='--', lw=1)
plt.title(f"MD Top{TOPN}（训练集）"); plt.xlabel("Mean Deviation")
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "A2_MD_topN.png"), dpi=220)

# ==================== B. Null-Importance 可视化 ====================
ni = pd.read_csv(paths["ni"])
ni["neglogp"] = -np.log10(ni["p_value"].clip(1e-300, 1))
# 火山图：imp_ratio vs -log10(p)
plt.figure(figsize=(8,6))
plt.scatter(ni["imp_ratio"], ni["neglogp"], s=12, alpha=0.6)
plt.axvline(1.0, color='r', linestyle='--', label="imp_ratio=1")
plt.axhline(-np.log10(NI_P_CUTOFF), color='g', linestyle='--', label=f"p={NI_P_CUTOFF:g}")
plt.xlabel("Importance Ratio (real/null)"); plt.ylabel("-log10(p)")
plt.title("Null-Importance 火山图")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "B1_NI_volcano.png"), dpi=220)

# TopN（按 imp_ratio 排序）真实均值 vs 乱序均值
top_ni = ni.sort_values("imp_ratio", ascending=False).head(TOPN)
x = np.arange(len(top_ni))
plt.figure(figsize=(12, 0.35*len(top_ni)+2))
barw = 0.45
plt.barh(top_ni["feature"][::-1], top_ni["real_mean"][::-1], height=barw, label="real_mean")
plt.barh(top_ni["feature"][::-1], top_ni["null_mean"][::-1], height=barw*0.7, alpha=0.7, label="null_mean")
plt.xlabel("Average Importance (gain normalized)")
plt.title(f"Null-Importance Top{TOPN}（real vs null）")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "B2_NI_topN_real_vs_null.png"), dpi=220)

# ==================== C. Spearman 可视化 ====================
sp = pd.read_csv(paths["sp"])
sp_cut = float(sp["abs_spearman"].quantile(SP_KEEP_Q)) if len(sp) else 1.0
# 直方图
plt.figure(figsize=(8,5))
plt.hist(sp["abs_spearman"], bins=50, edgecolor='k', alpha=0.8)
plt.axvline(sp_cut, color='r', linestyle='--', label=f"阈值@{SP_KEEP_Q:.0%}={sp_cut:.3f}")
plt.xlabel("|Spearman ρ|"); plt.ylabel("Count"); plt.title("Spearman 与 y 的相关分布（训练集）")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "C1_Spearman_hist.png"), dpi=220)

# TopN 条形图
top_sp = sp.sort_values("abs_spearman", ascending=False).head(TOPN).iloc[::-1]
plt.figure(figsize=(10, 0.35*len(top_sp)+2))
plt.barh(top_sp["feature"], top_sp["abs_spearman"])
plt.axvline(sp_cut, color='r', linestyle='--', lw=1)
plt.xlabel("|Spearman ρ|"); plt.title(f"Spearman Top{TOPN}")
plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "C2_Spearman_topN.png"), dpi=220)

# ==================== D. 综合排行榜 & 交集/并集 ====================
# 综合得分：imp_ratio * |rho|（可调权重）
ni_s = ni.set_index("feature")
sp_s = sp.set_index("feature")

common = ni_s.index.intersection(sp_s.index)
combo = pd.DataFrame({
    "imp_ratio": ni_s.loc[common, "imp_ratio"],
    "neglogp":  ni_s.loc[common, "neglogp"],
    "abs_spearman": sp_s.loc[common, "abs_spearman"]
})
combo["score"] = combo["imp_ratio"] * combo["abs_spearman"]
combo = combo.sort_values("score", ascending=False)
combo.to_csv(os.path.join(FIGDIR, "D1_combo_rank.csv"), encoding="utf-8-sig")

top_combo = combo.head(TOPN).iloc[::-1]
plt.figure(figsize=(10, 0.35*len(top_combo)+2))
plt.barh(top_combo.index, top_combo["score"])
plt.title(f"综合得分 Top{TOPN}（imp_ratio × |rho|）")
plt.xlabel("Score"); plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "D2_combo_score_topN.png"), dpi=220)

# 入选白名单（筛选完的最终特征）
if os.path.exists(paths["sel"]):
    with open(paths["sel"], "r", encoding="utf-8") as f:
        sel = json.load(f).get("selected_features", [])
    pd.Series(sel, name="selected_feature").to_csv(
        os.path.join(FIGDIR, "E_selected_features.csv"), index=False, encoding="utf-8-sig"
    )

print("✅ 图与表已输出到：", FIGDIR)
