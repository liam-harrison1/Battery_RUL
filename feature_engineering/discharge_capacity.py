import os
import pandas as pd
from glob import glob

# === 配置你的数据目录 ===
input_dir = r"C:\Users\13512\Desktop\MIT数据\raw_data"   # ← 改成放 CSV 的文件夹
output_csv = os.path.join(input_dir, r"C:\Users\13512\Desktop\discharge_capacity")

def extract_discharge_capacity_per_cycle(path):
    # 只读必要列（若存在）
    sample = pd.read_csv(path, nrows=1)
    cols = [c for c in ["discharge_capacity", "cycle_index", "step_type", "barcode", "file_name"]
            if c in sample.columns]
    df = pd.read_csv(path, usecols=cols if cols else None)

    # 元信息（没有就用文件名代替）
    barcode   = df["barcode"].iloc[0]   if "barcode"   in df.columns else os.path.splitext(os.path.basename(path))[0]
    file_name = df["file_name"].iloc[0] if "file_name" in df.columns else os.path.basename(path)

    # 仅保留放电阶段
    if "step_type" in df.columns:
        df = df[df["step_type"].astype(str).str.lower().str.contains("discharge", na=False)]

    # 必需列检查
    if "cycle_index" not in df.columns or "discharge_capacity" not in df.columns:
        raise ValueError(f"缺少必要列：{path} 需要包含 'cycle_index' 与 'discharge_capacity'。")

    # 按循环取当轮放电容量（最大值）
    per_cycle = (df.groupby("cycle_index", as_index=False)["discharge_capacity"]
                   .max()
                   .rename(columns={"discharge_capacity": "discharge_capacity_Ah"}))
    per_cycle.insert(0, "battery_id", barcode)
    per_cycle.insert(1, "source_file", file_name)
    return per_cycle

# 遍历目录内所有 CSV
csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
all_tables, errors = [], []

for p in csv_files:
    try:
        all_tables.append(extract_discharge_capacity_per_cycle(p))
    except Exception as e:
        errors.append((p, str(e)))

# 汇总保存
if all_tables:
    summary = pd.concat(all_tables, ignore_index=True)
    summary.to_csv(output_csv, index=False)
    print(f"OK: 共处理 {len(all_tables)}/{len(csv_files)} 个文件")
    print(f"已输出: {output_csv}")
else:
    print("未生成结果，请检查目录/文件格式。")

# 如需查看异常文件：
if errors:
    err_path = os.path.join(input_dir, "discharge_capacity_errors.csv")
    pd.DataFrame(errors, columns=["file", "error"]).to_csv(err_path, index=False)
    print(f"错误日志: {err_path}")
