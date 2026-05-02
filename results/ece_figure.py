import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="ECE graph")
# parser.add_argument("--data_path", type=str, help="result path") # single file
parser.add_argument("--data_paths", nargs="+", type=str, help="result paths") # multi file
parser.add_argument("--output_path", type=str, default= "./ece_graph_ood.png", help="output paths")
parser.add_argument("--title", type=str, default= "RLAR (OOD)", help="title name")
args = parser.parse_args()

# =========================
# 1. 데이터 로드
# =========================
# single file
# data_path = args.data_path
# output_path = data_path.split(".json")[0] + ".png"
# with open(data_path, "r") as f:
#     data = json.load(f)

# df = pd.DataFrame(data)

# multi files
data_paths = args.data_paths
output_path = args.output_path
dfs = []

for path in data_paths:
    with open(path, "r") as f:
        data = json.load(f)
    dfs.append(pd.DataFrame(data))

df = pd.concat(dfs, ignore_index=True)


# =========================
# 2. confidence 전처리
# =========================
# df["confidence"] = df["confidence"].astype(float) / 100.0 #rlvr

df["confidence"] = (
    df["confidence"]
    .astype(str)          # 혹시 모를 타입 대비
    .str.rstrip('.')      # 끝에 붙은 '.' 제거
    .astype(float)        # float 변환
    / 100.0               # rlvr 기준
)

# df["confidence"] = df["confidence"].astype(float) # otherwise

# =========================
# 3. binning
# =========================
bins = np.arange(0, 1.1, 0.1)
# labels = np.round(bins[:-1] + 0.05, 2)
labels = np.round(bins[:-1], 2)

df["conf_bin"] = pd.cut(
    df["confidence"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# =========================
# 4. count & accuracy
# =========================
count_per_bin = df["conf_bin"].value_counts().sort_index()
print(count_per_bin[0.4])
print(count_per_bin[0.5])
print(count_per_bin[0.6])
acc_per_bin = df.groupby("conf_bin", observed=True)["is_correct"].mean()

count_per_bin = count_per_bin.reindex(labels, fill_value=0)
acc_per_bin = acc_per_bin.reindex(labels)

x = np.arange(len(labels))
# tick 위치 하나 더 추가
xticks = np.arange(len(labels)+1)


# =========================
# 5. Plot
# =========================
fig, ax1 = plt.subplots(figsize=(10, 6))

# 🎨 파스텔 색상
pastel_red = "#FFB3B3"
pastel_blue = "#A7C7E7"

# 📦 count (bar)
ax1.bar(x, count_per_bin.values, color=pastel_red, edgecolor='black', label="Confidence frequency")
ax1.set_ylabel("Count", fontsize=14)
ax1.set_xlabel("Confidence Bin", fontsize=14)
ax1.set_xticks(x)

ax1.set_xticklabels(labels)

# =========================
# accuracy (NaN 제거 후)
# =========================
valid_mask = ~acc_per_bin.isna()
x_valid = x[valid_mask]
acc_valid = acc_per_bin[valid_mask]

# 📈 accuracy (line)
ax2 = ax1.twinx()
ax2.plot(x_valid, acc_valid, marker='o', color=pastel_blue, linewidth=2, label="Accuracy per bin")
ax2.set_ylabel("Accuracy", fontsize=14)

# tick 크기
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)

# ideal line (연한 회색)
ax2.plot(x, labels, linestyle='--', color='gray', alpha=0.5, label="Ideal calibration")

# legend 합치기
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left",
    fontsize=12,
    frameon=True,
    fancybox=True,
    framealpha=0.9
)
plt.title(args.title, fontsize=16)
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()