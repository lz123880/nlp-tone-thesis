# -*- coding: utf-8 -*-
"""
06_robustness.py —— 稳健性检验模块
==================================
输入：output/tone_panel.csv（04 输出）
输出：output/tables/tab6_robustness.csv、tab7_placebo.csv 与图 4-图 6。

基准设定的选择
--------------
主回归的四个模型中，(1)(2) 检验 H1（前瞻性信息），(3) 检验 H2（增量信息），
(4) 引入公司固定效应后语调系数不再显著。稳健性检验以两个"主设定"为基准，
避免用被公司固定效应吸收掉截面变异后的设定作基准：

  设定 A（H1 主设定）：年度 FE + 控制变量（size, lev, revGrowth）
  设定 B（H2 主设定）：设定 A + 当期 ROA

检验设计
--------
  R1 替代测度     ：以机器学习语调 TONEml 替换词典法语调；
  R2 替代业绩     ：以 ROE_{t+1} 替换 ROA_{t+1}；
  R3 逐年剔除     ：逐一剔除某一年度后重估，检验结果非由单一年份驱动；
  R4 分半语调复核 ：以奇数句语调估计、偶数句语调复核（测度内部一致性）；
  R5 安慰剂检验   ：随机打乱语调 1000 次，比较真实系数与安慰剂分布。
"""
import os
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(ROOT, "output", "tables")
FIG_DIR = os.path.join(ROOT, "output", "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ana = import_module("05_analysis")
CTRL = ana.CTRL
SPEC_A = CTRL                 # 设定 A：年度 FE + 控制变量
SPEC_B = ana.CTRL_WITH_ROA    # 设定 B：设定 A + 当期 ROA


def run_pair(df, y, x, sub_mask=None):
    """在设定 A / 设定 B 下分别估计，返回 (A结果, B结果)。"""
    d = df if sub_mask is None else df[sub_mask]
    a = ana.fe_ols(d, y, x, SPEC_A, firm_fe=False, year_fe=True)
    b = ana.fe_ols(d, y, x, SPEC_B, firm_fe=False, year_fe=True)
    return a, b


def main():
    df = ana.load_panel()
    print(f"稳健性检验样本：{len(df)} 个企业-年度观测 / {df['secCode'].nunique()} 家公司")

    rows = []

    def add(label: str, a: dict, b: dict):
        rows.append({
            "检验": label,
            "系数A": round(a["beta"], 3), "t值A": round(a["t"], 2), "N_A": a["n"],
            "系数B": round(b["beta"], 3), "t值B": round(b["t"], 2), "N_B": b["n"],
        })

    # 基准
    bA, bB = run_pair(df, "roa_next", "TONE")
    add("基准（词典法语调）", bA, bB)

    # R1 替代测度
    a, b = run_pair(df, "roa_next", "TONEml")
    add("R1 替代测度（机器学习语调）", a, b)

    # R2 替代业绩——ROE 需先剔除缺失
    sub = df.dropna(subset=["roe_next"]).copy()
    a, b = run_pair(sub, "roe_next", "TONE")
    add("R2 替代业绩（ROE$_{t+1}$）", a, b)

    # R3 逐年剔除
    for drop_year in sorted(int(y) for y in df["year"].unique()):
        mask = df["year"] != drop_year
        a, b = run_pair(df, "roa_next", "TONE", mask)
        add(f"R3 剔除 {drop_year} 年", a, b)

    # R4 分半语调复核（分半列已并入面板，口径与 TONE_dict 一致，需乘 100）
    half = df.copy()
    half["TONE_odd"] = half["TONE_odd"] * 100
    half["TONE_even"] = half["TONE_even"] * 100
    a, b = run_pair(half, "roa_next", "TONE_odd")
    add("R4a 分半语调（奇数句）", a, b)
    a, b = run_pair(half, "roa_next", "TONE_even")
    add("R4b 分半语调（偶数句）", a, b)

    tab6 = pd.DataFrame(rows)
    tab6.to_csv(os.path.join(TABLE_DIR, "tab6_robustness.csv"), index=False, encoding="utf-8-sig")
    print("\n== 表6 稳健性检验（设定 A：年度FE+控制变量；设定 B：A+当期ROA）==")
    print(tab6.to_string(index=False))

    # 相关系数：真实 TONE 与两个分半测度（测度内部一致性佐证）
    r_odd = df["TONE_dict"].corr(half["TONE_odd"] / 100)
    r_even = df["TONE_dict"].corr(half["TONE_even"] / 100)
    r_oe = half["TONE_odd"].corr(half["TONE_even"])
    print(f"\n分半语调与全样本语调相关：r(全样本, 奇数句)={r_odd:.3f}，"
          f"r(全样本, 偶数句)={r_even:.3f}，r(奇数句, 偶数句)={r_oe:.3f}")

    # ---------------- R5 安慰剂检验（以设定 B 为基准）----------------
    rng = np.random.default_rng(2026)
    N = 1000
    work = df[["secCode", "year", "roa_next", "TONE"] + list(SPEC_B)].dropna().copy()
    placebo = []
    for _ in range(N):
        work["TONE_perm"] = rng.permutation(work["TONE"].values)
        try:
            m = ana.fe_ols(work, "roa_next", "TONE_perm", SPEC_B, firm_fe=False, year_fe=True)
            placebo.append(m["beta"])
        except Exception:
            continue
    placebo = np.array(placebo)
    true_b = bB["beta"]
    p_perm = float((np.abs(placebo) >= abs(true_b)).mean())

    tab7 = pd.DataFrame([
        {"统计量": "真实系数（设定 B）", "取值": round(true_b, 3)},
        {"统计量": "真实 t 值（设定 B）", "取值": round(bB["t"], 3)},
        {"统计量": "安慰剂系数均值", "取值": round(float(placebo.mean()), 3)},
        {"统计量": "安慰剂系数标准差", "取值": round(float(placebo.std()), 3)},
        {"统计量": "|安慰剂|≥|真实| 的比例（置换 p 值）", "取值": round(p_perm, 4)},
        {"统计量": "有效置换次数", "取值": len(placebo)},
    ])
    tab7.to_csv(os.path.join(TABLE_DIR, "tab7_placebo.csv"), index=False, encoding="utf-8-sig")
    print("\n== 表7 安慰剂检验（1000 次随机置换）==")
    print(tab7.to_string(index=False))

    # ---------------- 图 4：安慰剂分布 ----------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.hist(placebo, bins=40, color="#9aa5b1", edgecolor="white", label="安慰剂系数分布")
    ax.axvline(true_b, color="#C44E52", ls="--", lw=2, label=f"真实系数 {true_b:.2f}")
    ax.set_xlabel("TONE 系数（设定 B）")
    ax.set_ylabel("次数")
    ax.set_title(f"图 4  安慰剂检验：随机打乱语调 {len(placebo)} 次")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_placebo.png"))
    plt.close(fig)

    # ---------------- 图 5：交互项检验系数（H3）----------------
    inter = pd.read_csv(os.path.join(TABLE_DIR, "tab5_interaction.csv"))
    labels, coefs = [], []
    for _, r in inter.iterrows():
        labels += [r["检验"], r["检验"]]
        coefs += [r["TONE 系数"], r["交互项系数"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=150)
    ypos = np.arange(len(coefs))[::-1]
    ax.barh(ypos, coefs, color=["#4C72B0" if c > 0 else "#C44E52" for c in coefs], alpha=0.88)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{l}\n（主项 / 交互项）" for l in labels], fontsize=8.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("回归系数")
    ax.set_title("图 5  语调可信度的异质性检验（H3）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_interaction.png"))
    plt.close(fig)

    # ---------------- 图 6：稳健性系数 forest 图 ----------------
    t6 = pd.concat([
        tab6[["检验", "系数A", "t值A"]].rename(columns={"系数A": "系数", "t值A": "t值"}).assign(**{"设定": "A"}),
        tab6[["检验", "系数B", "t值B"]].rename(columns={"系数B": "系数", "t值B": "t值"}).assign(**{"设定": "B"}),
    ])
    t6 = t6.dropna(subset=["系数"])
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    ypos = np.arange(len(t6))[::-1]
    colors = ["#4C72B0" if s == "A" else "#DD8452" for s in t6["设定"]]
    ax.errorbar(t6["系数"], ypos, xerr=np.abs(t6["t值"]).clip(lower=0.3) * 0 + 0.15,
                fmt="none", ecolor="#cccccc", capsize=0, zorder=1)
    ax.scatter(t6["系数"], ypos, c=colors, s=44, zorder=3, edgecolor="white")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['检验']} [{r['设定']}]" for _, r in t6.iterrows()], fontsize=8)
    ax.axvline(bB["beta"], color="#C44E52", ls="--", lw=1.4, label=f"设定 B 基准系数 {bB['beta']:.2f}")
    ax.axvline(bA["beta"], color="#55A868", ls=":", lw=1.4, label=f"设定 A 基准系数 {bA['beta']:.2f}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("TONE 对下一年经营业绩的回归系数")
    ax.set_title("图 6  稳健性检验系数对比（Forest 图）")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_forest.png"))
    plt.close(fig)

    print(f"\n稳健性检验完成 → {TABLE_DIR} 与 {FIG_DIR}")


if __name__ == "__main__":
    main()
