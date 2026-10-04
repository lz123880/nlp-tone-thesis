# -*- coding: utf-8 -*-
"""
05_analysis.py —— 实证分析模块
==============================
输入：output/tone_panel.csv（04 输出）
输出：output/tables/ 下的描述统计、相关矩阵、主回归、分组检验表；
      output/figures/ 下的图 1-图 3。

模型设定（对应论文第 4 章）：
  ROA_{i,t+1} = α + β·TONE_{i,t} + γ'Controls_{i,t} + μ_i + λ_t + ε_{i,t}
  口径说明：语调变量按百分数口径（×100）进入回归，
           系数含义为"净语调每提高 1 个百分点，下一年 ROA 变化多少个百分点"。
  模型 (1) 年度 FE；(2) 公司+年度 FE；(3) (2)+控制变量；
  模型 (4) 在 (3) 基础上加入当期 ROA —— H2（增量信息）的直接检验。
H3 检验：交互项设计。TONE×D_overOptim（过度乐观）与 TONE×D_highLev（高杠杆），
  若交互项系数显著为负，则支持策略性语调假说。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_DIR = os.path.join(ROOT, "output", "tables")
FIG_DIR = os.path.join(ROOT, "output", "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CTRL = ["size", "lev", "revGrowth"]          # 控制变量（不含当期业绩）
CTRL_WITH_ROA = CTRL + ["roa"]               # 增量信息检验用


def load_panel():
    df = pd.read_csv(os.path.join(ROOT, "output", "tone_panel.csv"), dtype={"secCode": str})
    need = ["TONE_dict", "roa_next", "roa", "size", "lev", "revGrowth"]
    df = df.dropna(subset=need).copy()
    # 语调按百分数口径进入回归，便于系数解读
    df["TONE"] = df["TONE_dict"] * 100
    df["TONEml"] = df["TONE_ml"] * 100
    return df


def fe_ols(df: pd.DataFrame, y: str, x: str, controls: list[str],
           firm_fe=True, year_fe=True, cluster="secCode") -> dict:
    """双向固定效应 OLS（虚拟变量法），公司层面聚类稳健标准误。"""
    cols = [x] + list(controls)
    d = df[["secCode", "year", y] + cols].dropna().copy()
    X = d[cols].astype(float)
    if firm_fe:
        X = pd.concat([X, pd.get_dummies(d["secCode"], prefix="f", drop_first=True).astype(float)], axis=1)
    if year_fe:
        X = pd.concat([X, pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(d[y].astype(float), X)
    res = m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return {"n": int(res.nobs), "r2": res.rsquared, "r2a": res.rsquared_adj,
            "beta": res.params[x], "t": res.tvalues[x], "p": res.pvalues[x],
            "se": res.bse[x], "res": res}


def stars(p: float) -> str:
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def main():
    df = load_panel()
    print(f"回归样本：{len(df)} 个企业-年度观测 / {df['secCode'].nunique()} 家公司 / "
          f"{sorted(int(y) for y in df['year'].unique())}")

    # ---------- 表 2 描述性统计 ----------
    desc_vars = ["TONE", "TONEml", "TONE_sd", "PosRatio", "NegRatio",
                 "roa_next", "roa", "size", "lev", "revGrowth"]
    desc = df[desc_vars].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.columns = ["样本量", "均值", "标准差", "最小值", "P25", "中位数", "P75", "最大值"]
    desc.round(4).to_csv(os.path.join(TABLE_DIR, "tab2_descriptive.csv"), encoding="utf-8-sig")
    print("\n== 表2 描述性统计 ==")
    print(desc.round(3).to_string())

    # ---------- 表 3 相关系数矩阵 ----------
    corr_vars = ["TONE", "TONEml", "TONE_sd", "roa_next", "roa", "lev", "revGrowth"]
    corr = df[corr_vars].corr()
    corr.round(3).to_csv(os.path.join(TABLE_DIR, "tab3_corr.csv"), encoding="utf-8-sig")
    print("\n== 表3 相关系数矩阵 ==")
    print(corr.round(3).to_string())

    # ---------- 表 4 主回归 ----------
    models = [
        ("(1) 年度FE", fe_ols(df, "roa_next", "TONE", [], firm_fe=False, year_fe=True)),
        ("(2) (1)+控制变量", fe_ols(df, "roa_next", "TONE", CTRL, firm_fe=False, year_fe=True)),
        ("(3) (2)+当期ROA（H2）", fe_ols(df, "roa_next", "TONE", CTRL_WITH_ROA, firm_fe=False, year_fe=True)),
        ("(4) 公司+年度FE+控制变量", fe_ols(df, "roa_next", "TONE", CTRL)),
    ]
    tab = pd.DataFrame([
        {"模型": name, "TONE 系数": round(m["beta"], 3), "聚类稳健SE": round(m["se"], 3),
         "t 值": round(m["t"], 2), "p 值": round(m["p"], 4), "显著性": stars(m["p"]),
         "N": m["n"], "R2": round(m["r2"], 3), "调整R2": round(m["r2a"], 3)}
        for name, m in models
    ])
    tab.to_csv(os.path.join(TABLE_DIR, "tab4_main_reg.csv"), index=False, encoding="utf-8-sig")
    print("\n== 表4 主回归（被解释变量：ROA_{t+1}，语调为百分数口径）==")
    print(tab.to_string(index=False))

    # ---------- 表 5 交互项检验（H3）----------
    df_i = df.copy()
    df_i["TONE_x_Over"] = df_i["TONE"] * df_i["D_overOptim"]
    df_i["TONE_x_Lev"] = df_i["TONE"] * df_i["D_highLev"]
    rows = []

    def rep_interact(y, x, inter, ctrl, label):
        d = df_i[["secCode", "year", y, x, inter] + ctrl].dropna().copy()
        X = pd.concat([d[[x, inter] + ctrl].astype(float),
                       pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)], axis=1)
        X = sm.add_constant(X)
        res = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": d["secCode"]})
        rows.append({
            "检验": label,
            "TONE 系数": round(res.params[x], 3), "t 值": round(res.tvalues[x], 2),
            "交互项系数": round(res.params[inter], 3), "交互项t值": round(res.tvalues[inter], 2),
            "交互项p值": round(res.pvalues[inter], 4),
            "交互项显著性": stars(res.pvalues[inter]), "N": int(res.nobs),
        })

    rep_interact("roa_next", "TONE", "TONE_x_Over", CTRL, "M1: TONE×过度乐观哑变量")
    rep_interact("roa_next", "TONE", "TONE_x_Lev", CTRL, "M2: TONE×高杠杆哑变量")
    grp = pd.DataFrame(rows)
    grp.to_csv(os.path.join(TABLE_DIR, "tab5_interaction.csv"), index=False, encoding="utf-8-sig")
    print("\n== 表5 语调可信度的交互项检验（H3）==")
    print(grp.to_string(index=False))

    # ---------- 图 1：语调分布 ----------
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.hist(df["TONE"], bins=24, color="#4C72B0", edgecolor="white")
    ax.axvline(df["TONE"].mean(), color="#C44E52", ls="--",
               label=f"均值 {df['TONE'].mean():.2f}")
    ax.set_xlabel("MD&A 净语调（百分数口径）")
    ax.set_ylabel("企业-年度个数")
    ax.set_title("图 1  管理层净语调的分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_tone_dist.png"))
    plt.close(fig)

    # ---------- 图 2：分年度语调均值 ----------
    yearly = df.groupby("year").agg(mean=("TONE", "mean"),
                                    sd=("TONE", "std"), n=("TONE", "size"))
    yearly["se"] = yearly["sd"] / np.sqrt(yearly["n"])
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.errorbar(yearly.index.astype(str), yearly["mean"], yerr=1.96 * yearly["se"],
                marker="o", capsize=4, color="#4C72B0")
    ax.set_xlabel("财年")
    ax.set_ylabel("净语调均值（含 95% 置信区间）")
    ax.set_title("图 2  管理层净语调的年度变化")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_tone_year.png"))
    plt.close(fig)

    # ---------- 图 3：语调与未来 ROA 散点 ----------
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    x, y = df["TONE"], df["roa_next"]
    ax.scatter(x, y, s=26, alpha=0.65, color="#4C72B0", edgecolor="white")
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, k * xs + b, color="#C44E52", lw=2,
            label=f"拟合线：ROA_next = {k:.2f}·TONE + {b:.2f}（r={x.corr(y):.2f}）")
    ax.set_xlabel("MD&A 净语调（百分数口径）")
    ax.set_ylabel("下一年 ROA（%）")
    ax.set_title("图 3  语调与未来经营业绩")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_tone_roa.png"))
    plt.close(fig)

    print(f"\n分析完成：表格 → {TABLE_DIR}\n          图 1-图 3 → {FIG_DIR}")


if __name__ == "__main__":
    main()
