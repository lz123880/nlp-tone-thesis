# -*- coding: utf-8 -*-
"""
04_merge_panel.py —— 面板构建模块
=================================
输入：data/measures.csv（03 输出）与 data/financials_raw.csv（01 输出）
输出：output/tone_panel.csv —— 企业-年度非平衡面板（核心交付表）

变量定义表（详见论文表 1）：
  secCode/secName/year       代码 / 简称 / 财年
  TONE_dict, TONE_ml, TONE_sd  语调测度（词典法 / 机器学习法 / 句级离散度）
  roa_next                   被解释变量：t+1 年总资产净利率（ZZCJLL，%）
  roa                        控制变量：t 年 ROA（%）
  size                       控制变量：ln(营业总收入，千元)
  lev                        控制变量：资产负债率（%）
  revGrowth                  控制变量：营业收入同比增速（%）
  profitGrowth               分组变量：归母净利润同比增速（%）
  D_overOptim                语调偏离：TONE 超出同年横截面均值 1 个标准差记 1
  D_highLev                  杠杆分组：lev 高于样本中位数记 1
处理：
  1) 语调表与财务表按 (secCode, secName, year) 合并；
  2) 从财务表直接取 (secCode, year+1) 的 ROA / ROE 构造 roa_next / roe_next
     （财务表多取两年，故 2024 财年的 t+1 被解释变量可得）；
  3) 对连续变量做上下 1% 缩尾（winsorize），缩尾前取值存入 *_raw 列备查；
  4) 缺失值观测在回归前剔除并记录剔除原因。
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def main():
    meas = pd.read_csv(os.path.join(DATA, "measures.csv"), dtype={"secCode": str})
    fin = pd.read_csv(os.path.join(DATA, "financials_raw.csv"), dtype={"secCode": str})

    panel = meas.merge(fin, on=["secCode", "secName", "year"], how="left")
    panel = panel.sort_values(["secCode", "year"]).reset_index(drop=True)

    # 未来业绩：直接从财务表取 t+1 年报口径（财务表已多取两年，保证 2024 → 2025 可得）
    fin_map_roa = {(r.secCode, r.year): r.roa for r in fin.itertuples()}
    fin_map_roe = {(r.secCode, r.year): r.roe for r in fin.itertuples()}
    panel["roa_next"] = [fin_map_roa.get((c, y + 1), np.nan)
                         for c, y in zip(panel["secCode"], panel["year"])]
    panel["roe_next"] = [fin_map_roe.get((c, y + 1), np.nan)
                         for c, y in zip(panel["secCode"], panel["year"])]

    # 规模：营业总收入（元）取对数
    panel["size"] = np.log(panel["revenue"].replace(0, np.nan))

    # 语调偏离（行业内-年度标准化）
    g = panel.groupby("year")["TONE_dict"]
    mu, sd = g.transform("mean"), g.transform("std")
    panel["D_overOptim"] = ((panel["TONE_dict"] - mu) > sd).astype(int)

    # 杠杆分组哑变量（高于样本中位数记 1）
    panel["D_highLev"] = (panel["lev"] > panel["lev"].median()).astype(int)

    # 缩尾（仅连续数值变量，0-1 变量除外）
    for c in ["TONE_dict", "TONE_ml", "TONE_sd", "roa", "roa_next", "size", "lev", "revGrowth"]:
        if c in panel:
            panel[c + "_raw"] = panel[c]
            panel[c] = winsorize(panel[c].astype(float))

    panel.to_csv(os.path.join(OUT, "tone_panel.csv"), index=False, encoding="utf-8-sig")

    valid = panel.dropna(subset=["TONE_dict", "roa_next", "roa", "size", "lev", "revGrowth"])
    print(f"面板构建完成：{len(panel)} 行（语调×财务成功匹配 {panel['roa'].notna().sum()} 行）")
    print(f"回归可用样本：{len(valid)} 个企业-年度观测，覆盖 {valid['secCode'].nunique()} 家公司")
    print(f"\n核心变量描述：")
    print(valid[["TONE_dict", "TONE_ml", "roa", "roa_next", "size", "lev", "revGrowth"]]
          .describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())
    print(f"\n→ output/tone_panel.csv")


if __name__ == "__main__":
    main()
