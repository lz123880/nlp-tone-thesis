# -*- coding: utf-8 -*-
"""
03_build_measures.py —— 测度构建模块（核心）
===========================================
输入：data/mdna_corpus.csv（02 的输出，含 tokens / sentences）
输出：data/measures.csv（企业-年度 × 测度）与 output/tables/tab_reliability.csv

测度体系
--------
TONE_dict  词典法净语调（主测度）
           = (正面词频 − 负面词频) / 总词数
           其中词频按句计算，支持"否定词翻转"与"程度副词加权"：
             否定词后 3 个词窗口内的情感词极性翻转；
             程度副词按档位加权（强化 2.0 / 弱化 0.5）。
TONE_ml    机器学习替代测度
           TF-IDF 词频特征 + 逻辑回归，以词典法极端组（上下 1/3 分位）为弱监督标签，
           5 折交叉验证预测全样本的"正面语调概率"（避免样本内乐观偏误）。
TONE_sd    句级语调标准差（管理层内部分歧的文本代理）
PosRatio / NegRatio  正、负面词占比（构建过程变量）
Skepticism 疑虑词占比（风险、不确定等），用于稳健性

信度检验
--------
split-half 分半信度：将每篇 MD&A 的句子按奇偶拆成两半，分别计算语调，
           两列相关系数经 Spearman-Brown 校正。
效度检验
--------
聚合效度：TONE_dict 与 TONE_ml 的相关系数；
已知组效度：归母净利同比为正组 vs 为负组的语调差异（t 检验）；
预测效度：TONE_t 与 ROA_{t+1} 的相关系数（详见 05 模块）。
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TABLE_DIR = os.path.join(ROOT, "output", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)
LEX = os.path.join(ROOT, "code", "lexicon")

DEGREE_W = {"strong": 2.0, "weak": 0.5}
STRONG_DEGREE = set("极其 极为 极度 非常 十分 格外 分外 特别 尤其 高度 显著 大幅 大力 大量 大范围 明显 更加 越发 日益 愈发 愈加 持续 不断 反复 频繁 密集 快速 迅速 急剧 急速 飞速 高速 严重 恶劣 深刻 深远 全面 全力 全方位 深度 充分 切实 有力 着力 积极稳步 高速增长".split())
WEAK_DEGREE = set("略有 略微 轻微 稍微 稍许 些许 小幅 一定程度 一定 逐步 渐进 缓慢 相对 比较 较为 较".split())
NEG_WINDOW = 3  # 否定词影响其后 3 个词


def load_lexicon():
    rd = lambda name: [w for w in open(os.path.join(LEX, name), encoding="utf-8").read().split() if w]
    return set(rd("positive.txt")), set(rd("negative.txt")), set(rd("negation.txt")), STRONG_DEGREE | WEAK_DEGREE


POS, NEG, NEGATION, DEGREE = load_lexicon()


def sentence_tone(sent_tokens: list[str]) -> tuple[float, int, int, int]:
    """单句语调：返回 (净语调, 正面计数, 负面计数, 否定翻转计数)。"""
    pos_cnt = neg_cnt = flip = 0
    n = len(sent_tokens)
    for i, w in enumerate(sent_tokens):
        if w not in POS and w not in NEG:
            continue
        polarity = 1 if w in POS else -1
        weight = 1.0
        # 程度副词：向前看 2 个词
        for j in range(max(0, i - 2), i):
            if sent_tokens[j] in STRONG_DEGREE:
                weight = DEGREE_W["strong"]
            elif sent_tokens[j] in WEAK_DEGREE:
                weight = max(weight, DEGREE_W["weak"])
        # 否定词：向前看 3 个词
        if any(sent_tokens[j] in NEGATION for j in range(max(0, i - NEG_WINDOW), i)):
            polarity = -polarity
            flip += 1
        if polarity > 0:
            pos_cnt += weight
        else:
            neg_cnt += weight
    return pos_cnt - neg_cnt, pos_cnt, neg_cnt, flip


def doc_measures(tokens: str, sentences_toks: list[list[str]]) -> dict:
    toks = tokens.split()
    total = len(toks)
    sent_scores, pos_all, neg_all = [], 0.0, 0.0
    for st in sentences_toks:
        s, p, n, _ = sentence_tone(st)
        sent_scores.append(s / max(1, len(st)))
        pos_all += p
        neg_all += n
    tone = (pos_all - neg_all) / max(1, total)
    return {
        "TONE_dict": tone,
        "PosRatio": pos_all / max(1, total),
        "NegRatio": neg_all / max(1, total),
        "TONE_sd": float(np.std(sent_scores, ddof=1)) if len(sent_scores) > 2 else np.nan,
        "nSent": len(sent_scores),
    }


def half_measures(sentences_toks: list[list[str]], part: str) -> float:
    idx = range(0, len(sentences_toks), 2) if part == "odd" else range(1, len(sentences_toks), 2)
    sel = [sentences_toks[i] for i in idx]
    pos = neg = 0.0
    total = sum(len(s) for s in sel)
    for st in sel:
        s, p, n, _ = sentence_tone(st)
        pos += p
        neg += n
    return (pos - neg) / max(1, total)


def build_ml_tone(df: pd.DataFrame, seed: int = 42) -> tuple[pd.Series, dict]:
    """TF-IDF + 逻辑回归弱监督语调，5 折交叉验证预测。

    返回 (语调概率序列, 模型表现指标)。
    """
    # 按百分位秩取上下 1/3 作弱标签（避免测度并列值导致分位区间为空）
    rank = df["TONE_dict"].rank(pct=True, method="average")
    y = pd.Series(np.nan, index=df.index)
    y[rank >= 2 / 3] = 1
    y[rank <= 1 / 3] = 0
    mask = y.notna()
    vec = TfidfVectorizer(max_features=4000, token_pattern=r"(?u)\b\w+\b",
                          min_df=3, max_df=0.9, sublinear_tf=True)
    X = vec.fit_transform(df.loc[mask, "tokens"])
    yv = y[mask].astype(int)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    prob = cross_val_predict(clf, X, yv, cv=5, method="predict_proba")[:, 1]

    # 交叉验证表现：准确率与 AUC（规模较小的极端组，用留一折评估）
    cv_acc = float(((prob >= 0.5).astype(int) == yv.values).mean())
    try:
        from sklearn.metrics import roc_auc_score
        cv_auc = float(roc_auc_score(yv.values, prob))
    except Exception:
        cv_auc = float("nan")
    metrics = {"n_label": int(mask.sum()), "n_feature": int(X.shape[1]),
               "cv_acc": cv_acc, "cv_auc": cv_auc}

    out = pd.Series(np.nan, index=df.index)
    out[mask] = prob
    # 未入极端组的样本用全样本模型补齐
    if (~mask).any():
        clf_full = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
            vec.transform(df.loc[mask, "tokens"]), yv)
        out[~mask] = clf_full.predict_proba(vec.transform(df.loc[~mask, "tokens"]))[:, 1]
    return out, metrics


def known_group_test(meas: pd.DataFrame) -> dict:
    """已知组效度：业绩增长组 vs 业绩下滑组的语调差异（Welch t 检验）。"""
    fin = pd.read_csv(os.path.join(DATA, "financials_raw.csv"), dtype={"secCode": str})
    key = {(r.secCode, int(r.year)): r.profitGrowth for r in fin.itertuples()
           if pd.notna(r.profitGrowth)}
    g = [key.get((c, int(y)), np.nan) for c, y in zip(meas["secCode"], meas["year"])]
    d = meas.assign(profitGrowth=g).dropna(subset=["profitGrowth"])
    up = d.loc[d["profitGrowth"] > 0, "TONE_dict"]
    down = d.loc[d["profitGrowth"] < 0, "TONE_dict"]
    t, p = sps.ttest_ind(up, down, equal_var=False)
    return {"n_up": int(up.size), "n_down": int(down.size),
            "mean_up": float(up.mean()), "mean_down": float(down.mean()),
            "diff": float(up.mean() - down.mean()), "t": float(t), "p": float(p)}


def predictive_validity(meas: pd.DataFrame) -> dict:
    """预测效度：TONE_t 与 ROA_{t+1} 的相关系数。"""
    fin = pd.read_csv(os.path.join(DATA, "financials_raw.csv"), dtype={"secCode": str})
    roa_fwd = {(r.secCode, int(r.year) - 1): r.roa for r in fin.itertuples() if pd.notna(r.roa)}
    y = [roa_fwd.get((c, int(yr)), np.nan) for c, yr in zip(meas["secCode"], meas["year"])]
    d = meas.assign(roa_next=y).dropna(subset=["roa_next"])
    r, p = sps.pearsonr(d["TONE_dict"], d["roa_next"])
    return {"n": len(d), "r": float(r), "p": float(p)}


def main():
    df = pd.read_csv(os.path.join(DATA, "mdna_corpus.csv"),
                     dtype={"secCode": str, "tokens": str, "sentTokens": str})
    recs = []
    for _, r in df.iterrows():
        # 优先使用 02 模块输出的句级分词；兼容旧语料（用 jieba 现场分词）
        if isinstance(r.get("sentTokens"), str) and r["sentTokens"].strip():
            sent_toks = [s.split() for s in r["sentTokens"].split("\n") if s.strip()]
        else:
            from importlib import import_module
            m02 = import_module("02_text_preprocess")
            sent_toks = [m02.tokenize(s) for s in str(r["sentences"]).split("\n")]
        m = doc_measures(str(r["tokens"]), sent_toks)
        m["TONE_odd"] = half_measures(sent_toks, "odd")
        m["TONE_even"] = half_measures(sent_toks, "even")
        m["secCode"], m["secName"], m["year"] = r["secCode"], r["secName"], r["year"]
        m["nChars"], m["nTokens"] = r["nChars"], r["nTokens"]
        recs.append(m)
    meas = pd.DataFrame(recs)

    # 疑虑度：风险类词占比（用负面词典中"风险/不确定"族）
    risk_words = set("风险 不确定性 挑战 隐患 危机 波动 不确定 敞口 承压 受挫 面临".split())
    meas["Skepticism"] = [
        sum(t in risk_words for t in toks.split()) / max(1, len(toks.split()))
        for toks in df["tokens"].astype(str)
    ]

    # 机器学习替代测度需要分词特征
    meas = meas.merge(df[["secCode", "year", "tokens"]], on=["secCode", "year"], how="left")
    meas["TONE_ml"], ml_metrics = build_ml_tone(meas)
    meas = meas.drop(columns=["tokens"])
    meas.to_csv(os.path.join(DATA, "measures.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 信度与效度 ----------------
    ok = meas.dropna(subset=["TONE_odd", "TONE_even"])
    r_half = sps.pearsonr(ok["TONE_odd"], ok["TONE_even"])[0]
    sb = 2 * r_half / (1 + r_half)
    r_agg = sps.pearsonr(meas["TONE_dict"], meas["TONE_ml"])[0]
    kg = known_group_test(meas)
    pv = predictive_validity(meas)

    rows = [
        {"检验": "分半信度（奇偶句拆分）", "统计量": "Pearson r", "取值": round(r_half, 4)},
        {"检验": "分半信度（Spearman-Brown 校正）", "统计量": "r_SB", "取值": round(sb, 4)},
        {"检验": "聚合效度（词典法 vs 机器学习法）", "统计量": "Pearson r", "取值": round(r_agg, 4)},
        {"检验": "已知组效度（业绩增长组 vs 下滑组语调差）", "统计量": "均值差", "取值": round(kg["diff"], 4)},
        {"检验": "已知组效度", "统计量": "Welch t 值", "取值": round(kg["t"], 4)},
        {"检验": "已知组效度", "统计量": "p 值", "取值": round(kg["p"], 4)},
        {"检验": "预测效度（TONE_t 与 ROA_{t+1}）", "统计量": "Pearson r", "取值": round(pv["r"], 4)},
        {"检验": "预测效度", "统计量": "p 值", "取值": round(pv["p"], 4)},
        {"检验": "机器学习测度交叉验证", "统计量": "准确率", "取值": round(ml_metrics["cv_acc"], 4)},
        {"检验": "机器学习测度交叉验证", "统计量": "AUC", "取值": round(ml_metrics["cv_auc"], 4)},
    ]
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TABLE_DIR, "tab_reliability.csv"), index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))
    print(f"\n机器学习弱监督：弱标签 {ml_metrics['n_label']} 篇、TF-IDF 特征 {ml_metrics['n_feature']} 维")
    print(f"已知组：增长组 n={kg['n_up']}（均值 {kg['mean_up']:.4f}）、"
          f"下滑组 n={kg['n_down']}（均值 {kg['mean_down']:.4f}）")
    print(f"\n测度构建完成：{len(meas)} 个企业-年度观测 → data/measures.csv")
    print(meas[["TONE_dict", "PosRatio", "NegRatio", "TONE_sd", "TONE_ml"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
