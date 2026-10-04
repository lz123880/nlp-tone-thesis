# -*- coding: utf-8 -*-
"""
02_text_preprocess.py —— 文本预处理模块
=======================================
输入：data/reports/*.pdf（巨潮资讯下载的年报原文）与 data/raw_index.csv
输出：data/mdna_corpus.csv（企业-年度 MD&A 文本与分词结果）

处理步骤：
  1) PyMuPDF 逐页抽取 PDF 文本；
  2) 以"第X节"为界切分章节，定位《管理层讨论与分析》（MD&A）章节正文；
  3) 清洗：去页眉页脚/页码/多余空白，归一化全角符号；
  4) jieba 分词（去除停用词与单字），同时保留句子切分供句级语调使用。
"""
import os
import re

import fitz  # PyMuPDF（pip 包名 pymupdf，兼容旧导入名 fitz）
import jieba
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LEX = os.path.join(ROOT, "code", "lexicon")

SEC_RE = re.compile(r"第[一二三四五六七八九十]{1,2}节[^\n]{0,30}")
# 章节名候选：正式准则式名称在前，旧版/特殊版式名称在后
MDNA_KEYS = ["管理层讨论与分析", "经营情况讨论与分析", "董事会报告"]
MIN_CHARS = 2_000
MAX_CHARS = 60_000  # MD&A 截断长度（超出部分多为表格，语调密度低）


def pdf_full_text(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        txt = page.get_text("text")
        pages.append(txt)
    doc.close()
    return "\n".join(pages)


def clean_page_noise(text: str) -> str:
    """去页眉页脚：公司名+页码横线、独立页码、目录点线等；归一化全角空格。"""
    text = text.replace("\u2003", "").replace("\u3000", "")
    text = re.sub(r"\n\s*\d+\s*/\s*\d+\s*\n", "\n", text)          # 12/150 形式页码
    text = re.sub(r"\n\s*第\s*\d+\s*页\s*共\s*\d+\s*页\s*\n", "\n", text)
    text = re.sub(r"\n[-—_=·.]{3,}\n", "\n", text)                 # 分隔线
    lines = [ln for ln in text.split("\n") if ln.strip() and len(ln.strip()) > 1]
    lines = [ln for ln in lines if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]  # 纯页码行
    return "\n".join(lines)


def is_real_heading_line(line: str) -> bool:
    """真正的章节标题行：短、无句读、无交叉引用语。排除'参见第三节 四 2（8）'类引用。"""
    if len(line) > 32:
        return False
    if any(k in line for k in ["。", "“", "”", "《", "》", "参见", "详见", "如下", "同上"]):
        return False
    return True


def split_sections(text: str) -> list[tuple[str, str]]:
    """按'第X节'切块，并合并因**页眉重复章节名**而被切碎的连续同标题块。"""
    marks = [m for m in SEC_RE.finditer(text) if is_real_heading_line(m.group().strip())]
    blocks: list[list] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[m.start():end]
        head = m.group()
        if blocks and blocks[-1][0] == head and len(blocks[-1][1]) < 6000:
            blocks[-1][1] += seg  # 同名页眉造成的碎片：向后合并
        else:
            blocks.append([head, seg])
    return [(h, s) for h, s in blocks]


def extract_mdna(text: str) -> str:
    """定位 MD&A 章节：优先准则式标题，逐级兜底。"""
    blocks = split_sections(text)
    for key in MDNA_KEYS:
        cand = [seg for head, seg in blocks if key in head]
        if cand:
            mdna = max(cand, key=len)
            if len(mdna) >= MIN_CHARS:
                return finish(mdna)
        # 兜底 1：标题行不含但正文含关键词的块
        cand = [seg for head, seg in blocks if key in seg[:400]]
        if cand:
            mdna = max(cand, key=len)
            if len(mdna) >= MIN_CHARS:
                return finish(mdna)
    # 兜底 2：全文关键词窗口（无"第X节"版式，如部分公司自定结构）
    for key in MDNA_KEYS:
        k = text.find(key)
        if k >= 0:
            nxt = [mm.start() for mm in SEC_RE.finditer(text) if mm.start() > k + 200]
            seg = text[k:nxt[0]] if nxt else text[k:k + MAX_CHARS]
            if len(seg) >= MIN_CHARS:
                return finish(seg)
    return ""


def finish(mdna: str) -> str:
    mdna = mdna[:MAX_CHARS]
    mdna = re.sub(r"\n\s*[一二三四五六七八九十]{1,3}、\s*", "\n", mdna)
    mdna = re.sub(r"\n\s*\(\s*[一二三四五六七八九十\d]{1,3}\s*\)\s*", "\n", mdna)
    mdna = re.sub(r"[ \t]+", "", mdna)
    mdna = re.sub(r"\n{2,}", "\n", mdna)
    return mdna.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？；;\n]", text)
    return [p for p in (s.strip() for s in parts) if len(p) >= 4]


def load_stopwords() -> set[str]:
    base = set("的了和与及及对在是为将由从以按通过本次报告期年度公司我们其中主要相关情况如下同时此外另另外以及或者但但是然而不过因此所以由于根据按照对于关于截至目前同时仍在其中等等各类各项目工作方面方面工作将会已经尚未正在持续不断进一步大力加快推动促进开展实施完成进行保持实现达到符合满足起到起到积极作用".split())
    return base


STOP = load_stopwords()


def tokenize(text: str) -> list[str]:
    words = jieba.lcut(text)
    return [w for w in words if len(w) >= 2 and w not in STOP
            and not re.fullmatch(r"[\d.%.,%/()（）一二三四五六七八九十百千万亿]+", w)]


def main():
    idx = pd.read_csv(os.path.join(DATA, "raw_index.csv"), dtype={"secCode": str})
    rows = []
    for _, r in idx.iterrows():
        pdf = r["pdf"]
        if not isinstance(pdf, str) or not pdf or not os.path.exists(pdf):
            continue
        try:
            raw = pdf_full_text(pdf)
            mdna = extract_mdna(clean_page_noise(raw))
        except Exception as exc:
            print(f"[warn] {r['secCode']} {r['year']} 解析失败：{exc}")
            continue
        if len(mdna) < MIN_CHARS:
            print(f"[warn] {r['secCode']} {r['year']} MD&A 过短（{len(mdna)} 字符），弃用")
            continue
        sents = split_sentences(mdna)
        toks = tokenize(mdna)
        sent_tok_lines = [" ".join(tokenize(s)) for s in sents]  # 句级分词（供句级语调用）
        rows.append({
            "secCode": r["secCode"], "secName": r["secName"], "year": int(r["year"]),
            "title": r["title"], "nChars": len(mdna), "nSent": len(sents), "nTokens": len(toks),
            "mdna": mdna, "sentences": "\n".join(sents), "sentTokens": "\n".join(sent_tok_lines),
            "tokens": " ".join(toks),
        })
        print(f"  {r['secCode']} {r['secName']} {r['year']} MD&A {len(mdna)} 字符 / {len(sents)} 句 / {len(toks)} 词")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "mdna_corpus.csv"), index=False, encoding="utf-8-sig")
    print(f"\n语料构建完成：{len(out)} 个企业-年度文档 → data/mdna_corpus.csv")


if __name__ == "__main__":
    main()
