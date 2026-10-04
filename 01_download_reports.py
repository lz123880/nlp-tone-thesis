# -*- coding: utf-8 -*-
"""
01_download_reports.py —— 数据获取模块
=====================================
两条独立的数据轨道，均为公开真实数据：

(A) 年报原文：巨潮资讯网（cninfo）公开披露接口
    POST http://www.cninfo.com.cn/new/hisAnnouncement/query
    检索指定公司指定年度的《年度报告》PDF，下载到 data/reports/。
    处理步骤：构造检索条件 -> 过滤标题（剔除摘要/英文版/修订稿）-> 下载 PDF -> 记录索引。

(B) 财务指标：东方财富数据中心公开接口（F10 主要指标，年报口径）
    GET https://datacenter-web.eastmoney.com/api/data/v1/get
    reportName=RPT_F10_FINANCE_MAINFINADATA
    取 ZZCJLL(总资产净利率ROA)、ROEJQ(加权ROE)、ZCFZL(资产负债率)、
       TOTALOPERATEREVETZ(营收同比)、PARENTNETPROFITTZ(归母净利同比) 等字段。

用法：
    python 01_download_reports.py --years 2022 2023 2024
    python 01_download_reports.py --years 2022 --only 000333,000651   # 调试子样本
输出：
    data/reports/*.pdf                       年报原文
    data/raw_index.csv                       公告下载索引
    data/financials_raw.csv                  财务指标原始表
"""
import argparse
import json
import os
import re
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORT_DIR = os.path.join(DATA, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ---------------------------------------------------------------------------
# 便利样本：40 家非金融 A 股上市公司（2018 年前上市、披露连续、行业分散）。
# 说明：这是"便利样本"而非随机抽样，选取标准为披露连续、市值较大、行业代表性，
#       便于教学演示下控制在可下载规模内；论文局限部分对此有专门讨论。
# ---------------------------------------------------------------------------
FIRMS = [
    ("000333", "美的集团"), ("000651", "格力电器"), ("600690", "海尔智家"), ("000921", "海信家电"),
    ("600519", "贵州茅台"), ("000858", "五粮液"), ("000568", "泸州老窖"), ("600887", "伊利股份"),
    ("603288", "海天味业"), ("002714", "牧原股份"), ("600276", "恒瑞医药"), ("000538", "云南白药"),
    ("600085", "同仁堂"), ("300760", "迈瑞医疗"), ("002415", "海康威视"), ("002241", "歌尔股份"),
    ("000725", "京东方A"), ("002475", "立讯精密"), ("600584", "长电科技"), ("600104", "上汽集团"),
    ("000625", "长安汽车"), ("002594", "比亚迪"), ("600309", "万华化学"), ("600585", "海螺水泥"),
    ("000708", "中信特钢"), ("300750", "宁德时代"), ("002460", "赣锋锂业"), ("601012", "隆基绿能"),
    ("600031", "三一重工"), ("000157", "中联重科"), ("601100", "恒立液压"), ("000002", "万科A"),
    ("600048", "保利发展"), ("601888", "中国中免"), ("600900", "长江电力"), ("601668", "中国建筑"),
    ("601857", "中国石油"), ("600028", "中国石化"), ("601899", "紫金矿业"), ("600150", "中国船舶"),
]

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STOCK_JSON = "http://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_PDF = "http://static.cninfo.com.cn/{}"
EM_MAIN = "https://datacenter-web.eastmoney.com/api/data/v1/get"

_ORG_CACHE = {}


def load_org_ids(session: requests.Session) -> dict:
    """巨潮检索的 stock 参数要求 '代码,orgId'，从全量股票清单获取映射。"""
    if _ORG_CACHE:
        return _ORG_CACHE
    path = os.path.join(DATA, "cninfo_stock_list.json")
    if os.path.exists(path):
        j = json.load(open(path, encoding="utf-8"))
    else:
        r = session.get(CNINFO_STOCK_JSON, headers=UA, timeout=60)
        open(path, "w", encoding="utf-8").write(r.text)
        j = r.json()
    for s in j.get("stockList", []):
        _ORG_CACHE[s["code"]] = s["orgId"]
    print(f"  已加载股票-组织ID映射 {len(_ORG_CACHE)} 条")
    return _ORG_CACHE

# 财报披露窗口：t 年年报在 t+1 年 1-6 月披露
DISCLOSE_WINDOW = {y: (f"{y + 1}-01-01~{y + 1}-06-30") for y in range(2018, 2027)}

# 标题黑名单：剔除摘要、英文版、修订/更正/取消等非首次版本
TITLE_EXCLUDE = re.compile(r"摘要|英文|修订|更正|取消|撤回|意见|问询|回复|补充|更新后|已作废")


def search_announcement(code: str, year: int, session: requests.Session, org_id: str) -> dict | None:
    """在巨潮检索 code 公司 year 财年的《年度报告》首次披露 PDF。"""
    payload = {
        "pageNum": 1, "pageSize": 30, "column": "szse", "tabName": "fulltext",
        "plate": "", "stock": f"{code},{org_id}", "searchkey": "", "secid": "",
        "category": "category_ndbg_szsh", "trade": "",
        "seDate": DISCLOSE_WINDOW[year], "sortName": "", "sortType": "", "isHLtitle": "true",
    }
    for attempt in range(3):
        try:
            r = session.post(CNINFO_QUERY, data=payload, headers=UA, timeout=30)
            anns = r.json().get("announcements") or []
            break
        except Exception as exc:  # 网络抖动重试
            if attempt == 2:
                print(f"    [warn] {code} {year} 检索失败：{exc}")
                return None
            time.sleep(2)
    want = f"{year}年年度报告"
    hits = [
        a for a in anns
        if want in (a.get("announcementTitle") or "")
        and not TITLE_EXCLUDE.search(a.get("announcementTitle") or "")
        and (a.get("adjunctType") or "PDF") == "PDF"
    ]
    if not hits:
        return None
    hits.sort(key=lambda a: len(a.get("announcementTitle") or ""))  # 标题最短者通常是首次披露版
    return hits[0]


def download_pdf(url: str, path: str, session: requests.Session) -> bool:
    for attempt in range(3):
        try:
            with session.get(url, headers=UA, timeout=120, stream=True) as r:
                r.raise_for_status()
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                os.replace(tmp, path)
            return os.path.getsize(path) > 30_000
        except Exception as exc:
            if attempt == 2:
                print(f"    [warn] 下载失败 {url}：{exc}")
                return False
            time.sleep(3)
    return False


def fetch_reports(years: list[int], only: list[str] | None) -> pd.DataFrame:
    session = requests.Session()
    orgs = load_org_ids(session)
    rows = []
    todo = [f for f in FIRMS if not only or f[0] in only]
    for i, (code, name) in enumerate(todo, 1):
        org = orgs.get(code, "")
        for year in years:
            pdf_path = os.path.join(REPORT_DIR, f"{code}_{year}.pdf")
            if os.path.exists(pdf_path):
                rows.append({"secCode": code, "secName": name, "year": year,
                             "title": "(cached)", "adjunctUrl": "", "pdf": pdf_path})
                continue
            hit = search_announcement(code, year, session, org)
            if hit is None:
                print(f"  [{i:>2}/{len(todo)}] {code} {name} {year} 未匹配到年度报告")
                rows.append({"secCode": code, "secName": name, "year": year,
                             "title": "(missing)", "adjunctUrl": "", "pdf": ""})
                continue
            url = CNINFO_PDF.format(hit["adjunctUrl"])
            ok = download_pdf(url, pdf_path, session)
            print(f"  [{i:>2}/{len(todo)}] {code} {name} {year} "
                  f"{'OK ' + str(hit.get('adjunctSize', 0)) + 'KB' if ok else 'FAIL'}")
            rows.append({"secCode": code, "secName": name, "year": year,
                         "title": hit.get("announcementTitle", ""),
                         "adjunctUrl": hit.get("adjunctUrl", ""),
                         "pdf": pdf_path if ok else ""})
            time.sleep(0.8)
    return pd.DataFrame(rows)


def fetch_financials(years: list[int], only: list[str] | None) -> pd.DataFrame:
    """东财 F10 主要指标（年报口径）。多取两年，保证 ROA_{t+1} 可得。"""
    session = requests.Session()
    out = []
    todo = [f for f in FIRMS if not only or f[0] in only]
    for i, (code, name) in enumerate(todo, 1):
        market = "SH" if code.startswith("6") else "SZ"
        params = {
            "reportName": "RPT_F10_FINANCE_MAINFINADATA", "columns": "ALL",
            "filter": f'(SECUCODE="{code}.{market}")(REPORT_TYPE="年报")',
            "pageNumber": 1, "pageSize": 12, "sortTypes": -1,
            "sortColumns": "REPORT_DATE", "source": "HSF10", "client": "PC",
        }
        try:
            r = session.get(EM_MAIN, params=params, headers=UA, timeout=30)
            data = (r.json().get("result") or {}).get("data") or []
        except Exception as exc:
            print(f"  [{i:>2}/{len(todo)}] {code} 财务数据失败：{exc}")
            continue
        for rec in data:
            y = int(str(rec.get("REPORT_DATE", ""))[:4])
            out.append({
                "secCode": code, "secName": name, "year": y,
                "roa": rec.get("ZZCJLL"), "roe": rec.get("ROEJQ"),
                "lev": rec.get("ZCFZL"), "revGrowth": rec.get("TOTALOPERATEREVETZ"),
                "profitGrowth": rec.get("PARENTNETPROFITTZ"), "revenue": rec.get("TOTALOPERATEREVE"),
                "assetTurnoverDays": rec.get("ZZCZZTS"),
            })
        print(f"  [{i:>2}/{len(todo)}] {code} {name} 财务指标 {len(data)} 期")
        time.sleep(0.5)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    only = [c.strip() for c in args.only.split(",") if c.strip()]

    print(f"== A. 下载年报原文（巨潮资讯，{args.years}）==")
    idx = fetch_reports(args.years, only or None)
    idx.to_csv(os.path.join(DATA, "raw_index.csv"), index=False, encoding="utf-8-sig")
    ok = (idx["pdf"] != "").sum()
    print(f"   报告下载成功 {ok}/{len(idx)}")

    print("== B. 下载财务指标（东方财富数据中心）==")
    fin = fetch_financials(args.years, only or None)
    fin.to_csv(os.path.join(DATA, "financials_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"   财务记录 {len(fin)} 条")

    print("完成。")


if __name__ == "__main__":
    main()
