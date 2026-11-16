"""
data_yahoo.py — بيانات المؤشرات والسلع والفوركس مجاناً (بلا مفتاح، بلا حساب).

المصدر: واجهة Yahoo Finance Chart (عامة). تُستخدم هنا فقط لجلب أسعار تاريخية،
وهي بيانات سوق عامة لا بيانات شخصية.

التغطية المقاسة (2026-11): 5,502 شمعة يومية من 2005-01-03 إلى اليوم لأغلب
الرموز — أي 22 سنة، وهي أطول بكثير من أي عيّنة استخدمناها سابقاً (4-7 سنوات).

لماذا هذا مهم لمشروعنا:
    أداة شركة التمويل (FundedNext/FTMO) هي NAS100 / SP500 / US30 — أي ^NDX /
    ^GSPC / ^DJI. وكان مستودعنا يسمّي هذا المسار «INDEX_CFD» ويعجز عن اختباره
    لأن البيانات كانت مدفوعة أو محجوبة. الآن هو متاح مجاناً وبعمق 22 سنة.

تحذير أمانة: ^NDX مؤشر (لا عقد CFD). فرق السبريد والعمولة يُضاف في المحرّك
كتكاليف تقديرية معلنة، وليس صفراً.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (backtest-lab)"})

# الأدوات التي تُتداول فعلاً في شركات التمويل + سلع وفوركس للمقارنة
UNIVERSE = {
    "^NDX": "US100 (Nasdaq 100)",
    "^GSPC": "US500 (S&P 500)",
    "^DJI": "US30 (Dow Jones)",
    "GC=F": "الذهب",
    "CL=F": "النفط WTI",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "SI=F": "الفضة",
    "HG=F": "النحاس",
}


def download(symbol: str, start: str = "2005-01-01", use_cache: bool = True) -> pd.DataFrame:
    f = CACHE / f"{symbol.replace('^','IDX_').replace('=','_')}_1d.csv"
    if use_cache and f.is_file():
        df = pd.read_csv(f, parse_dates=["DateTime"])
        if len(df) > 100:
            return df

    p1 = int(time.mktime(time.strptime(start, "%Y-%m-%d")))
    p2 = int(time.time())
    r = S.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
              params={"interval": "1d", "period1": p1, "period2": p2}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "DateTime": pd.to_datetime(ts, unit="s").normalize(),
        "Open": q["open"], "High": q["high"], "Low": q["low"],
        "Close": q["close"], "Volume": q["volume"],
    }).dropna(subset=["Open", "High", "Low", "Close"])
    df = df.drop_duplicates("DateTime").sort_values("DateTime").reset_index(drop=True)
    df.to_csv(f, index=False)
    return df


def download_universe(use_cache: bool = True, verbose: bool = True) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, label in UNIVERSE.items():
        try:
            df = download(sym, use_cache=use_cache)
        except Exception as e:
            if verbose:
                print(f"  {sym:10s} فشل: {str(e)[:60]}")
            continue
        if len(df) > 250:
            out[sym] = df
            if verbose:
                print(f"  {sym:10s} {label:20s} {len(df):5d} شمعة  {df.DateTime.min().date()} → {df.DateTime.max().date()}")
    return out


if __name__ == "__main__":
    d = download_universe()
    print(f"\nنجح: {len(d)}/{len(UNIVERSE)} أداة")
