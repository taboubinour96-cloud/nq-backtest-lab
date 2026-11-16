"""
data_yahoo_crypto.py — الكريبتو من Yahoo (اقتراح Arena 1994، تحقّقتُ منه).

الفائدة المقاسة: تاريخ أطول بكثير من Binance Futures.
    BTC-USD · LTC-USD : من 2014-09-17  (12.2 سنة)
    ETH/XRP/DOGE/BNB/ADA/LINK : من 2017-11-09  (9 سنوات)
    مقابل Binance Vision : من 2020-01-01 (6.8 سنة)

لماذا يهم: كنت كتبتُ في تقريري أن «ضعف الكريبتو قد يكون أثراً لطول البيانات
(6 سنوات) لا لضعف الأصل». هذه الوحدة تسمح باختبار ذلك بدل تخمينه.

فروق يجب التصريح بها:
  • Yahoo يعطي سعر **فوري (spot)** بالدولار، لا عقد دائم. لا تمويل ولا رافعة.
  • الحجم (Volume) موجود لكنه ليس حجم عقود دائمة — لا يصلح لترتيب السيولة.
  • لذلك تُستخدم هذه الوحدة لاختبار الميزة على تاريخ أطول، لا لاختيار الأسواق.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache_yahoo"
CACHE.mkdir(exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (backtest-lab)"})

# الرموز الستة في مستودعنا، بصيغة Yahoo
REPO6 = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
         "BNBUSDT": "BNB-USD", "XRPUSDT": "XRP-USD", "DOGEUSDT": "DOGE-USD"}


def download(symbol: str, start: str = "2014-01-01", use_cache: bool = True) -> pd.DataFrame:
    f = CACHE / f"{symbol.replace('^','IDX_').replace('=','_')}_1d.csv"
    if use_cache and f.is_file():
        df = pd.read_csv(f, parse_dates=["DateTime"])
        if len(df) > 100:
            return df
    p1 = int(time.mktime(time.strptime(start, "%Y-%m-%d")))
    r = S.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
              params={"interval": "1d", "period1": p1, "period2": int(time.time())}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "DateTime": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "Open": q["open"], "High": q["high"], "Low": q["low"],
        "Close": q["close"], "Volume": q["volume"],
    }).dropna(subset=["Open", "High", "Low", "Close"])
    df = df.drop_duplicates("DateTime").sort_values("DateTime").reset_index(drop=True)
    df.to_csv(f, index=False)
    return df


def repo6(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    out = {}
    for repo_sym, y_sym in REPO6.items():
        try:
            df = download(y_sym, use_cache=use_cache)
            if len(df) > 250:
                out[repo_sym] = df
        except Exception as e:
            print(f"  {y_sym} فشل: {str(e)[:50]}")
    return out


if __name__ == "__main__":
    d = repo6()
    for s, df in d.items():
        print(f"{s:10s} {len(df):5d} شمعة  {df.DateTime.min().date()} → {df.DateTime.max().date()}")
