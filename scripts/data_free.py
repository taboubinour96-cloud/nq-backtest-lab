"""
data_free.py — بيانات أسعار مجانية 100%، بلا حساب، بلا مفتاح API، بلا بطاقة.

المصدر: data.binance.vision (Binance Vision) — مستودع عام على S3، نُشر بموجب
قانون الشفافية في Binance، وهو نفس المصدر الذي اختاره قرار المشروع رقم 4
(«كل البيانات من ملفات عامة (data.binance.vision) — صفر مخاطرة قانونية وتشغيلية»).

لماذا هو حلّ للمشكلة:
  • لا يلزم تسجيل، لا KYC، لا مفاتيح، لا حدود على IP المستخدم (S3 عام).
  • شموع يومية للمستقبليات (UM futures) من 2019-09 إلى اليوم.
  • ملفات شهرية مضغوطة: شهر واحد/ملف، والتاريخ كامل ⇒ يمكن بناء عيّنة
    كبيرة (مئات الصفقات) وهو الشرط الذي يطلبه دستور الباك-تست عندنا (≥400).

الوظائف:
  list_symbols()          -> كل رموز المستقبليات المتاحة
  has_data(symbol, month) -> هل توجد بيانات لذلك الشهر (HEAD، رخيص)
  available_from(symbol)  -> أقدم شهر متاح
  download_daily(symbol)  -> DataFrame يومي كامل (Open/High/Low/Close/Volume)
  download_many(...)      -> عدة رموز بالتوازي + تخزين مؤقت

التشغيل المباشر يطبع تقريراً عن المتاح فعلاً (لا افتراضات).
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

COLS = ["open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2027) for m in range(1, 13)]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "backtest-lab/1.0"})
    return s


S = _session()


def list_symbols(prefix: str = "data/futures/um/monthly/klines/") -> list[str]:
    """كل الرموز عبر S3 listing (مقسّم على صفحات)."""
    out: list[str] = []
    marker = ""
    while True:
        r = S.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=30)
        r.raise_for_status()
        txt = r.text
        for chunk in txt.split("<CommonPrefixes><Prefix>")[1:]:
            out.append(chunk.split("</Prefix>")[0].replace(prefix, "").rstrip("/"))
        if "<IsTruncated>true</IsTruncated>" not in txt:
            break
        marker = txt.split("<NextMarker>")[1].split("</NextMarker>")[0] if "<NextMarker>" in txt else ""
        if not marker:
            break
    return sorted(set(out))


def _url(symbol: str, month: str) -> str:
    return f"{BASE}/data/futures/um/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip"


def has_data(symbol: str, month: str) -> bool:
    try:
        return S.head(_url(symbol, month), timeout=20, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


def symbol_months(symbol: str) -> list[str]:
    """قائمة الأشهر المتاحة لرمز — طلب S3 واحد بدل 40 طلب HEAD."""
    prefix = f"data/futures/um/monthly/klines/{symbol}/1d/"
    try:
        r = S.get(S3, params={"prefix": prefix}, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return []
    out = []
    for chunk in r.text.split("<Key>")[1:]:
        key = chunk.split("</Key>")[0]
        name = key.rsplit("/", 1)[-1]                     # BTCUSDT-1d-2025-01.zip
        m = name.replace(f"{symbol}-1d-", "").replace(".zip", "")
        if len(m) == 7 and m[4] == "-":
            out.append(m)
    return sorted(set(out))


def available_from(symbol: str) -> str | None:
    ms = symbol_months(symbol)
    return ms[0] if ms else None


def download_daily(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """كل الشموع اليومية لرمز، من أول شهر متاح إلى آخر شهر."""
    f = CACHE / f"{symbol}_1d.csv"
    if use_cache and f.is_file():
        df = pd.read_csv(f, parse_dates=["DateTime"])
        if len(df):
            return df

    months = symbol_months(symbol)
    if not months:
        return pd.DataFrame()

    frames = []
    misses = 0
    for m in months:
        try:
            r = S.get(_url(symbol, m), timeout=30)
        except requests.RequestException:
            misses += 1
            if misses >= 5:
                break
            continue
        if r.status_code != 200:
            misses += 1
            if misses >= 5:            # خمسة أشهر مفقودة = توقّف السجل
                break
            continue
        misses = 0
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
        except Exception:
            continue
        if str(df.iloc[0, 0]) == "open_time":      # بعض الملفات فيها سطر عنوان
            df = df.iloc[1:]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["DateTime"] = pd.to_datetime(d["open_time"].astype("int64"), unit="ms")
    for c in ("Open", "High", "Low", "Close", "Volume", "quote_volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = (d.dropna(subset=["Open", "High", "Low", "Close"])
          .drop_duplicates("DateTime").sort_values("DateTime").reset_index(drop=True))
    d[["DateTime", "Open", "High", "Low", "Close", "Volume", "quote_volume"]].to_csv(f, index=False)
    return d


def download_many(symbols: list[str], workers: int = 8, verbose: bool = True) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(workers) as ex:
        for sym, df in zip(symbols, ex.map(download_daily, symbols)):
            if len(df):
                out[sym] = df
                if verbose:
                    print(f"  {sym:14s} {len(df):5d} شمعة  {df.DateTime.min().date()} → {df.DateTime.max().date()}")
    return out


if __name__ == "__main__":
    syms = list_symbols()
    print(f"عدد رموز المستقبليات المتاحة على Binance Vision: {len(syms)}")
    usdt = [s for s in syms if s.endswith("USDT")]
    print(f"منها USDT: {len(usdt)}")
    test = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
    print("\nاختبار تحميل فعلي:")
    d = download_many(test)
    for s, df in d.items():
        print(f"{s}: {len(df)} شمعة، من {df.DateTime.min().date()} إلى {df.DateTime.max().date()}")
