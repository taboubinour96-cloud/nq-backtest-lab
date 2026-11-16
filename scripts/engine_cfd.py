"""
engine_cfd.py — نفس قواعد مشروعنا، لكن على المؤشرات والسلع والفوركس،
بتكاليف CFD حقيقية معلنة (وليس صفراً).

القواعد كما هي، بلا أي تغيير:
    EMA200 · MACD(12,26,9) · ATR(14) · وقف 1.5×ATR · متحرك 3.0×ATR · مخاطرة 0.8%

التكاليف (من جدول PROP_FIRM_RULES_2026.md في المستودع، نفس الأرقام):
    spread 1.5 نقطة · انزلاق 1.0 نقطة · عمولة 5 لكل لوت · قيمة النقطة 10

الفرق الوحيد عن النسخة الكريبتو: الأداة تُقاس بوحدة R (مخاطرة معيارية)،
ويُطبَق «تكلفة الدوران» كنسبة ثابتة من المخاطرة تقديراً للسبريد.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "ema": 200, "macd": [12, 26, 9], "atr": 14,
    "sl_atr": 1.5, "trail_atr": 3.0, "risk_pct": 0.008,
    "spread_pts": 1.5, "slippage_pts": 1.0, "commission_per_lot": 5.0, "point_value": 10.0,
}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["EMA200"] = c.ewm(span=RULES["ema"], adjust=False).mean()
    df["MACD"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean())
    df["SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([(df.High - df.Low).abs(), (df.High - pc).abs(), (df.Low - pc).abs()],
                   axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / RULES["atr"], adjust=False).mean()
    return df


def backtest(df: pd.DataFrame, market: str, cost_R: float = 0.03) -> list[dict]:
    """cost_R: تكلفة الدوران بالكامل كنسبة من المخاطرة (1R).
    السبريد 1.5 + انزلاق 1.0 نقطة على وقف 1.5×ATR ⇒ ~2-4% من R في المؤشرات.
    الافتراضي 3% مُعلن، ونُجري حساسية عليه في study."""
    d = df.reset_index(drop=True)
    o, h, l, c = (d[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    atr, ema, macd, sig = (d[x].to_numpy(float) for x in ("ATR", "EMA200", "MACD", "SIG"))
    dates = d["DateTime"].to_numpy()

    ms, ss = pd.Series(macd).shift(1).to_numpy(), pd.Series(sig).shift(1).to_numpy()
    long_sig = (c > ema) & (macd > sig) & (ms <= ss)
    short_sig = (c < ema) & (macd < sig) & (ms >= ss)
    ready = ~np.isnan(atr) & ~np.isnan(ema)

    trades, pos = [], None
    for i in range(1, len(d)):
        if not ready[i] or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if pos is not None:
            prev_c = c[i - 1]
            if pos["dir"] == "long":
                pos["sl"] = max(pos["sl"], prev_c - RULES["trail_atr"] * atr[i])
                hit = l[i] <= pos["sl"]
            else:
                pos["sl"] = min(pos["sl"], prev_c + RULES["trail_atr"] * atr[i])
                hit = h[i] >= pos["sl"]
            if hit:
                exit_px = pos["sl"]
                gross = (exit_px - pos["entry"]) * pos["sign"]
                r = gross / pos["stop_dist"] - cost_R
                trades.append({"market": market, "dir": pos["dir"],
                               "entry_date": str(pd.Timestamp(pos["entry"])),
                               "exit_date": str(pd.Timestamp(dates[i])),
                               "r": float(r), "entry": pos["entry"], "exit": exit_px})
                pos = None
            else:
                continue
        if pos is None:
            side = "long" if long_sig[i - 1] else ("short" if short_sig[i - 1] else None)
            if side is None:
                continue
            entry, sign = o[i], (1 if long_sig[i - 1] else -1)
            stop_dist = RULES["sl_atr"] * atr[i]
            if stop_dist <= 0:
                continue
            pos = {"dir": side, "sign": sign, "entry": entry,
                   "entry_ts": dates[i], "sl": entry - sign * stop_dist, "stop_dist": stop_dist}
    # تحويل التاريخ
    for t in trades:
        t["entry_date"] = t.pop("entry_date")
    return trades
