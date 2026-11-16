"""
engine.py — محرّك باك-تست مجاني 100%، بنفس قواعد مشروعنا حرفياً.

القواعد منسوخة حرفياً من nq100_portfolio/trend_universe.py (لم أغيّر رقماً واحداً):
    EMA 200 فلتر اتجاه · MACD(12,26,9) تقاطع · ATR(14)
    وقف 1.5×ATR · وقف متحرك 3.0×ATR (يتحرك بإغلاق الأمس فقط)
    مخاطرة 0.8% لكل صفقة · تكاليف taker 0.055% + انزلاق 0.01% لكل جهة
    إشارة على إغلاق الشمعة i-1 · تنفيذ على افتتاح الشمعة i  (بلا تسريب مستقبل)

الفروق الوحيدة (وهي شفافية وليست تحسيناً):
    1) عدد الأسواق: أي عدد نريد (بدل 6).
    2) قياس مجمّع بوحدة R (وهو نفسه ما يفعله المستودع في kit()).
    3) النقد القابل للاستخدام يُحرَّر عند الإغلاق (المستودع يحجز الخطر للأبد —
       وهذا خطأ محاسبي يقلّل مركّب العائد، صححته لأنه يصبّ في صالح الدقة لا في
       صالح التفاؤل: زيادة عدد الصفقات لا زيادة ربح الصفقة).
    4) خيار سقف تعرّض (leverage cap) لحساب صغير: مجموع المخاطر المفتوحة ≤ سقف،
       لأن حساب 100$ لا يستطيع تحمّل 20 مركزاً في آن واحد.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RULES = {
    "ema": 200, "macd": [12, 26, 9], "atr": 14,
    "sl_atr": 1.5, "trail_atr": 3.0, "risk_pct": 0.008,
    "taker_per_side": 0.00055, "slippage_per_side": 0.0001,
    "entry": "signal on close of bar i-1, fill at open of bar i",
    "exit": "ATR stop 1.5 / trailing 3.0 (moved with previous close only)",
}


def rules_hash() -> str:
    return hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()[:16]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["EMA200"] = c.ewm(span=RULES["ema"], adjust=False).mean()
    e12 = c.ewm(span=RULES["macd"][0], adjust=False).mean()
    e26 = c.ewm(span=RULES["macd"][1], adjust=False).mean()
    df["MACD"] = e12 - e26
    df["SIG"] = df["MACD"].ewm(span=RULES["macd"][2], adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([(df["High"] - df["Low"]).abs(),
                    (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / RULES["atr"], adjust=False).mean()
    return df


@dataclass
class Position:
    market: str
    direction: str
    sign: int
    entry: float
    sl: float
    qty: float
    stop_dist: float
    entry_i: int
    entry_date: pd.Timestamp


def backtest_market(df: pd.DataFrame, market: str, equity0: float = 100.0,
                    max_risk_open: float | None = None) -> list[dict]:
    """يرجع كل الصفقات المغلقة. max_risk_open = سقف مجموع المخاطر المفتوحة
    كنسبة من الرصيد (None = بلا سقف، كسلوك المستودع الأصلي)."""
    d = df.reset_index(drop=True)
    o, h, l, c = (d[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    atr, ema, macd, sig = (d[x].to_numpy(float) for x in ("ATR", "EMA200", "MACD", "SIG"))
    dates = d["DateTime"].to_numpy()

    above, below = c > ema, c < ema
    ms, ss = pd.Series(macd).shift(1).to_numpy(), pd.Series(sig).shift(1).to_numpy()
    long_sig = above & (macd > sig) & (ms <= ss)
    short_sig = below & (macd < sig) & (ms >= ss)
    ready = ~np.isnan(atr) & ~np.isnan(ema)

    equity = equity0
    trades: list[dict] = []
    pos: Position | None = None

    for i in range(1, len(d)):
        if not ready[i] or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        if pos is not None:
            prev_c = c[i - 1]
            if pos.direction == "long":
                pos.sl = max(pos.sl, prev_c - RULES["trail_atr"] * atr[i])
                hit = l[i] <= pos.sl
            else:
                pos.sl = min(pos.sl, prev_c + RULES["trail_atr"] * atr[i])
                hit = h[i] >= pos.sl
            if hit:
                exit_px = pos.sl
                slip = RULES["slippage_per_side"]
                exit_px = exit_px * (1 - slip) if pos.direction == "long" else exit_px * (1 + slip)
                gross = (exit_px - pos.entry) * pos.qty * pos.sign
                fee = (pos.entry * pos.qty + exit_px * pos.qty) * RULES["taker_per_side"]
                net = gross - fee
                risk_amount = pos.stop_dist * pos.qty
                equity += net
                trades.append({
                    "market": market, "dir": pos.direction,
                    "entry_date": str(pd.Timestamp(pos.entry_date)),
                    "exit_date": str(pd.Timestamp(dates[i])),
                    "entry": pos.entry, "exit": exit_px, "qty": pos.qty,
                    "net": net, "r": net / risk_amount if risk_amount else None,
                    "risk": risk_amount, "notional": pos.entry * pos.qty,
                    "bars": i - pos.entry_i,
                })
                pos = None
            else:
                continue

        if pos is None:
            side = "long" if long_sig[i - 1] else ("short" if short_sig[i - 1] else None)
            if side is None:
                continue
            entry = o[i]
            sign = 1 if side == "long" else -1
            stop_dist = RULES["sl_atr"] * atr[i]
            if stop_dist <= 0:
                continue
            risk_amount = equity * RULES["risk_pct"]
            qty = risk_amount / stop_dist
            if qty <= 0:
                continue
            if max_risk_open is not None:
                risk_frac = risk_amount / equity
                if risk_frac > max_risk_open:
                    continue
            pos = Position(market, side, sign, entry, entry - sign * stop_dist,
                           qty, stop_dist, i, pd.Timestamp(dates[i]))
    return trades


def run_universe(data: dict[str, pd.DataFrame], equity0: float = 100.0,
                 max_risk_open: float | None = None) -> list[dict]:
    out = []
    for sym, df in data.items():
        out.extend(backtest_market(add_indicators(df), sym, equity0, max_risk_open))
    out.sort(key=lambda t: (t["exit_date"], t["market"]))
    return out


# ------------------------------------------------------------------ الحكم
def judge(trades: list[dict], cut_date: str) -> dict:
    """نفس معايير backtest_discipline عندنا (وهي معايير الفيديو الذي اعتمده المالك)."""
    def block(ts: list[dict]) -> dict:
        if not ts:
            return {"n": 0}
        r = np.array([t["r"] for t in ts if t["r"] is not None], float)
        pos, neg = r[r > 0].sum(), -r[r < 0].sum()
        pf = pos / neg if neg > 0 else float("inf")
        eq = np.cumsum(r)
        dd_R = float(np.min(eq - np.maximum.accumulate(eq)))
        n = len(r)
        verdict = ("<30 مجرد فكرة" if n < 30 else "<100 مؤشر" if n < 100
                   else "<400 غير موثوق بعد" if n < 400 else "≥400 عيّنة يُعتمد عليها")
        best = np.sort(r)[::-1]
        return {
            "n": n,
            "pf": round(float(pf), 3),
            "win_rate_pct": round(float((r > 0).mean() * 100), 1),
            "expectancy_R": round(float(r.mean()), 4),
            "net_R": round(float(r.sum()), 1),
            "max_dd_R": round(dd_R, 1),
            "sample_verdict": verdict,
            "stress_remove_best2_R": round(float(r.sum() - best[:2].sum()), 2),
            "stress_remove_best3_R": round(float(r.sum() - best[:3].sum()), 2),
            "top3_concentration_pct": round(float(best[:3].sum() / r.sum() * 100), 1) if r.sum() > 0 else None,
        }

    design = [t for t in trades if t["exit_date"] <= cut_date]
    blind = [t for t in trades if t["exit_date"] > cut_date]
    d, b = block(design), block(blind)
    ok = (b.get("pf", 0) or 0) >= 1.3 and b.get("n", 0) >= 100 and (d.get("pf") or 0) > 0
    return {
        "cut_date": cut_date,
        "design": d, "blind": b,
        "blind_confirms": bool(ok),
        "verdict": ("PASS — الفترة المغلقة تؤكد التصميم" if ok else
                    "FAIL — الفترة المغلقة لم تؤكد التصميم"),
        "gate": {
            "blind_PF>=1.3": (b.get("pf", 0) or 0) >= 1.3,
            "blind_trades>=100": b.get("n", 0) >= 100,
            "blind_trades>=400": b.get("n", 0) >= 400,
            "survives_best3_removal": (b.get("stress_remove_best3_R") or 0) > 0,
        },
    }
