"""
study_yahoo_crypto.py — الكريبتو بتاريخ أطول عبر Yahoo (اقتراح Arena 1994).

المعيار (غير أدائي، مُعلَن مسبقاً):
    كل عملة لها سجل يومي على Yahoo **≥ 8 سنوات** (بدأت قبل 2018-01-01).
    لا فرز بالربحية ولا بالحجم. القائمة كاملة في crypto_yahoo_universe.csv،
    بما فيها العملات التي ستصبح خاسرة.

الهدف المعلن: اختبار الفرضية التي كتبتها في تقريري —
    «ضعف الكريبتو قد يكون أثراً لطول البيانات (6 سنوات) لا لضعف الأصل».
هذا السكربت يجيب عليها: نفس القواعد، تاريخ أطول، نفس التقسيم.

الفروق عن Binance Vision (مصرَّح بها):
    • سعر فوري spot لا عقد دائم ⇒ لا تمويل ولا رافعة.
    • الحجم ليس حجم عقود ⇒ لا يُستخدم لترتيب السيولة.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import data_yahoo_crypto as YC
import engine as E

CUT = "2024-10-14"
CUTS = ["2021-01-01", "2022-07-01", "2023-07-01", "2024-10-14", "2025-06-01"]

# المعيار: سجل ≥ 8 سنوات (بدأ قبل 2018-01-01) — قائمة ثابتة من فحص Yahoo
LONG_HISTORY = ["BTC", "ETH", "XRP", "BCH", "LTC", "ADA", "DOGE", "BNB", "LINK",
                "ETC", "XLM", "TRX", "NEO", "DASH", "ZEC", "EOS", "XTZ", "OMG",
                "BAT", "REP", "ZRX", "MKR"]
REPO6 = ["BTC", "ETH", "XRP", "BNB", "DOGE", "SOL"]


def load(coins: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for c in coins:
        try:
            df = YC.download(f"{c}-USD")
            if len(df) > 250:
                out[f"{c}USDT"] = df
        except Exception as e:
            print(f"  {c} فشل: {str(e)[:40]}")
    return out


def report(tr: list[dict], label: str) -> dict:
    j = E.judge(tr, CUT)
    b = j["blind"]
    dd = max(abs(E.judge(tr, c)["blind"]["max_dd_R"] or 0) for c in CUTS)
    risk = 0.10 / dd * 100 if dd else float("nan")
    span = (pd.Timestamp(max(t["exit_date"] for t in tr)) - pd.Timestamp(CUT)).days / 365.25
    tpm = b["n"] / (span * 12) if span > 0 else float("nan")
    monthly = b["expectancy_R"] * tpm * risk / 100 * 100 if dd else float("nan")
    row = {"label": label, "n": b["n"], "pf": b["pf"], "exp_R": b["expectancy_R"],
           "dd_R": b["max_dd_R"], "worst_dd": round(dd, 1), "risk_pct": round(risk, 3),
           "monthly_pct": round(float(monthly), 3), "income_25k": round(25000 * monthly / 100 * 0.8),
           "income_200k": round(200_000 * monthly / 100 * 0.8),
           "gates": int(sum(j["gate"].values())), "best3": b["stress_remove_best3_R"],
           "robust": {c: E.judge(tr, c)["blind"]["pf"] for c in CUTS}}
    print(f"{label:28s} n={row['n']:4d} PF {row['pf']:.2f} {row['exp_R']:+.3f}R "
          f"DD {row['dd_R']:6.1f}R مخاطرة {row['risk_pct']:.3f}% {row['monthly_pct']:+.3f}%/شهر "
          f"200k→{row['income_200k']:,}$ بوابات {row['gates']}/4")
    return row


def main() -> int:
    fam = dict(zip(*[pd.read_csv(HERE / "classic_universe.csv")[c]
                     for c in ("symbol", "family")]))
    cl12 = sorted([t for t in pd.read_csv(HERE / "trades_classic.csv").to_dict("records")
                   if fam[t["market"]] in ("index", "commodity")], key=lambda t: t["exit_date"])

    d6 = load(REPO6)
    d22 = load(LONG_HISTORY)
    pd.DataFrame([{"symbol": f"{c}-USD", "first_bar": str(load([c])[f'{c}USDT'].DateTime.min().date())
                   if f"{c}USDT" in load([c]) else None} for c in LONG_HISTORY]
                 ).to_csv(HERE / "crypto_yahoo_universe.csv", index=False)

    out = {}
    print("=== الكريبتو من Yahoo، نفس قواعد المستودع ===")
    t6 = E.run_universe(d6, 100.0)
    t22 = E.run_universe(d22, 100.0)
    out["crypto6_yahoo"] = report(t6, "R6 · Yahoo (spot)")
    out["crypto22_yahoo"] = report(t22, "R22 · Yahoo (سجل ≥8 سنوات)")
    out["R6_yahoo+C12"] = report(sorted(t6 + cl12, key=lambda t: t["exit_date"]), "R6 + C12")
    out["R22_yahoo+C12"] = report(sorted(t22 + cl12, key=lambda t: t["exit_date"]), "R22 + C12")

    print("\nالصمود (R22+C12):")
    for c, v in out["R22_yahoo+C12"]["robust"].items():
        print(f"   {c}: PF {v:.2f}")

    print("\nمقارنة المصادر لنفس الرموز الستة (PF خارج العينة):")
    print("   Bybit (المستودع) 1.286 · Binance Vision 1.211 · Yahoo spot 1.239")

    (HERE / "result_v4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n✅ result_v4.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
