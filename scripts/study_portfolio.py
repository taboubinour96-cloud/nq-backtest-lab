"""
study_portfolio.py — الاختبار الفاصل: هل الحلّ «أسواق أكثر»، أم «فئات أصول مختلفة»؟

أربع محافظ، نفس القواعد، نفس التقسيم الزمني (2024-10-14)، نفس التكاليف:
    A) كريبتو 6 (ما لدينا اليوم)
    B) كريبتو 24 (توسّع أفقي — ما اقترحه تقريري الأول)
    C) مؤشرات+سلع+فوركس 10 (فئة مختلفة، 22 سنة)
    D) مختلطة: كريبتو 24 + العشر أدوات (فئات متعددة)

لكل محفظة نحسب:
    PF خارج العينة · متوسط R · أقصى تراجع بوحدة R · المخاطرة القصوى المسموحة
    لحدّ 10% · العائد الشهري المتوقع · الدخل على حساب ممول 25k و 200k

مبدأ الأمانة: المخاطرة القصوى تُشتق من التراجع **المقاس**، لا من افتراض.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_free as D
import data_yahoo as Y
import engine as E
import engine_cfd as C

OUT = Path(__file__).parent
CUT = "2024-10-14"


def stats(trades: list[dict]) -> dict:
    return E.judge(trades, CUT)


def money(tr: list[dict], label: str) -> dict:
    j = stats(tr)
    b = j["blind"]
    exp, dd = b.get("expectancy_R"), abs(b.get("max_dd_R") or 0)
    span = (pd.Timestamp(max(t["exit_date"] for t in tr)) - pd.Timestamp(CUT)).days / 365.25
    tpm = b["n"] / (span * 12) if span > 0 else float("nan")
    # المخاطرة القصوى التي تُبقي التراجع التاريخي داخل 10% من الحساب
    risk_max = (0.10 / dd * 100) if dd else float("nan")
    monthly_pct = exp * tpm * (risk_max / 100) * 100 if dd else float("nan")
    return {
        "label": label,
        "trades_total": len(tr),
        "blind_n": b["n"], "blind_pf": b["pf"], "blind_expectancy_R": exp,
        "blind_max_dd_R": -dd, "trades_per_month": round(tpm, 2),
        "max_risk_pct_per_trade": round(risk_max, 3),
        "expected_monthly_pct": round(monthly_pct, 3),
        "months_to_8pct": round(8 / monthly_pct, 1) if monthly_pct else None,
        "income_25k_80pct": round(25000 * monthly_pct / 100 * 0.8, 0) if monthly_pct else 0,
        "income_200k_80pct": round(200_000 * monthly_pct / 100 * 0.8, 0) if monthly_pct else 0,
        "gate": j["gate"], "verdict": j["verdict"],
        "survives_best3": b.get("stress_remove_best3_R"),
        "win_rate": b.get("win_rate_pct"),
    }


def main() -> int:
    liq = pd.read_csv(OUT / "universe_liquidity.csv")
    pool = list(liq.symbol[:24])
    crypto = {}
    for s in pool:
        f = D.CACHE / f"{s}_1d.csv"
        if f.is_file():
            crypto[s] = pd.read_csv(f, parse_dates=["DateTime"])

    idx = Y.download_universe(verbose=False)
    print(f"كريبتو متاح: {len(crypto)} · أدوات أخرى: {len(idx)}")

    def run_crypto(syms):
        return E.run_universe({s: crypto[s] for s in syms}, equity0=100.0)

    def run_idx():
        out = []
        for s, df in idx.items():
            out += C.backtest(C.add_indicators(df), s, cost_R=0.03)
        out.sort(key=lambda t: t["exit_date"])
        return out

    six = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    six = [s for s in six if s in crypto]

    boxes = {
        "A) كريبتو 6 (اليوم)": run_crypto(six),
        "B) كريبتو 24 (توسّع أفقي)": run_crypto(pool),
        "C) مؤشرات/سلع/فوركس 10": run_idx(),
    }
    mixed = boxes["B) كريبتو 24 (توسّع أفقي)"] + boxes["C) مؤشرات/سلع/فوركس 10"]
    mixed.sort(key=lambda t: t["exit_date"])
    boxes["D) مختلطة 24+10"] = mixed

    rows = []
    for label, tr in boxes.items():
        m = money(tr, label)
        rows.append(m)
        print(f"\n{'='*84}\n{label}")
        print(f"  صفقات كلية {m['trades_total']} · مغلقة {m['blind_n']} · {m['trades_per_month']}/شهر")
        print(f"  blind: PF {m['blind_pf']} · {m['blind_expectancy_R']:+.3f}R/صفقة · تراجع {m['blind_max_dd_R']:.1f}R · فوز {m['win_rate']}%")
        print(f"  أقصى مخاطرة لنجاة 10%: {m['max_risk_pct_per_trade']}% → "
              f"{m['expected_monthly_pct']}%/شهر · 8% في {m['months_to_8pct']} شهر")
        print(f"  الدخل: 25k → {m['income_25k_80pct']:,.0f}$ · 200k → {m['income_200k_80pct']:,.0f}$")
        print(f"  البوابات: {m['gate']}")
        print(f"  الحكم: {m['verdict']}")

    pd.DataFrame(rows).to_csv(OUT / "portfolio_comparison.csv", index=False)
    (OUT / "portfolio_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\nحُفظ: portfolio_comparison.csv · portfolio_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
