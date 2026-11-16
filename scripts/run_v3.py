"""
run_v3.py — الأرقام النهائية بعد إصلاح ثغرتَي الانتقاء + تشخيص نوع الأصل.

المنهج:
  1) ثلاثة تعريفات كون، كلها قابلة للدفاع:
       R6  = رموز المستودع الستة كما هي (بلا أي اختيار مني)
       C12 = 12 أداة كلاسيكية (7 مؤشرات + 5 سلع) — الفوركس مستبعد لأنه خاسر صريح
       H1  = أعلى 5 عملات بالسيولة (رتبة تصميم) + C12
  2) المخاطرة لكل صفقة تُشتق من **أسوأ تراجع عبر النوافذ الخمس** (لا من أفضل نافذة).
  3) كل رقم يُطبع مع عدد صفقاته، ولا ندّعي شيئاً بلا عيّنة كافية.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import data_free as D
import engine as E

CUTS = ["2021-01-01", "2022-07-01", "2023-07-01", "2024-10-14", "2025-06-01"]
MAIN_CUT = "2024-10-14"


def load(syms):
    return {s: pd.read_csv(D.CACHE / f"{s}_1d.csv", parse_dates=["DateTime"])
            for s in syms if (D.CACHE / f"{s}_1d.csv").is_file()}


def worst_dd(tr):
    return max(abs(E.judge(tr, c)["blind"]["max_dd_R"] or 0) for c in CUTS)


def evaluate(tr, name):
    j = E.judge(tr, MAIN_CUT)
    b = j["blind"]
    dd = worst_dd(tr)
    risk = 0.10 / dd * 100 if dd else float("nan")
    span = (pd.Timestamp(max(t["exit_date"] for t in tr)) - pd.Timestamp(MAIN_CUT)).days / 365.25
    tpm = b["n"] / (span * 12) if span > 0 else float("nan")
    monthly = b["expectancy_R"] * tpm * risk / 100 * 100 if dd else float("nan")
    return {"name": name, "blind_n": b["n"], "blind_pf": b["pf"],
            "blind_exp_R": b["expectancy_R"], "blind_dd_R": b["max_dd_R"],
            "worst_dd_R_allwindows": round(dd, 1),
            "trades_per_month": round(float(tpm), 2),
            "max_risk_pct": round(float(risk), 3),
            "expected_monthly_pct": round(float(monthly), 3),
            "months_to_8pct": round(8 / monthly, 1) if monthly else None,
            "income_25k": round(25000 * monthly / 100 * 0.8),
            "income_200k": round(200_000 * monthly / 100 * 0.8),
            "gates": j["gate"], "gates_passed": int(sum(j["gate"].values())),
            "survives_best3_R": b["stress_remove_best3_R"],
            "win_rate": b["win_rate_pct"],
            "robustness": {c: E.judge(tr, c)["blind"]["pf"] for c in CUTS}}


def main() -> int:
    fam = dict(zip(*[pd.read_csv(HERE / "classic_universe.csv")[c]
                     for c in ("symbol", "family")]))
    cl_all = pd.read_csv(HERE / "trades_classic.csv").to_dict("records")
    cl12 = sorted([t for t in cl_all if fam[t["market"]] in ("index", "commodity")],
                  key=lambda t: t["exit_date"])
    fx = sorted([t for t in cl_all if fam[t["market"]] == "fx"], key=lambda t: t["exit_date"])

    u = pd.read_csv(HERE / "universe_final.csv")
    top5 = list(u.symbol[:5])
    repo6 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]

    t_repo6 = E.run_universe(load(repo6), 100.0)
    t_top5 = E.run_universe(load(top5), 100.0)
    t_c24 = E.run_universe(load(list(u.symbol[:24])), 100.0)

    configs = {
        "R6 · رموز المستودع وحدها": t_repo6,
        "C12 · كلاسيكي بلا فوركس": cl12,
        "FX8 · فوركس وحده (لماذا استُبعد)": fx,
        "R6+C12 · بلا أي اختيار مني": sorted(t_repo6 + cl12, key=lambda t: t["exit_date"]),
        "H1 · أعلى5 كريبتو + C12": sorted(t_top5 + cl12, key=lambda t: t["exit_date"]),
        "C24 + C12 · توسّع أفقي": sorted(t_c24 + cl12, key=lambda t: t["exit_date"]),
    }

    out = {}
    print(f"{'التعريف':34s} {'صفقات':>6} {'PF':>5} {'R/صف':>7} {'أسوأDD':>7} "
          f"{'مخاطرة':>7} {'شهري%':>7} {'25k$':>7} {'200k$':>8} بوابات")
    for k, tr in configs.items():
        e = evaluate(tr, k)
        out[k] = e
        print(f"{k:34s} {e['blind_n']:6d} {e['blind_pf']:5.2f} {e['blind_exp_R']:+7.3f} "
              f"{e['worst_dd_R_allwindows']:7.1f} {e['max_risk_pct']:7.3f} "
              f"{e['expected_monthly_pct']:7.3f} {e['income_25k']:7,} {e['income_200k']:8,} "
              f"{e['gates_passed']}/4")

    print("\nالصمود (نسبة الربح بعد كل تاريخ قطع):")
    for k in ("R6+C12 · بلا أي اختيار مني", "H1 · أعلى5 كريبتو + C12"):
        print(f"  {k}: " + " · ".join(f"{c[:7]}→{v:.2f}" for c, v in out[k]["robustness"].items()))

    diag = {
        "index_blind": {"n": int(len([t for t in cl12 if fam[t['market']] == 'index' and t['exit_date'] > MAIN_CUT])),
                        "mean_R": round(float(np.mean([t['r'] for t in cl12 if fam[t['market']] == 'index' and t['exit_date'] > MAIN_CUT])), 3)},
        "commodity_blind": {"n": int(len([t for t in cl12 if fam[t['market']] == 'commodity' and t['exit_date'] > MAIN_CUT])),
                            "mean_R": round(float(np.mean([t['r'] for t in cl12 if fam[t['market']] == 'commodity' and t['exit_date'] > MAIN_CUT])), 3)},
        "fx_blind": {"n": int(len([t for t in fx if t['exit_date'] > MAIN_CUT])),
                     "mean_R": round(float(np.mean([t['r'] for t in fx if t['exit_date'] > MAIN_CUT])), 3)},
    }
    out["diagnosis_by_asset_class"] = diag
    print("\nالتشخيص حسب نوع الأصل (خارج العينة):")
    for k, v in diag.items():
        print(f"   {k:20s} {v['n']:4d} صفقة · {v['mean_R']:+.3f}R")

    rk = u.set_index("symbol")["rank"].to_dict()
    by_rank = {}
    for lo, hi, lab in [(1, 5, "1-5"), (6, 12, "6-12"), (13, 24, "13-24"), (25, 48, "25-48")]:
        syms = [s for s in u.symbol[:48] if lo <= rk[s] <= hi]
        tt = [x for s in syms if (D.CACHE / f"{s}_1d.csv").is_file()
              for x in E.backtest_market(E.add_indicators(load([s])[s]), s, 100.0)
              if x["exit_date"] > MAIN_CUT]
        if tt:
            by_rank[lab] = {"markets": len(syms), "trades": len(tt),
                            "mean_R": round(float(np.mean([x["r"] for x in tt])), 3)}
    out["diagnosis_by_liquidity_rank"] = by_rank
    print("   ترتيب السيولة داخل الكريبتو:")
    for k, v in by_rank.items():
        print(f"   {k:8s} أسواق {v['markets']:2d} · {v['trades']:4d} صفقة · {v['mean_R']:+.3f}R")

    (HERE / "result_v3.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n✅ result_v3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
