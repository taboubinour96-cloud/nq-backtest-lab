"""
run_v2.py — التشغيل النهائي بعد إصلاح ثغرتَي الانتقاء.

الثغرة 1 (أصلحها study_honest.py): كان ترتيب العملات بحجم ملف 2026-09
         (معيار مستقبلي). الآن الترتيب بسيولة **فترة التصميم** فقط.
الثغرة 2 (أصلحها classic_universe.py): كانت الأدوات العشر مختارة بعد رؤية
         أن الذهب والفضة الأفضل. الآن القائمة = كل ما تتيحه شركات التمويل
         من ثلاث عائلات (20 أداة) بلا انتقاء أداء.

يُنتج result_v2.json — وهو الملف الذي يُقارَن به عمل Arena 1994 المستقل.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# البحث عن البيانات في: مجلد السكربت، أو ../data، أو ./data
DATA_DIR = next((p for p in (HERE, HERE.parent / "data", HERE / "data",
                             HERE.parent.parent / "data") if (p / "universe_final.csv").is_file()),
                HERE)
sys.path.insert(0, str(HERE))
import data_free as D
import engine as E
import engine_cfd as C

CUT = "2024-10-14"
CUTS = ["2021-01-01", "2022-07-01", "2023-07-01", "2024-10-14", "2025-06-01"]


def load_pool(n: int) -> dict[str, pd.DataFrame]:
    u = pd.read_csv(DATA_DIR / "universe_final.csv")
    out = {}
    for s in list(u.symbol[:n]):
        p = D.CACHE / f"{s}_1d.csv"
        if p.is_file():
            out[s] = pd.read_csv(p, parse_dates=["DateTime"])
    return out


def classic_trades() -> tuple[list[dict], pd.DataFrame]:
    f = DATA_DIR / "trades_classic.csv"
    if f.is_file():
        df = pd.read_csv(f)
        return df.to_dict("records"), pd.read_csv(DATA_DIR / "classic_universe.csv")
    raise SystemExit("شغّل classic_universe.py أولاً")


def econ(tr: list[dict], cut: str = CUT) -> dict:
    j = E.judge(tr, cut)
    b = j["blind"]
    dd = abs(b.get("max_dd_R") or 0)
    risk = 0.10 / dd * 100 if dd else float("nan")
    span = (pd.Timestamp(max(t["exit_date"] for t in tr)) - pd.Timestamp(cut)).days / 365.25
    tpm = b["n"] / (span * 12) if span > 0 else float("nan")
    monthly = b["expectancy_R"] * tpm * risk / 100 * 100 if dd else float("nan")
    return {"judge": j, "trades_per_month": round(float(tpm), 2),
            "max_risk_pct": round(float(risk), 3),
            "expected_monthly_pct": round(float(monthly), 4),
            "months_to_8pct": round(8 / monthly, 1) if monthly else None,
            "income_25k": round(25000 * monthly / 100 * 0.8),
            "income_200k": round(200_000 * monthly / 100 * 0.8)}


def main() -> int:
    out: dict = {"cut": CUT, "cut_dates_robustness": CUTS, "rules": E.RULES,
                 "rules_hash": E.rules_hash()}
    classic, cu = classic_trades()
    print(f"أدوات كلاسيكية: {len(cu)} · صفقاتها {len(classic)}")

    crypto_sets = {n: load_pool(n) for n in (6, 12, 24, 48)}
    print("أحجام كون الكريبتو المتاحة: " + ", ".join(f"{n}={len(v)}" for n, v in crypto_sets.items()))

    boxes: dict[str, list[dict]] = {}
    for n, data in crypto_sets.items():
        tr = E.run_universe(data, 100.0)
        boxes[f"crypto{n}"] = tr
    boxes["classic20"] = classic

    for label in ("crypto6", "crypto12", "crypto24", "crypto48"):
        if label in boxes:
            boxes[f"{label}+classic"] = sorted(boxes[label] + classic,
                                               key=lambda t: t["exit_date"])

    results = {}
    print(f"\n{'المحفظة':18s} {'صفقات':>7} {'PF':>6} {'R/صفقة':>8} {'DD(R)':>8} "
          f"{'مخاطرة%':>8} {'شهري%':>7} {'200k$':>8} بوابات")
    for k, tr in boxes.items():
        e = econ(tr)
        results[k] = e
        b = e["judge"]["blind"]
        g = sum(1 for v in e["judge"]["gate"].values() if v)
        print(f"{k:18s} {b['n']:7d} {b['pf']:6.2f} {b['expectancy_R']:+8.3f} "
              f"{b['max_dd_R']:8.1f} {e['max_risk_pct']:8.3f} {e['expected_monthly_pct']:7.3f} "
              f"{e['income_200k']:8,} {g}/4")
    out["portfolios"] = results

    # ---- الصمود على خمسة تواريخ، للمحفظة الموصى بها ----
    best = "crypto24+classic" if "crypto24+classic" in boxes else "classic20"
    robust = {}
    for cut in CUTS:
        j = E.judge(boxes[best], cut)["blind"]
        robust[cut] = {"n": j["n"], "pf": j["pf"], "expectancy_R": j["expectancy_R"],
                       "max_dd_R": j["max_dd_R"]}
    out["robustness"] = {"portfolio": best, "by_cut": robust}
    print(f"\nالصمود ({best}):")
    for k, v in robust.items():
        print(f"   قطع {k}: n={v['n']:4d} PF {v['pf']:5.2f} {v['expectancy_R']:+.3f}R DD {v['max_dd_R']:7.1f}R")

    # ---- حساسية التكاليف ----
    costs = {}
    for mult in (1.0, 2.0, 3.0):
        tr = []
        for s, df in crypto_sets[24].items():
            for x in E.backtest_market(E.add_indicators(df), s, 100.0):
                cf = 2 * 0.00065 * x["notional"] / x["risk"] if x["risk"] else 0
                x["r"] -= cf * (mult - 1)
                tr.append(x)
        for t in classic:
            y = dict(t)
            y["r"] = t["r"] - 0.03 * (mult - 1)
            tr.append(y)
        tr.sort(key=lambda t: t["exit_date"])
        costs[f"x{mult}"] = E.judge(tr, CUT)["blind"]
    out["cost_stress"] = costs
    print("\nالتكاليف: " + " · ".join(f"×{k[1:]} PF {v['pf']:.3f}" for k, v in costs.items()))

    # ---- الارتباط ----
    df = pd.DataFrame([t for t in boxes[best] if t["exit_date"] > CUT])
    df["month"] = pd.to_datetime(df.exit_date).dt.to_period("M")
    piv = df.pivot_table(index="month", columns="market", values="r", aggfunc="sum").fillna(0)
    corr = piv.corr().values
    ev = np.linalg.eigvalsh(np.nan_to_num(corr))
    ev = ev[ev > 0]
    eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    out["correlation"] = {
        "avg_pairwise": round(float(np.nanmean(corr[np.triu_indices_from(corr, 1)])), 3),
        "effective_bets": round(eff, 1), "markets": int(piv.shape[1]),
        "max_pair": None}
    pairs = [(corr[i, j], piv.columns[i], piv.columns[j])
             for i in range(len(corr)) for j in range(i + 1, len(corr)) if not np.isnan(corr[i, j])]
    if pairs:
        v, a, b = max(pairs)
        out["correlation"]["max_pair"] = {"r": round(float(v), 3), "a": a, "b": b}
    print(f"الارتباط: متوسط {out['correlation']['avg_pairwise']} · "
          f"رهانات فعّالة {eff:.1f}/{piv.shape[1]} · أعلى زوج {out['correlation']['max_pair']}")

    # ---- كل شيء في ملف واحد ----
    out["universes"] = {
        "crypto_pool": list(crypto_sets[24].keys()),
        "classic_pool": list(cu.symbol),
        "crypto_selection": json.loads((DATA_DIR / "universe_meta.json").read_text()),
        "classic_selection": json.loads((DATA_DIR / "classic_meta.json").read_text()),
    }
    (HERE / "result_v2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print("\n✅ result_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
