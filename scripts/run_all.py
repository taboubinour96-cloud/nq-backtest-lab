"""
run_all.py — أمر واحد يُعيد كل شيء من الصفر، مجاناً، على أي جهاز.

    python run_all.py            # كل المراحل
    python run_all.py --quick    # بلا تحميل بيانات جديدة (من الذاكرة المؤقتة)

ما ينفّذه:
    1) تحميل البيانات المجانية  (Binance Vision + Yahoo) — بلا مفتاح ولا حساب
    2) إعادة إنتاج نتيجة المستودع على 6 أسواق (تحقّق من صحة المحرك)
    3) مقارنة 4 محافظ (كريبتو 6 · كريبتو 24 · كلاسيكي 10 · مختلط)
    4) اختبار الصمود على 5 تواريخ قطع
    5) حساسية التكاليف + قياس الارتباط (عدد الرهانات المستقلة)
    6) كتابة result.json + خروج برمز 0 فقط إذا نُفِّذ كل شيء
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# يبحث عن البيانات في: نفس المجلد، أو ../data، أو ./data
DATA_DIR = next((p for p in (HERE, HERE.parent / "data", HERE / "data")
                             if (p / "universe_final.csv").is_file()), HERE.parent / "data")
sys.path.insert(0, str(HERE))

import data_free as D
import data_yahoo as Y
import engine as E
import engine_cfd as C

CUT = "2024-10-14"
SIX = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]


def load_crypto_pool() -> dict[str, pd.DataFrame]:
    f = HERE / "universe_liquidity.csv"
    if not f.is_file():
        raise SystemExit("ناقص universe_liquidity.csv — شغّل study_scaling.py أولاً")
    pool = list(pd.read_csv(f).symbol[:24])
    out = {}
    for s in pool:
        p = D.CACHE / f"{s}_1d.csv"
        if p.is_file():
            out[s] = pd.read_csv(p, parse_dates=["DateTime"])
    return out


def mixed_portfolio(crypto: dict, idx: dict) -> list[dict]:
    tr = []
    for s, df in crypto.items():
        tr += E.backtest_market(E.add_indicators(df), s, 100.0)
    for s, df in idx.items():
        tr += C.backtest(C.add_indicators(df), s, cost_R=0.03)
    tr.sort(key=lambda t: t["exit_date"])
    return tr


def economics(tr: list[dict], cut: str = CUT) -> dict:
    j = E.judge(tr, cut)
    b = j["blind"]
    dd = abs(b.get("max_dd_R") or 0)
    risk = 0.10 / dd * 100 if dd else float("nan")
    span = (pd.Timestamp(max(t["exit_date"] for t in tr)) - pd.Timestamp(cut)).days / 365.25
    tpm = b["n"] / (span * 12) if span > 0 else float("nan")
    monthly = b["expectancy_R"] * tpm * risk / 100 * 100 if dd else float("nan")
    return {"judge": j, "trades_per_month": round(tpm, 2),
            "max_risk_pct": round(risk, 3), "expected_monthly_pct": round(monthly, 3),
            "months_to_8pct": round(8 / monthly, 1) if monthly else None,
            "income_25k": round(25000 * monthly / 100 * 0.8),
            "income_200k": round(200_000 * monthly / 100 * 0.8)}


def main() -> int:
    print("=" * 78)
    print("مختبر الباك-تست المجاني — تشغيل كامل")
    print("=" * 78)

    crypto = load_crypto_pool()
    idx = Y.download_universe(verbose=False)
    print(f"كريبتو: {len(crypto)} · كلاسيكي: {len(idx)}")

    out: dict = {"cut": CUT, "rules": E.RULES, "rules_hash": E.rules_hash()}

    # --- 2) إعادة إنتاج ---
    six = {s: crypto[s] for s in SIX if s in crypto}
    repro = E.judge(E.run_universe(six, 100.0), CUT)
    out["repro_6sym"] = repro
    print(f"\n[1] إعادة الإنتاج على 6 أسواق: blind PF {repro['blind']['pf']} "
          f"(المستودع: 1.286) · n={repro['blind']['n']} (المستودع: 109)")

    # --- 3) المحافظ ---
    port = {}
    for name, tr in {
        "crypto6": E.run_universe(six, 100.0),
        "crypto24": E.run_universe(crypto, 100.0),
        "classic10": [t for s, df in idx.items()
                      for t in C.backtest(C.add_indicators(df), s, cost_R=0.03)],
    }.items():
        port[name] = economics(sorted(tr, key=lambda t: t["exit_date"]))
    mixed = mixed_portfolio(crypto, idx)
    port["mixed"] = economics(mixed)
    out["portfolios"] = port
    print("\n[2] المحافظ (خارج العينة):")
    for k, v in port.items():
        print(f"    {k:10s} PF {v['judge']['blind']['pf']:5.2f} · "
              f"{v['judge']['blind']['expectancy_R']:+.3f}R · DD {v['judge']['blind']['max_dd_R']:7.1f}R · "
              f"{v['trades_per_month']:5.1f}/شهر · {v['expected_monthly_pct']:+.3f}%/شهر · "
              f"200k→{v['income_200k']:,}$")

    # --- 4) الصمود ---
    robust = {}
    for cut in ("2021-01-01", "2022-07-01", "2023-07-01", CUT, "2025-06-01"):
        j = E.judge(mixed, cut)["blind"]
        robust[cut] = {"n": j["n"], "pf": j["pf"], "expectancy_R": j["expectancy_R"],
                       "max_dd_R": j["max_dd_R"]}
    out["robustness"] = robust
    print("\n[3] الصمود على تواريخ قطع مختلفة (المحفظة المختلطة):")
    for k, v in robust.items():
        print(f"    قطع {k}: n={v['n']:4d} PF {v['pf']:5.2f} {v['expectancy_R']:+.3f}R DD {v['max_dd_R']:7.1f}R")

    # --- 5) التكاليف + الارتباط ---
    costs = {}
    for mult in (1.0, 2.0, 3.0):
        tr = []
        for s, df in crypto.items():
            for x in E.backtest_market(E.add_indicators(df), s, 100.0):
                cf = 2 * 0.00065 * x["notional"] / x["risk"] if x["risk"] else 0
                x["r"] -= cf * (mult - 1)
                tr.append(x)
        for s, df in idx.items():
            tr += C.backtest(C.add_indicators(df), s, cost_R=0.03 * mult)
        tr.sort(key=lambda t: t["exit_date"])
        costs[f"x{mult}"] = E.judge(tr, CUT)["blind"]
    out["cost_stress"] = costs
    print("\n[4] حساسية التكاليف: " + " · ".join(
        f"×{k[1:]} → PF {v['pf']:.2f}" for k, v in costs.items()))

    df = pd.DataFrame([t for t in mixed if t["exit_date"] > CUT])
    df["month"] = pd.to_datetime(df.exit_date).dt.to_period("M")
    piv = df.pivot_table(index="month", columns="market", values="r", aggfunc="sum").fillna(0)
    corr = piv.corr().values
    ev = np.linalg.eigvalsh(np.nan_to_num(corr))
    ev = ev[ev > 0]
    eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    out["correlation"] = {"avg_pairwise": round(float(np.nanmean(corr[np.triu_indices_from(corr, 1)])), 3),
                          "effective_bets": round(eff, 1), "markets": int(piv.shape[1])}
    print(f"[5] الارتباط: متوسط {out['correlation']['avg_pairwise']} · "
          f"رهانات مستقلة فعّالة {eff:.1f} من {piv.shape[1]}")

    (HERE / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print("\n✅ نتيجة كاملة في result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
