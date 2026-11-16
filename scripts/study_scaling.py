"""
study_scaling.py — السؤال الوحيد الذي يهم: هل التوسّع من 6 إلى عشرة أضعاف الأسواق
يُصلح الميزة، أم يوزّع الخسارة على أسواق أكثر؟

المنهج (بلا غش):
  1) الأهلية: كل رموز USDT على Binance Vision، نأخذ الأكثر سيولة فقط.
  2) الاختيار بالسيولة **خلال فترة التصميم وحدها** (لا نعرف المستقبل):
     ترتيب الرموز بمتوسط حجم التداول اليومي بالدولار قبل تاريخ القطع.
     ⇒ لا انتقاء على أساس النتيجة (لا survivorship ولا cherry-picking).
  3) تقسيم زمني واحد: design ≤ 2024-10-14 (نفس قطع المستودع) · blind بعده.
  4) نفس القواعد حرفياً (engine.py) لكل الأكوان.
  5) الحكم بنفس بواباتنا: blind PF ≥ 1.3 · صفقات ≥ 100 · ≥400 · يصمد لحذف أفضل 3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_free as D
import engine as E

OUT = Path(__file__).parent
CUT = "2024-10-14"           # نفس تاريخ قطع المستودع (trend_universe.json)
DESIGN_END = pd.Timestamp(CUT)
POOL_LIMIT = 110              # كم رمزاً نُنزّل (الأكثر سيولة)


def liquidity_probe() -> list[str]:
    """سيولة تقريبية: حجم ملف الشهر الأخير على S3 (بديل رخيص عن تحميل الكل)."""
    syms = [s for s in D.list_symbols() if s.endswith("USDT")]
    print(f"رموز USDT المتاحة: {len(syms)}")

    from concurrent.futures import ThreadPoolExecutor
    def size(s):
        try:
            r = D.S.head(D._url(s, "2026-09"), timeout=15, allow_redirects=True)
            return s, int(r.headers.get("Content-Length", 0)) if r.status_code == 200 else 0
        except Exception:
            return s, 0

    with ThreadPoolExecutor(16) as ex:
        sizes = dict(ex.map(size, syms))
    ranked = [s for s, n in sorted(sizes.items(), key=lambda kv: -kv[1]) if n > 0]
    print(f"منها لها بيانات الشهر الأخير: {len(ranked)}")
    return ranked[:POOL_LIMIT]


def main() -> int:
    pool = liquidity_probe()
    cache_n = sum(1 for s in pool if (D.CACHE / f"{s}_1d.csv").is_file())
    print(f"في الذاكرة المؤقتة: {cache_n}/{len(pool)} — أنزّل الباقي…")
    data = D.download_many(pool, workers=10, verbose=False)
    print(f"نُزّل: {len(data)} رمزاً")

    # ---- الترتيب بالسيولة داخل فترة التصميم فقط ----
    rows = []
    for s, df in data.items():
        dgn = df[df.DateTime <= DESIGN_END]
        if len(dgn) < 400:          # يكفي لحساب EMA200 + تاريخ معقول
            continue
        rows.append({"symbol": s, "bars_design": len(dgn),
                     "median_usd_vol": float(dgn.quote_volume.median()),
                     "first": str(dgn.DateTime.min().date())})
    liq = pd.DataFrame(rows).sort_values("median_usd_vol", ascending=False).reset_index(drop=True)
    liq.to_csv(OUT / "universe_liquidity.csv", index=False)
    print(f"صالح للاختبار: {len(liq)} رمزاً (≥400 شمعة قبل القطع)")
    print(liq.head(12).to_string(index=False))

    universes = {}
    for n in (6, 12, 24, 48, 90):
        n = min(n, len(liq))
        universes[n] = list(liq.symbol[:n])

    results, trades_by_n = {}, {}
    for n, syms in universes.items():
        sub = {s: data[s] for s in syms}
        tr = E.run_universe(sub, equity0=100.0)
        trades_by_n[n] = tr
        j = E.judge(tr, CUT)
        # معدل الصفقات في فترة الاختبار (الفترة المغلقة) — هو ما يهم للدخل
        blind = [t for t in tr if t["exit_date"] > CUT]
        span_years = (pd.Timestamp(max(t["exit_date"] for t in tr)) - DESIGN_END).days / 365.25
        tpm = len(blind) / (span_years * 12) if span_years > 0 else float("nan")
        results[n] = {"n_symbols": n, "trades": len(tr), "blind_trades": len(blind),
                      "trades_per_month_blind": round(float(tpm), 2),
                      "judge": j, "symbols": syms}
        b = j["blind"]
        print(f"\n=== {n} سوقاً | صفقات كلية {len(tr)} · مغلقة {len(blind)} · {tpm:.1f}/شهر")
        print(f"    blind: PF {b.get('pf')} · متوسط {b.get('expectancy_R')}R · "
              f"DD {b.get('max_dd_R')}R · حذف أفضل3 {b.get('stress_remove_best3_R')}R")
        print(f"    الحكم: {j['verdict']} | البوابات: {j['gate']}")

    pd.DataFrame([{**{"n": n, **{k: v for k, v in r["judge"]["blind"].items()}},
                   "trades_per_month": r["trades_per_month_blind"]}
                  for n, r in results.items()]).to_csv(OUT / "scaling_blind.csv", index=False)

    # صفقات الكون الكبير كاملة، لتحليل لاحق
    big = max(universes)
    pd.DataFrame(trades_by_n[big]).to_csv(OUT / f"trades_{big}symbols.csv", index=False)

    (OUT / "scaling_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nحُفظ: scaling_results.json · scaling_blind.csv · trades_{big}symbols.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
