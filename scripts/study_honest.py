"""
study_honest.py — إصلاح ثغرة في اختيار الكون، اكتشفها سؤال Arena 1994.

الثغرة: في study_scaling.py رتّبتُ الرموز بحجم ملف شهر 2026-09 على S3.
        ذلك معيار **مستقبلي** (look-ahead) — يُدخل انتقاءً على أساس نتيجة
        لم تكن معلومة وقت القرار. غير مقبول.

الإصلاح: الترتيب بمتوسط حجم التداول اليومي بالدولار **داخل فترة التصميم
        وحدها** (من بداية الشهر حتى تاريخ القطع 2024-10-14)، وكل الرموز
        المؤهَّلة تدخل الطلب (لا اختيار مسبق).

شرط الأهلية (معلن مسبقاً، لا بعد رؤية النتيجة):
    ≥ 300 شمعة صالحة **قبل** تاريخ القطع   (يكفي لحساب EMA200 + تاريخ)
    وبيانات تبدأ قبل 2024-01-01             (حتى لا تكون العملة جديدة)

المخرجات: universe_final.csv (القائمة النهائية + رتبتها + سبب الأهلية)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_free as D

OUT = Path(__file__).parent
CUT = pd.Timestamp("2024-10-14")
MIN_BARS = 300
MIN_START = pd.Timestamp("2024-01-01")


def eligible_symbols() -> list[str]:
    syms = [s for s in D.list_symbols() if s.endswith("USDT") and not s.endswith("USDCUSDT")]
    print(f"رموز USDT المتاحة: {len(syms)}")

    def first_month(s):
        ms = D.symbol_months(s)
        return s, (ms[0] if ms else None)

    with ThreadPoolExecutor(24) as ex:
        firsts = dict(ex.map(first_month, syms))
    keep = [s for s, m in firsts.items() if m and m <= "2024-01"]
    print(f"لها سجل قبل 2024-01: {len(keep)}")
    return sorted(keep), firsts


def main() -> int:
    keep, firsts = eligible_symbols()
    print(f"أنزّل {len(keep)} رمزاً (الكل مؤهّل، لا انتقاء مسبق)…")
    data = D.download_many(keep, workers=12, verbose=False)
    print(f"نُزّل بنجاح: {len(data)}")

    rows = []
    for s, df in data.items():
        dgn = df[df.DateTime <= CUT]
        if len(dgn) < MIN_BARS:
            continue
        rows.append({
            "symbol": s,
            "bars_design": int(len(dgn)),
            "median_usd_vol_design": float(dgn.quote_volume.median()),
            "first_bar": str(dgn.DateTime.min().date()),
            "last_bar": str(df.DateTime.max().date()),
        })
    uni = pd.DataFrame(rows).sort_values("median_usd_vol_design", ascending=False).reset_index(drop=True)
    uni.insert(0, "rank", uni.index + 1)
    uni.to_csv(OUT / "universe_final.csv", index=False)

    print(f"\nالمؤهَّل بعد شرط {MIN_BARS} شمعة تصميم: {len(uni)}")
    print(uni.head(15)[["rank", "symbol", "bars_design", "median_usd_vol_design", "first_bar"]].to_string(index=False))

    meta = {
        "cut_date": str(CUT.date()),
        "eligibility": {"min_bars_before_cut": MIN_BARS, "min_first_month": "2024-01"},
        "ranking_metric": "median daily quote volume (USD) computed ONLY on bars <= cut_date",
        "selection_rule": "top 24 by rank for the crypto pool; no other filter applied",
        "total_usdt_symbols": len(firsts),
        "eligible_symbols": len(keep),
        "passed_min_bars": int(len(uni)),
        "pool_used": [s for s in uni.symbol[:24]],
        "previous_flawed_pool": None,
    }
    try:
        old = pd.read_csv(OUT / "universe_liquidity.csv")
        meta["previous_flawed_pool"] = list(old.symbol[:24])
    except Exception:
        pass
    (OUT / "universe_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\nحُفظ: universe_final.csv · universe_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
