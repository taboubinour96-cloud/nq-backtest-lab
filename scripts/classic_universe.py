"""
classic_universe.py — القائمة الكلاسيكية بمعيار **مُعلَن مسبقاً** لا اختيار بعد النتيجة.

المعيار (مكتوب قبل التشغيل، ومنفَّذ حرفياً):
    كل أداة يستخدمها متداولو شركات التمويل، من ثلاث عائلات فقط:
        مؤشرات أسهم أمريكية · سلع · عملات رئيسية
    مع شرط توفّر 15 سنة بيانات يومية على Yahoo على الأقل.

لماذا هذا مهم: في النسخة الأولى اخترت 10 أدوات بعد أن رأيت أن الذهب والفضة
كانا الأفضل. ذلك انتقاء. الآن القائمة مبنية على «ما يُتداول في الشركة»،
ونُبلّغ بالنتيجة كما تخرج، حتى لو ضعفت.

النتيجة الكاملة تُحفظ في classic_universe.csv مع أرقام كل أداة منفصلة،
ليتمكّن أي مراجع من استبعاد أي أداة وإعادة الحساب.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import engine_cfd as C
import data_yahoo as Y

OUT = Path(__file__).parent
CUT = "2024-10-14"

# ---- القائمة المُعلَنة مسبقاً: ما تتيحه شركات التمويل فعلياً ----
UNIVERSE = {
    # مؤشرات
    "^NDX": ("US100 · Nasdaq 100", "index"),
    "^GSPC": ("US500 · S&P 500", "index"),
    "^DJI": ("US30 · Dow Jones", "index"),
    "^RUT": ("US2000 · Russell 2000", "index"),
    "^GDAXI": ("GER40 · DAX", "index"),
    "^FTSE": ("UK100 · FTSE 100", "index"),
    "^N225": ("JP225 · Nikkei 225", "index"),
    # سلع
    "GC=F": ("XAUUSD · الذهب", "commodity"),
    "SI=F": ("XAGUSD · الفضة", "commodity"),
    "CL=F": ("USOIL · النفط WTI", "commodity"),
    "NG=F": ("NGAS · الغاز الطبيعي", "commodity"),
    "HG=F": ("COPPER · النحاس", "commodity"),
    # عملات رئيسية
    "EURUSD=X": ("EURUSD", "fx"), "GBPUSD=X": ("GBPUSD", "fx"),
    "USDJPY=X": ("USDJPY", "fx"), "AUDUSD=X": ("AUDUSD", "fx"),
    "USDCAD=X": ("USDCAD", "fx"), "USDCHF=X": ("USDCHF", "fx"),
    "NZDUSD=X": ("NZDUSD", "fx"), "EURJPY=X": ("EURJPY", "fx"),
}


def main() -> int:
    rows, all_trades = [], []
    print(f"{'الأداة':10s} {'الاسم':22s} {'شمعة':>6} {'صفقات':>6} {'متوسطR':>8} {'PF':>6}")
    for sym, (label, family) in UNIVERSE.items():
        try:
            df = Y.download(sym)
        except Exception as e:
            print(f"{sym:10s} فشل: {str(e)[:40]}")
            continue
        years = (df.DateTime.max() - df.DateTime.min()).days / 365.25
        if years < 15:
            print(f"{sym:10s} {label:22s} مستبعد: {years:.1f} سنة فقط")
            continue
        tr = C.backtest(C.add_indicators(df), sym, cost_R=0.03)
        if not tr:
            continue
        r = pd.Series([t["r"] for t in tr])
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
        rows.append({"symbol": sym, "label": label, "family": family,
                     "bars": len(df), "years": round(years, 1), "trades": len(tr),
                     "mean_R_all": round(float(r.mean()), 4), "pf_all": round(float(pf), 3),
                     "first_bar": str(df.DateTime.min().date())})
        all_trades += tr
        print(f"{sym:10s} {label:22s} {len(df):6d} {len(tr):6d} {r.mean():+8.3f} {pf:6.2f}")

    u = pd.DataFrame(rows)
    u.to_csv(OUT / "classic_universe.csv", index=False)
    all_trades.sort(key=lambda t: t["exit_date"])
    pd.DataFrame(all_trades).to_csv(OUT / "trades_classic.csv", index=False)

    meta = {"cut_date": CUT,
            "rule": "every instrument a prop firm actually offers, from 3 families, with >=15 years of daily data on Yahoo; no performance-based selection",
            "families": {"index": int((u.family == "index").sum()),
                         "commodity": int((u.family == "commodity").sum()),
                         "fx": int((u.family == "fx").sum())},
            "symbols": list(u.symbol)}
    (OUT / "classic_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nالمجموع: {len(u)} أداة · صفقات {len(all_trades)}")
    print("حُفظ: classic_universe.csv · classic_meta.json · trades_classic.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
