"""
metrics.py
----------
Measures the claims made in the README / resume so you can back them with numbers.

1. Data-inconsistency reduction  (README: "~35% reduction in data inconsistencies")
   Counts problems in the raw data and again in the cleaned data:
   missing values, duplicate rows, IQR outliers, invalid values (negative / impossible),
   inconsistent text spellings (M / male / Male ), mixed date formats.

2. Cleaning effort + speed       (README: "reusable functions cut repeat cleaning work by ~30%")
   - Speed:  median time and records/second of clean_dataset().
   - Effort: approx. lines of code needed to clean N datasets with copy-pasted
             one-off scripts vs. writing the reusable module once and calling it N times.

Usage:
    python metrics.py            # reads data/raw_*.csv and data/cleaned_*.csv, prints + saves a report
"""

import inspect
import json
import os
import re
import statistics
import time

import pandas as pd

DATE_RE = r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$"
AGE_RE = re.compile(r"(^|_|\s)age($|_|\s)")
NON_NEGATIVE = ("price", "amount", "income", "salary", "quantity", "spend", "purchase", "revenue", "score")
SYNONYMS = {"m": "male", "f": "female"}


def count_inconsistencies(df):
    """Return a dict {problem type: count} for one dataframe."""
    out = {"Missing values": int(df.isna().sum().sum()),
           "Duplicate rows": int(df.duplicated().sum()),
           "Outliers (IQR)": 0, "Invalid values": 0,
           "Inconsistent text": 0, "Mixed date formats": 0}
    for c in df.columns:
        s, low = df[c], str(c).lower()
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna()
            if x.nunique() > 12:
                q1, q3 = x.quantile([.25, .75])
                i = q3 - q1
                out["Outliers (IQR)"] += int(((x < q1 - 1.5 * i) | (x > q3 + 1.5 * i)).sum())
            if AGE_RE.search(low):
                out["Invalid values"] += int(((x < 0) | (x > 120)).sum())
            elif any(k in low for k in NON_NEGATIVE):
                out["Invalid values"] += int((x < 0).sum())
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            x = s.dropna().astype(str)
            if x.empty:
                continue
            if any(k in low for k in ("date", "time")):
                out["Mixed date formats"] += int((~x.str.strip().str.match(DATE_RE)).sum())
            elif x.nunique() <= 50:
                canon = x.str.strip().str.lower().replace(SYNONYMS)
                out["Inconsistent text"] += max(0, x.nunique() - canon.nunique())
    return out


def inconsistency_report(raw_df, clean_df):
    b, a = count_inconsistencies(raw_df), count_inconsistencies(clean_df)
    tb, ta = sum(b.values()), sum(a.values())
    table = pd.DataFrame({"Problem type": list(b), "Before": list(b.values()), "After": [a[k] for k in b]})
    return dict(before_total=tb, after_total=ta,
                reduction_pct=round((tb - ta) / tb * 100, 1) if tb else 0.0, table=table)


def _code_lines(module):
    src = inspect.getsource(module)
    return len([l for l in src.splitlines()
                if l.strip() and not l.strip().startswith(("#", '"""', "'''"))])


def benchmark(raw_df, clean_fn, runs=5, n_datasets=3):
    times = []
    for _ in range(runs):
        d = raw_df.copy()
        t = time.perf_counter()
        clean_fn(d)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    loc = _code_lines(inspect.getmodule(clean_fn))
    one_off, reusable = n_datasets * loc, loc + n_datasets      # +1 call line per dataset
    return dict(median_seconds=round(med, 3), records_per_second=int(len(raw_df) / med) if med else 0,
                module_lines=loc, n_datasets=n_datasets, one_off_lines=one_off, reusable_lines=reusable,
                effort_reduction_pct=round((one_off - reusable) / one_off * 100, 1))


def verdicts(rep, bench=None):
    v = [f"Inconsistencies: {rep['before_total']:,} -> {rep['after_total']:,}  ({rep['reduction_pct']}% reduction)  "
         + ("[README claim of ~35% is SUPPORTED]" if rep["reduction_pct"] >= 35
            else f"[below the 35% claim - write '{rep['reduction_pct']:.0f}%' on the resume instead]")]
    if bench:
        v.append(f"Speed: cleans {bench['records_per_second']:,} records/second "
                 f"(median {bench['median_seconds']} s)")
        v.append(f"Reuse (estimated): ~{bench['one_off_lines']} lines (copy-paste x{bench['n_datasets']}) vs "
                 f"~{bench['reusable_lines']} lines (reusable) = ~{bench['effort_reduction_pct']}% less code (estimated)"
                 + ("  [ESTIMATE only: lines-of-code count, not measured time or effort]"))
    return v


def save_report(rep, bench, path="outputs/metrics_report.md"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Project metrics report\n\n" + rep["table"].to_string(index=False) + "\n\n")
        f.write("\n".join(f"- {line}" for line in verdicts(rep, bench)) + "\n")
    with open(path.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(inconsistencies=rep["table"].to_dict("records"), reduction_pct=rep["reduction_pct"],
                       benchmark=bench), f, indent=2)
    return path


if __name__ == "__main__":
    from src.cleaning import clean_dataset
    raw = pd.read_csv("data/raw_customer_data.csv")
    clean = pd.read_csv("data/cleaned_customer_data.csv")
    rep, bench = inconsistency_report(raw, clean), benchmark(raw, clean_dataset)
    print(rep["table"].to_string(index=False), "\n")
    print("\n".join(verdicts(rep, bench)))
    print("\nSaved:", save_report(rep, bench))
