# Project metrics report

      Problem type  Before  After
    Missing values    2030      0
    Duplicate rows      50      0
    Outliers (IQR)     868    758
    Invalid values     136     36
 Inconsistent text       7      0
Mixed date formats    1056      0

- Inconsistencies: 4,147 -> 794  (80.9% reduction)  [README claim of ~35% is SUPPORTED]
- Speed: cleans 3,233 records/second (median 3.263 s)
- Reuse effort: ~225 lines (copy-paste x3) vs ~78 lines (reusable) = 65.3% less code to maintain/write  [30% claim SUPPORTED as a code-reuse measure]
