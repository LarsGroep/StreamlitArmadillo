# armadillo_scoring

Source-agnostic **artist speed-scoring kit** (UvA × LOFI talent-scout project).
One number per artist — the **speed score** — plus a breakdown that explains it,
computed from ~5 composite signals.

Developed and validated entirely on the **public** Billboard dataset
(`archive/billboard_24years_lyrics_spotify.csv`). **No LOFI company data is used
anywhere in this package** (NDA), and `.gitignore` blocks `*lofi*.csv` patterns
from ever being committed.

```
raw source ──loader──▶ ArtistRecords ──signals──▶ 5 signals ──score──▶ speed score
 (billboard,            (canonical                (manual or            + breakdown
  chartmetric, …)        schema)                   PCA)                    │
                                                              validate ◀──┘
                                                         (proxy AUC / precision@k)
```

## Quick start

```bash
python examples/run_billboard.py            # full demo, both signal modes
python examples/run_billboard.py --top 20 --mode manual
python -m pytest tests/ -q                  # 75 tests
```

## The input schema (`schema.py`)

The whole pipeline speaks exactly one data shape, the **`ArtistRecord`**:

```python
ArtistRecord(
    artist_id="Dua Lipa",            # stable, source-unique string
    attributes={                     # canonical, normalized features
        "chart_quality": 0.59,
        "reach": 0.94,
        "recency": 1.0,
        ...                          # any subset of the canonical vocabulary
    },
    name="Dua Lipa",                 # optional, display only
    source="billboard",              # optional provenance tag
)
```

Rules of the contract:

1. **Attribute names come from `schema.CANONICAL_ATTRIBUTES`** (9 names today).
   Unknown names fail validation — extend the vocabulary deliberately, not ad hoc.
2. **Values are normalized to [0, 1] and oriented "higher = more promising".**
   If a raw metric is better when low (chart rank, days-since-release), the
   *loader* inverts it. Downstream code never tracks polarity.
3. **Missing attributes are omitted, never invented.** The pipeline averages
   over what exists and median-fills at the signal level.

The 9 attributes and the composite signal each feeds (`schema.SIGNAL_CONFIG`):

| signal     | attributes                       | intuition                       |
|------------|----------------------------------|---------------------------------|
| growth     | `rank_trend`, `new_entry_rate`   | is the trajectory improving?    |
| engagement | `chart_quality`, `content_appeal`| how strongly does content land? |
| live       | `live_affinity`                  | live-performance dimension      |
| audience   | `reach`, `catalog_breadth`       | how broad is the footprint?     |
| momentum   | `recency`, `recent_activity`     | how hot is the artist *now*?    |

The five signals combine into the speed score via **`schema.SIGNAL_WEIGHTS`** —
five numbers, summing to 1. That is the entire tunable surface (deliberately:
no twenty loose per-feature weights).

## How scoring works

- **`signals.to_signals(records, mode=...)`** reduces attributes to the 5 signals.
  - `mode="manual"` — average each `SIGNAL_CONFIG` group. Transparent default.
  - `mode="pca"` — 5 principal components (deterministic; each component is
    oriented to align with the schema's higher=better direction, then min-max
    scaled). Use as a data-driven cross-check on the manual grouping;
    explained-variance ratios ride along in `df.attrs`. Caveat: a component
    mixes attributes with both signs, so `contrib_pc*` values are valid score
    shares but are **not** readable as "more of attribute X = better" — the
    attribute-level story only holds for manual mode.
- **`score.speed_score(signals_df)`** returns per artist: `speed_score` and one
  `contrib_<signal>` column per signal (= weight × value). The contributions
  **sum to the score exactly** — the breakdown *is* the explanation (no SHAP,
  no surrogate model). `score.explain(scored, artist_id)` pretty-prints one row.

## Connecting a new source (e.g. Chartmetric)

The core never changes. You write **one loader** and optionally tune the config:

1. Create `loaders/chartmetric.py` with a `load() -> list[ArtistRecord]`.
2. Map raw metrics onto the canonical vocabulary, e.g.:
   - listener/follower growth rate → `rank_trend` (growth)
   - playlist adds per month → `new_entry_rate` (growth)
   - streams-per-listener, save rate → `chart_quality` / `content_appeal` (engagement)
   - event/ticket signals → `live_affinity` (live)
   - monthly listeners, market count → `reach` / `catalog_breadth` (audience)
   - last-28-day deltas → `recency` / `recent_activity` (momentum)
3. Normalize each to [0, 1] **within your cohort** (percentile rank is robust),
   orient higher = better, omit what you can't compute.
4. Run `schema.validate_records(records, strict_range=True)` — fix anything it
   reports.
5. Score. Adjust `SIGNAL_WEIGHTS` (or pass `weights=` to `speed_score`) if the
   business priorities differ per source.

That's the whole integration: the billboard loader is ~190 lines, most of it
documentation.

## What the proxy validation means (and doesn't)

`validate.py` builds a **proxy label** because no ground-truth "this artist
broke through" flag exists in public data: an artist counts as a **hit** if
their best chart position ever reached the top 10 (configurable, or top-X%).
It then reports:

- **AUC** — probability a random hit outscores a random non-hit
  (0.5 = random, 1.0 = perfect separation);
- **precision@k** — of the top-k artists by speed score, the share that are
  hits ("if the scout meets the top k, how many are real?"), with the base
  rate printed for context.

Current results on the public billboard set (1,039 artists, 21.5% base rate):

| mode   | AUC   | precision@10 | precision@25 | precision@50 |
|--------|-------|--------------|--------------|--------------|
| manual | 0.845 | 90%          | 80%          | 72%          |
| pca    | 0.895 | 100%         | 96%          | 88%          |

(These numbers are pinned by an end-to-end test in `tests/test_loader_billboard.py`,
so this table cannot silently drift from the code.)

**What this DOES show:** the pipeline is wired correctly end-to-end and
produces a strongly discriminative ranking — hits concentrate at the top
(~4× the base rate in the top 10).

**What this does NOT show:** that the score predicts *future* breakout. The
signals and the proxy label are both derived from the same chart data, so part
of the separation is circular by construction (e.g. `chart_quality` and "peaked
top-10" are correlated by definition). Treat these numbers as a **plumbing
check and a relative yardstick between configurations**, not as forecast
accuracy. A real predictive claim needs **time-split validation** — score
artists using data up to year *t*, label them by what happened after *t* —
which becomes possible once a longitudinal source (Chartmetric history) is
connected.

## Layout

```
armadillo_scoring/
├── schema.py              # canonical contract: ArtistRecord, SIGNAL_CONFIG, weights
├── signals.py             # attributes -> 5 composite signals (manual | pca)
├── score.py               # signals -> speed score + exact contribution breakdown
├── validate.py            # proxy label + AUC / precision@k
├── loaders/
│   └── billboard.py       # the ONLY billboard-specific file
examples/run_billboard.py  # end-to-end demo
tests/                     # schema, signals, score, validate, loader, e2e (75 tests)
```
