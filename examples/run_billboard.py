"""End-to-end demo on the PUBLIC billboard dataset.

    load -> validate input -> signals (manual + PCA) -> score -> proxy validation

Run from the repo root:
    python examples/run_billboard.py [--top N] [--mode manual|pca|both]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from armadillo_scoring import schema, signals, score, validate  # noqa: E402
from armadillo_scoring.loaders import billboard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10, help="how many artists to print")
    parser.add_argument(
        "--mode", choices=("manual", "pca", "both"), default="both",
        help="signal reduction mode(s) to run",
    )
    parser.add_argument(
        "--csv", default=str(billboard.DEFAULT_CSV), help="path to the billboard CSV"
    )
    args = parser.parse_args()

    # 1. Load: billboard rows -> canonical ArtistRecords ----------------------
    records = billboard.load(args.csv)
    print(f"Loaded {len(records)} artists from {Path(args.csv).name}")

    # 2. Schema validation: refuse to score garbage ----------------------------
    problems = schema.validate_records(records, strict_range=True)
    if problems:
        for artist, issues in list(problems.items())[:5]:
            print(f"  INVALID {artist}: {issues}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} records failed schema validation.")
    print("Schema validation: all records OK")

    # 3. Proxy label (chart-derived, see validate.py docstring) ----------------
    raw = billboard.load_raw(args.csv)
    labels = validate.hit_labels_from_chart(
        raw, artist_col=billboard.ARTIST_COL, rank_col=billboard.RANK_COL, top_rank=10
    )

    modes = ("manual", "pca") if args.mode == "both" else (args.mode,)
    for mode in modes:
        print(f"\n{'=' * 66}\nMODE: {mode}\n{'=' * 66}")

        # 4. Reduce attributes -> ~5 composite signals -------------------------
        sig = signals.to_signals(records, mode=mode)

        # 5. Weight signals -> one speed score + contribution breakdown --------
        scored = score.speed_score(sig)
        weights = scored.attrs["weights"]
        print("weights: " + "  ".join(f"{k}={v:.2f}" for k, v in weights.items()))

        if mode == "pca":
            ratios = sig.attrs["explained_variance_ratio"]
            print("explained variance: "
                  + "  ".join(f"pc{i+1}={r:.0%}" for i, r in enumerate(ratios)))

        # 6. Top-N with their breakdown ----------------------------------------
        print(f"\nTop {args.top} artists by speed score:")
        for artist_id in scored.head(args.top).index:
            print(score.explain(scored, artist_id))

        # 7. Proxy validation ---------------------------------------------------
        report = validate.evaluate(scored[score.SCORE_COL], labels)
        print(f"\nProxy validation (hit = best chart position <= 10):\n{report}")

    print(
        "\nNOTE: signals and proxy label share the chart as their source, so these\n"
        "numbers show the pipeline ranks discriminatively — not that it predicts\n"
        "future breakout. See armadillo_scoring/README.md."
    )


if __name__ == "__main__":
    main()
