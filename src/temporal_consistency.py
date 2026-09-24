"""
Temporal consistency analysis requested by Reviewer 1.

Place this file in the same directory as sws_updated_http2_removed.py.
Provide the four raw crawling-session CSV files in the same order as their dates.

Example:
    python temporal_consistency_analysis.py \
      --session-files session1.csv session2.csv session3.csv session4.csv \
      --session-dates 2026-01-15 2026-02-15 2026-03-15 2026-04-15 \
      --output-dir revised_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from sws_updated_http2_removed import add_sws_variants, calculate_components

KEY_METRICS = ["total_requests", "final_total_bytes", "final_co2_grams", "SWS_revised"]


def prepare_session(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df, _, _ = calculate_components(df)
    # A single Monte Carlo iteration is sufficient here because only SWS_revised is needed.
    df, _, _, _, _ = add_sws_variants(df, random_iter=1, seed=42)
    if "url" not in df.columns:
        raise ValueError(f"The file {path} must contain a 'url' column for between-session matching.")
    return df[["url"] + KEY_METRICS].copy()


def pairwise_table(sessions: List[pd.DataFrame], dates: List[str], names: List[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            merged = sessions[i].merge(sessions[j], on="url", suffixes=("_a", "_b"))
            for metric in KEY_METRICS:
                a = merged[f"{metric}_a"]
                b = merged[f"{metric}_b"]
                sp = spearmanr(a, b, nan_policy="omit")
                kt = kendalltau(a, b, nan_policy="omit")
                denom = np.maximum((np.abs(a) + np.abs(b)) / 2.0, 1e-12)
                relative_diff = np.abs(a - b) / denom
                rows.append(
                    {
                        "session_a": names[i],
                        "date_a": dates[i],
                        "session_b": names[j],
                        "date_b": dates[j],
                        "matched_homepages": int(len(merged)),
                        "metric": metric,
                        "spearman": float(sp.statistic),
                        "spearman_p": float(sp.pvalue),
                        "kendall_tau": float(kt.statistic),
                        "kendall_p": float(kt.pvalue),
                        "median_absolute_relative_difference": float(np.median(relative_diff)),
                        "mean_absolute_relative_difference": float(np.mean(relative_diff)),
                    }
                )
    return pd.DataFrame(rows)


def session_summary(sessions: List[pd.DataFrame], dates: List[str], names: List[str]) -> pd.DataFrame:
    rows = []
    for name, date, df in zip(names, dates, sessions):
        row = {"session": name, "date": date, "successful_homepages": int(len(df))}
        for metric in KEY_METRICS:
            row[f"{metric}_mean"] = float(df[metric].mean())
            row[f"{metric}_std"] = float(df[metric].std())
            row[f"{metric}_median"] = float(df[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal consistency across four crawling sessions.")
    parser.add_argument("--session-files", nargs=4, required=True)
    parser.add_argument("--session-dates", nargs=4, required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    paths = [Path(p) for p in args.session_files]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    sessions = [prepare_session(path) for path in paths]
    names = [f"Session {i + 1}" for i in range(4)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pairwise_table(sessions, args.session_dates, names)
    summary = session_summary(sessions, args.session_dates, names)
    pairwise.to_csv(output_dir / "temporal_consistency_pairwise.csv", index=False, encoding="utf-8-sig")
    pairwise.to_excel(output_dir / "temporal_consistency_pairwise.xlsx", index=False)
    summary.to_csv(output_dir / "temporal_session_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_excel(output_dir / "temporal_session_summary.xlsx", index=False)

    print("Temporal consistency analysis completed.")
    print("Outputs:", output_dir.resolve())


if __name__ == "__main__":
    main()
