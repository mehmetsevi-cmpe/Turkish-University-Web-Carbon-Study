"""
Revised Sustainable Web Score (SWS) analysis for Reviewer 1, Revision 2.

Key methodological change
-------------------------
The HTTP/2 component is excluded from the primary SWS because it is constant
(False/0) across the analysed sample and therefore carries no discriminatory
information. The remaining original weights are renormalised while preserving
their relative proportions:

    CO2 proxy = 0.50
    Media     = 0.25
    Requests  = 0.125
    Third-party dependency = 0.125

The script also implements:
- exact deterministic alternative weight vectors;
- entropy-derived weights;
- 10,000-run Dirichlet Monte Carlo weight uncertainty analysis;
- distributions/quantiles of ranking metrics across simulations;
- leave-one-component-out analysis;
- explicit sample-specific normalisation parameters;
- relative, non-normative performance-band labels;
- an HTTP/2 diagnostic output for the reviewer response.

Run:
    python sws_updated_http2_removed.py \
        --input results_playwrightnew3.csv \
        --output-dir revised_outputs \
        --random-iter 10000 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr


DEFAULT_INPUT = "results_playwrightnew3.csv"
OUTPUT_DIR = Path(".")

ACTIVE_COMPONENTS = ["score_co2", "score_media", "score_reqs", "score_third"]
COMPONENT_LABELS = {
    "score_co2": "CO2-proxy score",
    "score_media": "Media score",
    "score_reqs": "Request-count score",
    "score_third": "Third-party score",
}

# HTTP/2 removed; remaining former weights renormalised to sum to 1.
REVISED_WEIGHTS: Dict[str, float] = {
    "score_co2": 0.50,
    "score_media": 0.25,
    "score_reqs": 0.125,
    "score_third": 0.125,
}

RELATIVE_BINS = [-np.inf, 40, 60, 80, np.inf]
RELATIVE_LABELS = [
    "Lower relative performance",
    "Lower-intermediate relative performance",
    "Upper-intermediate relative performance",
    "Higher relative performance",
]


def _as_bool(value) -> bool:
    """Safely convert common CSV representations to Boolean."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    return value in {"true", "1", "yes", "y", "h2", "http2", "http/2"}


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0, upper=1)


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    if any(v < 0 for v in weights.values()):
        raise ValueError("Weights must be non-negative.")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return {k: float(v / total) for k, v in weights.items()}


def weighted_score(df: pd.DataFrame, weights: Dict[str, float], output_col: str) -> pd.Series:
    """Create a 0-100 weighted score from normalised component scores."""
    weights = _normalise_weights(weights)
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise ValueError(f"Missing component columns: {missing}")
    score = sum(df[k] * w for k, w in weights.items()) * 100.0
    return score.rename(output_col)


def normalise_revised_without(exclude: Iterable[str]) -> Dict[str, float]:
    exclude = set(exclude)
    remaining = {k: v for k, v in REVISED_WEIGHTS.items() if k not in exclude}
    if not remaining:
        raise ValueError("No weights remain after exclusion.")
    return _normalise_weights(remaining)


def entropy_weights(df: pd.DataFrame, component_cols: List[str]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Calculate entropy weights and return a transparent calculation table."""
    x = df[component_cols].astype(float).clip(lower=0, upper=1).copy()
    n = len(x)
    if n < 2:
        raise ValueError("Entropy weighting requires at least two observations.")

    eps = 1e-12
    col_sums = x.sum(axis=0).replace(0, eps)
    p = x / col_sums
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p + eps)).sum(axis=0)
    divergence = 1.0 - entropy

    variable = x.var(axis=0) > 1e-12
    divergence = divergence.where(variable, 0.0)
    if float(divergence.sum()) <= eps:
        weights = pd.Series(1.0 / len(component_cols), index=component_cols)
    else:
        weights = divergence / divergence.sum()

    details = pd.DataFrame(
        {
            "component": component_cols,
            "variance": [float(x[c].var()) for c in component_cols],
            "entropy": [float(entropy[c]) for c in component_cols],
            "divergence": [float(divergence[c]) for c in component_cols],
            "entropy_weight": [float(weights[c]) for c in component_cols],
        }
    )
    return {c: float(weights[c]) for c in component_cols}, details


def calculate_components(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate the four active components and diagnostic/normalisation tables."""
    df = df.copy()
    required = [
        "final_co2_grams",
        "total_requests",
        "image_requests",
        "video_requests",
        "third_party_bytes",
        "final_total_bytes",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input file lacks required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    max_co2 = float(df["final_co2_grams"].max()) or 1.0
    max_requests = float(df["total_requests"].max()) or 1.0

    df["score_co2"] = _clip01(1.0 - (df["final_co2_grams"] / max_co2))

    denom_requests = df["total_requests"].replace(0, np.nan)
    media_ratio = (df["image_requests"] + df["video_requests"]) / denom_requests
    df["score_media"] = _clip01(1.0 - media_ratio.fillna(0.0))

    df["score_reqs"] = _clip01(1.0 - (df["total_requests"] / max_requests))

    denom_bytes = df["final_total_bytes"].replace(0, np.nan)
    third_ratio = df["third_party_bytes"] / denom_bytes
    df["score_third"] = _clip01(1.0 - third_ratio.fillna(0.0))

    normalisation = pd.DataFrame(
        [
            {
                "parameter": "CO2_proxy_upper_bound",
                "source_variable": "final_co2_grams",
                "value": max_co2,
                "unit": "gCO2-equivalent proxy per page load",
                "basis": "maximum observed value in the reference sample",
            },
            {
                "parameter": "request_count_upper_bound",
                "source_variable": "total_requests",
                "value": max_requests,
                "unit": "requests",
                "basis": "maximum observed value in the reference sample",
            },
        ]
    )

    if "uses_http2" in df.columns:
        parsed = df["uses_http2"].apply(_as_bool)
        http2_diag = pd.DataFrame(
            [
                {
                    "total_observations": int(len(parsed)),
                    "true_count": int(parsed.sum()),
                    "false_count": int((~parsed).sum()),
                    "unique_parsed_values": int(parsed.nunique(dropna=False)),
                    "component_status": "excluded from revised SWS because it is constant and non-discriminatory",
                }
            ]
        )
    else:
        http2_diag = pd.DataFrame(
            [
                {
                    "total_observations": int(len(df)),
                    "true_count": np.nan,
                    "false_count": np.nan,
                    "unique_parsed_values": np.nan,
                    "component_status": "uses_http2 column not present; HTTP/2 is not used in revised SWS",
                }
            ]
        )

    return df, normalisation, http2_diag


def deterministic_rank_metrics(
    df: pd.DataFrame,
    variant_cols: List[str],
    base_col: str = "SWS_revised",
) -> pd.DataFrame:
    rows = []
    base_top10 = set(df.nlargest(10, base_col).index)
    base_bottom10 = set(df.nsmallest(10, base_col).index)
    base_ranks = pd.Series(df[base_col]).rank(method="average", ascending=False).to_numpy()

    for col in variant_cols:
        if col == base_col or col not in df.columns:
            continue
        sp = spearmanr(df[base_col], df[col], nan_policy="omit")
        kt = kendalltau(df[base_col], df[col], nan_policy="omit")
        top10 = set(df.nlargest(10, col).index)
        bottom10 = set(df.nsmallest(10, col).index)
        ranks = pd.Series(df[col]).rank(method="average", ascending=False).to_numpy()
        rows.append(
            {
                "variant": col,
                "spearman_with_revised": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "kendall_with_revised": float(kt.statistic),
                "kendall_p": float(kt.pvalue),
                "top10_overlap_ratio": len(base_top10 & top10) / 10.0,
                "bottom10_overlap_ratio": len(base_bottom10 & bottom10) / 10.0,
                "mean_absolute_rank_change": float(np.mean(np.abs(base_ranks - ranks))),
                "median_absolute_rank_change": float(np.median(np.abs(base_ranks - ranks))),
                "mean_score": float(df[col].mean()),
                "std_score": float(df[col].std()),
                "min_score": float(df[col].min()),
                "max_score": float(df[col].max()),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_rank_uncertainty(
    df: pd.DataFrame,
    iterations: int,
    seed: int,
    base_col: str = "SWS_revised",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Dirichlet random-weight uncertainty analysis on ranking metrics."""
    if iterations < 1:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(ACTIVE_COMPONENTS)), size=iterations)
    x = df[ACTIVE_COMPONENTS].astype(float).to_numpy()
    simulated_scores = x @ weights.T * 100.0  # shape: n x iterations

    base_values = df[base_col].to_numpy()
    base_rank = rankdata(-base_values, method="average")
    base_top = set(np.argsort(-base_values)[:10])
    base_bottom = set(np.argsort(base_values)[:10])

    rows = []
    for i in range(iterations):
        scores = simulated_scores[:, i]
        rank = rankdata(-scores, method="average")
        sp = spearmanr(base_values, scores)
        kt = kendalltau(base_values, scores)
        top = set(np.argsort(-scores)[:10])
        bottom = set(np.argsort(scores)[:10])
        abs_change = np.abs(base_rank - rank)
        rows.append(
            {
                "simulation": i + 1,
                "seed": seed,
                "weight_distribution": f"Dirichlet({','.join(['1'] * len(ACTIVE_COMPONENTS))})",
                **{f"weight_{c.replace('score_', '')}": float(weights[i, j]) for j, c in enumerate(ACTIVE_COMPONENTS)},
                "spearman_with_revised": float(sp.statistic),
                "kendall_with_revised": float(kt.statistic),
                "top10_overlap_ratio": len(base_top & top) / 10.0,
                "bottom10_overlap_ratio": len(base_bottom & bottom) / 10.0,
                "mean_absolute_rank_change": float(abs_change.mean()),
                "median_absolute_rank_change": float(np.median(abs_change)),
                "maximum_absolute_rank_change": float(abs_change.max()),
            }
        )

    distribution = pd.DataFrame(rows)
    metrics = [
        "spearman_with_revised",
        "kendall_with_revised",
        "top10_overlap_ratio",
        "bottom10_overlap_ratio",
        "mean_absolute_rank_change",
        "median_absolute_rank_change",
        "maximum_absolute_rank_change",
    ]
    summary_rows = []
    for metric in metrics:
        s = distribution[metric]
        summary_rows.append(
            {
                "metric": metric,
                "iterations": iterations,
                "seed": seed,
                "weight_distribution": f"Dirichlet({','.join(['1'] * len(ACTIVE_COMPONENTS))})",
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p2_5": float(s.quantile(0.025)),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "p97_5": float(s.quantile(0.975)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return distribution, pd.DataFrame(summary_rows), simulated_scores


def add_sws_variants(
    df: pd.DataFrame,
    random_iter: int = 10000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["SWS_revised"] = weighted_score(df, REVISED_WEIGHTS, "SWS_revised")
    df["SWS"] = df["SWS_revised"]

    equal_weights = {c: 1.0 / len(ACTIVE_COMPONENTS) for c in ACTIVE_COMPONENTS}
    df["SWS_equal_weight"] = weighted_score(df, equal_weights, "SWS_equal_weight")

    df["SWS_without_CO2"] = weighted_score(
        df,
        normalise_revised_without(["score_co2"]),
        "SWS_without_CO2",
    )

    entropy_w, entropy_details = entropy_weights(df, ACTIVE_COMPONENTS)
    df["SWS_data_driven"] = weighted_score(df, entropy_w, "SWS_data_driven")

    for comp in ACTIVE_COMPONENTS:
        readable = comp.replace("score_", "")
        col = f"SWS_leaveout_{readable}"
        df[col] = weighted_score(df, normalise_revised_without([comp]), col)

    mc_distribution, mc_summary, simulated_scores = monte_carlo_rank_uncertainty(
        df,
        iterations=random_iter,
        seed=seed,
    )
    df["SWS_random_weight_mean"] = simulated_scores.mean(axis=1)
    df["SWS_random_weight_std"] = simulated_scores.std(axis=1)
    df["SWS_random_weight_p025"] = np.percentile(simulated_scores, 2.5, axis=1)
    df["SWS_random_weight_p975"] = np.percentile(simulated_scores, 97.5, axis=1)

    df["SWS_relative_band"] = pd.cut(
        df["SWS_revised"],
        bins=RELATIVE_BINS,
        labels=RELATIVE_LABELS,
        right=False,
    )
    df["SWS_sample_quartile"] = pd.qcut(
        df["SWS_revised"].rank(method="first"),
        q=4,
        labels=["Q1 lowest", "Q2 lower-middle", "Q3 upper-middle", "Q4 highest"],
    )

    return df, entropy_w, entropy_details, mc_distribution, mc_summary


def build_weight_table(entropy_w: Dict[str, float]) -> pd.DataFrame:
    rows = [
        {"formulation": "Revised heuristic SWS", **REVISED_WEIGHTS},
        {
            "formulation": "Equal-weight SWS",
            **{c: 1.0 / len(ACTIVE_COMPONENTS) for c in ACTIVE_COMPONENTS},
        },
        {
            "formulation": "CO2-excluded SWS",
            **normalise_revised_without(["score_co2"]),
        },
        {"formulation": "Entropy/data-driven SWS", **entropy_w},
    ]
    for comp in ACTIVE_COMPONENTS:
        rows.append(
            {
                "formulation": f"Leave out {comp.replace('score_', '')}",
                **normalise_revised_without([comp]),
            }
        )
    return pd.DataFrame(rows)


def category_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["SWS_relative_band"].value_counts().reindex(RELATIVE_LABELS, fill_value=0)
    return pd.DataFrame(
        {
            "SWS_range": ["0-40", "40-60", "60-80", "80-100"],
            "relative_performance_band": RELATIVE_LABELS,
            "count": [int(counts[label]) for label in RELATIVE_LABELS],
            "percentage": [float(counts[label] / len(df) * 100.0) for label in RELATIVE_LABELS],
        }
    )


def save_plots(
    df: pd.DataFrame,
    rank_table: pd.DataFrame,
    mc_distribution: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(df["SWS_revised"], bins=20)
    plt.xlabel("Revised SWS (0-100; within-sample relative index)")
    plt.ylabel("Number of homepages")
    plt.title(f"Distribution of Revised SWS (n = {len(df)})")
    plt.tight_layout()
    plt.savefig(output_dir / "sws_revised_distribution.png", dpi=300)
    plt.close()

    corr = df[ACTIVE_COMPONENTS].corr(method="spearman")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, label="Spearman correlation")
    ticks = np.arange(len(ACTIVE_COMPONENTS))
    labels = [COMPONENT_LABELS[c] for c in ACTIVE_COMPONENTS]
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    for i in range(len(ACTIVE_COMPONENTS)):
        for j in range(len(ACTIVE_COMPONENTS)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("Correlation Matrix of Revised SWS Components")
    plt.tight_layout()
    plt.savefig(output_dir / "component_correlation_matrix_revised.png", dpi=300)
    plt.close()

    if not rank_table.empty:
        plot_df = rank_table.sort_values("spearman_with_revised", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["variant"], plot_df["spearman_with_revised"])
        plt.xlabel("Spearman rank correlation with revised SWS")
        plt.ylabel("SWS variant")
        plt.title("Deterministic Rank Stability under Alternative Scoring Assumptions")
        plt.xlim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / "deterministic_rank_stability.png", dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["final_co2_grams"], df["SWS_revised"], alpha=0.75)
    plt.xlabel("Standardised data-transfer-based CO2-emission proxy")
    plt.ylabel("Revised SWS (within-sample relative index)")
    plt.title(f"CO2-Emission Proxy vs Revised SWS (n = {len(df)})")
    plt.tight_layout()
    plt.savefig(output_dir / "co2_proxy_vs_sws_revised.png", dpi=300)
    plt.close()

    counts = df["SWS_relative_band"].value_counts().reindex(RELATIVE_LABELS, fill_value=0)
    plt.figure(figsize=(10, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("Number of homepages")
    plt.xlabel("Descriptive relative SWS band")
    plt.title(f"Distribution of Relative SWS Bands (n = {len(df)})")
    plt.xticks(rotation=20, ha="right")
    for idx, val in enumerate(counts.values):
        plt.text(idx, val, f"{val}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "relative_band_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(mc_distribution["spearman_with_revised"], bins=30)
    plt.xlabel("Spearman correlation with revised SWS")
    plt.ylabel("Number of Monte Carlo simulations")
    plt.title("Monte Carlo Distribution of Rank Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "monte_carlo_spearman_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(mc_distribution["mean_absolute_rank_change"], bins=30)
    plt.xlabel("Mean absolute rank change")
    plt.ylabel("Number of Monte Carlo simulations")
    plt.title("Monte Carlo Distribution of Ranking Changes")
    plt.tight_layout()
    plt.savefig(output_dir / "monte_carlo_rank_change_distribution.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the revised four-component SWS and reviewer-requested robustness analyses."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV produced by the Playwright crawler.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for outputs.")
    parser.add_argument("--random-iter", type=int, default=10000, help="Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)
    df, normalisation, http2_diag = calculate_components(df)
    df, entropy_w, entropy_details, mc_distribution, mc_summary = add_sws_variants(
        df,
        random_iter=args.random_iter,
        seed=args.seed,
    )

    variant_cols = [
        "SWS_equal_weight",
        "SWS_without_CO2",
        "SWS_data_driven",
    ] + [c for c in df.columns if c.startswith("SWS_leaveout_")]
    rank_table = deterministic_rank_metrics(df, variant_cols)
    weight_table = build_weight_table(entropy_w)
    categories = category_table(df)
    component_corr = df[ACTIVE_COMPONENTS].corr(method="spearman").reset_index().rename(columns={"index": "component"})

    df_sorted = df.sort_values("SWS_revised", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(output_dir / "university_sws_revised.csv", index=False, encoding="utf-8-sig")
    df_sorted.to_excel(output_dir / "university_sws_revised.xlsx", index=False)

    outputs = {
        "normalization_parameters": normalisation,
        "http2_diagnostic": http2_diag,
        "sws_exact_weights": weight_table,
        "entropy_weight_details": entropy_details,
        "deterministic_rank_stability": rank_table,
        "monte_carlo_rank_distribution": mc_distribution,
        "monte_carlo_rank_summary": mc_summary,
        "relative_band_distribution": categories,
        "component_correlation_matrix_revised": component_corr,
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        table.to_excel(output_dir / f"{name}.xlsx", index=False)

    save_plots(df_sorted, rank_table, mc_distribution, output_dir)

    print("\nRevised SWS analysis completed.")
    print(f"Input: {input_path.resolve()}")
    print(f"Rows: {len(df_sorted)}")
    print(f"Monte Carlo iterations: {args.random_iter}")
    print(f"Random seed: {args.seed}")
    print("Primary weights:", REVISED_WEIGHTS)
    print("HTTP/2 diagnostic:")
    print(http2_diag.to_string(index=False))
    print("Normalization parameters:")
    print(normalisation.to_string(index=False))
    print("Main outputs written to:", output_dir.resolve())


if __name__ == "__main__":
    main()
