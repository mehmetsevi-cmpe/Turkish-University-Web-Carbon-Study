"""
Correlation and point-by-point statistical outputs for the revised four-component SWS.

Run after sws_updated_http2_removed.py:
    python korelasyon_updated_http2_removed.py \
        --input revised_outputs/university_sws_revised.csv \
        --output-dir revised_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

ACTIVE_COMPONENTS = ["score_co2", "score_media", "score_reqs", "score_third"]
SWS_VARIANTS = ["SWS_revised", "SWS_equal_weight", "SWS_without_CO2", "SWS_data_driven"]


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> Tuple[float, float]:
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan
    if method == "pearson":
        r, p = pearsonr(tmp["x"], tmp["y"])
    elif method == "spearman":
        res = spearmanr(tmp["x"], tmp["y"])
        r, p = res.statistic, res.pvalue
    elif method == "kendall":
        res = kendalltau(tmp["x"], tmp["y"])
        r, p = res.statistic, res.pvalue
    else:
        raise ValueError(method)
    return float(r), float(p)


def _interpret_corr(sws_col: str) -> str:
    if sws_col == "SWS_revised":
        return (
            "Expected internal association; not independent validation because the "
            "data-transfer-based CO2 proxy is a component of the revised SWS."
        )
    if sws_col == "SWS_without_CO2":
        return "Circularity check using a CO2-proxy-excluded SWS formulation."
    return "Sensitivity analysis under an alternative scoring assumption."


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sws_col in [c for c in SWS_VARIANTS if c in df.columns]:
        for method in ["pearson", "spearman", "kendall"]:
            coef, p = _safe_corr(df["final_co2_grams"], df[sws_col], method)
            rows.append(
                {
                    "x": "final_co2_grams (standardised data-transfer-based emission proxy)",
                    "y": sws_col,
                    "method": method,
                    "coefficient": coef,
                    "p_value": p,
                    "interpretation_note": _interpret_corr(sws_col),
                }
            )
    return pd.DataFrame(rows)


def partial_spearman(df: pd.DataFrame, x: str, y: str, covars: List[str]) -> Dict[str, float]:
    """Rank-residual implementation of exploratory partial Spearman association."""
    cols = [x, y] + covars
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(data)
    if n <= len(covars) + 3:
        return {"x": x, "y": y, "n": n, "coefficient": np.nan, "p_value": np.nan, "df": np.nan}

    ranked = data.rank(method="average")
    x_cov = ranked[covars].to_numpy()
    rx = ranked[x].to_numpy()
    ry = ranked[y].to_numpy()

    model_x = LinearRegression().fit(x_cov, rx)
    model_y = LinearRegression().fit(x_cov, ry)
    resid_x = rx - model_x.predict(x_cov)
    resid_y = ry - model_y.predict(x_cov)

    r, _ = pearsonr(resid_x, resid_y)
    dof = n - len(covars) - 2
    if dof > 0 and abs(r) < 1:
        t_stat = r * np.sqrt(dof / max(1e-15, 1 - r**2))
        p = 2 * stats.t.sf(abs(t_stat), dof)
    else:
        p = np.nan

    return {
        "x": x,
        "y": y,
        "controlled_variables": ", ".join(covars),
        "n": n,
        "coefficient": float(r),
        "p_value": float(p) if not np.isnan(p) else np.nan,
        "df": int(dof),
        "interpretation_note": (
            "Exploratory conditional association among mathematically coupled variables; "
            "not an independent or causal effect."
        ),
    }


def partial_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    covars = ["total_requests", "image_requests", "video_requests", "third_party_bytes"]
    if "SWS_revised" not in df.columns:
        return pd.DataFrame()
    return pd.DataFrame([partial_spearman(df, "final_co2_grams", "SWS_revised", covars)])


def rank_stability_table(df: pd.DataFrame, base_col: str = "SWS_revised") -> pd.DataFrame:
    variant_cols = [c for c in SWS_VARIANTS if c in df.columns and c != base_col]
    variant_cols += [c for c in df.columns if c.startswith("SWS_leaveout_")]

    base_top10 = set(df.nlargest(10, base_col).index)
    base_bottom10 = set(df.nsmallest(10, base_col).index)
    base_ranks = df[base_col].rank(method="average", ascending=False).to_numpy()

    rows = []
    for col in variant_cols:
        sp = spearmanr(df[base_col], df[col], nan_policy="omit")
        kt = kendalltau(df[base_col], df[col], nan_policy="omit")
        top10 = set(df.nlargest(10, col).index)
        bottom10 = set(df.nsmallest(10, col).index)
        ranks = df[col].rank(method="average", ascending=False).to_numpy()
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
                "interpretation": (
                    "Stability relative to the revised heuristic baseline; this does not validate the baseline score."
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_component_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in ACTIVE_COMPONENTS:
        col = f"SWS_leaveout_{comp.replace('score_', '')}"
        if col not in df.columns:
            continue
        sp = spearmanr(df["SWS_revised"], df[col], nan_policy="omit")
        kt = kendalltau(df["SWS_revised"], df[col], nan_policy="omit")
        rows.append(
            {
                "excluded_component": comp,
                "leaveout_score_column": col,
                "spearman_with_revised": float(sp.statistic),
                "kendall_with_revised": float(kt.statistic),
                "mean_leaveout_score": float(df[col].mean()),
                "std_leaveout_score": float(df[col].std()),
                "interpretation": (
                    "Lower rank similarity indicates greater sensitivity to excluding this component."
                ),
            }
        )
    return pd.DataFrame(rows)


def component_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[ACTIVE_COMPONENTS].corr(method="spearman").reset_index().rename(columns={"index": "component"})


def write_reviewer_response_points(output_path: Path) -> None:
    text = """# Reviewer 1 statistical response notes (revised four-component SWS)

1. HTTP/2 is excluded from the primary SWS because it is constant and carries no discriminatory information.
2. The revised primary weights are CO2 proxy 0.50, media 0.25, requests 0.125, and third-party dependency 0.125.
3. Correlations with the revised SWS describe expected internal association, not independent validation.
4. Partial Spearman is interpreted as an exploratory conditional association among mathematically coupled variables, not an independent or causal effect.
5. Deterministic sensitivity comparisons assess stability relative to the revised heuristic baseline; they do not validate that baseline.
6. Monte Carlo uncertainty must be reported using the distribution and quantiles of rank correlations, overlap ratios, and rank changes, rather than only an average random-weight score.
7. The calculated CO2 variable should be called a standardised data-transfer-based emission proxy rather than an actual carbon-footprint measurement.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation analyses for revised SWS.")
    parser.add_argument("--input", default="university_sws_revised.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path.resolve()}\n"
            "Run sws_updated_http2_removed.py first."
        )

    df = pd.read_csv(input_path)
    outputs = {
        "correlation_results_revised": correlation_table(df),
        "partial_correlation_results_revised": partial_correlation_table(df),
        "rank_stability_table_revised": rank_stability_table(df),
        "leave_one_component_out_revised": leave_one_component_table(df),
        "component_correlation_matrix_revised": component_correlation_table(df),
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        table.to_excel(output_dir / f"{name}.xlsx", index=False)

    write_reviewer_response_points(output_dir / "reviewer_response_statistical_notes.md")
    print("Revised correlation analyses completed.")
    print("Outputs:", output_dir.resolve())


if __name__ == "__main__":
    main()
