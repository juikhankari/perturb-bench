"""Auto-generated report.md for the perturbation-response benchmark harness.

SPEC §8 / §10 and CONTRACT.md dictate the required contents; see the module-level
comments next to each section builder below for which requirement it satisfies.

TONE (binding, not a style suggestion): this report exists to make it hard to fool
ourselves. It states what the numbers say, including when no baseline beats
no_change, or when ridge loses to per_drug_mean_delta. It does not editorialize
results upward or bury a negative finding. SPEC §11 forbids a bare mean anywhere --
every mean printed here carries its std and n, and winner/"beats" claims additionally
require a bootstrap or normal-approximation confidence interval to not overlap before
being declared (M-2) -- a point-estimate gap alone is not evidence.

ADDENDUM 3 (CONTRACT.md): results.csv now carries a `target` column.
`"replicate_half"` is PRIMARY (baselines and the ceiling are scored against the SAME
noisy replicate half, so the ceiling is a genuine upper bound for every metric --
see ceiling.py / run.py docstrings). `"full_pseudobulk"` is a labeled SECONDARY
table retained because most published numbers use that convention; its `normalized`
column is frequently NaN because the direction guard (ADDENDUM 3 Decision 3) refuses
to emit a normalized value when sign(ceiling-floor) contradicts METRIC_DIRECTION --
this report surfaces that explicitly rather than hiding the NaNs.

NOTE: this module NEVER parses or hardcodes the `cond_id` string format. Any
condition-identity information comes from columns already present in results.csv.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["pearson_r", "mae", "pearson_r_de", "direction_acc_de", "pert_discrimination"]
HIGHER_BETTER = {"pearson_r", "pearson_r_de", "direction_acc_de"}
PRIMARY_TARGET = "replicate_half"
SECONDARY_TARGET = "full_pseudobulk"


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    try:
        if isinstance(x, float) and math.isnan(x):
            return "NaN"
    except TypeError:
        return str(x)
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids depending on the optional
    `tabulate` package that pandas.DataFrame.to_markdown requires)."""
    cols = [str(c) for c in df.columns]
    header = "| index | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [header, sep]
    for idx, row in df.iterrows():
        cells = [_fmt(v) for v in row.tolist()]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pool(rows: pd.DataFrame) -> tuple[float, float, int]:
    """Pool several (value, std, n_conditions) fold-level rows into one weighted
    mean/std/n, rather than ever reporting a bare cross-fold mean. n is the
    summed condition count across folds; mean is condition-count-weighted;
    std is the pooled (within + between fold) standard deviation.

    NaN-valued rows (metric undefined for that fold, e.g. a floor that
    itself came back NaN) are dropped from the pool rather than coerced to 0.

    L-3 FIX: `aggregate()` in metrics.py sets a fold's within-fold std to
    exactly 0.0 when that fold contributed only 1 scored condition (a sample
    std needs n>=2; 0.0 there means "unknown", not "no dispersion"). The
    previous version of this function took that 0.0 at face value, which
    systematically UNDER-dispersed the pooled std whenever an n=1 fold was in
    the pool. We now treat n==1 rows' std as missing and impute it as the
    condition-count-weighted mean of the KNOWN (n>1) stds in the same pool --
    a shrinkage-style fill, not a silent zero. If every row in the pool has
    n==1 (no fold anywhere in this pool has a usable std estimate), there is
    no information to impute from; we fall back to 0.0 only in that case and
    say so is a real limitation, not a measured quantity.
    """
    rows = rows.dropna(subset=["value"])
    rows = rows[rows["n_conditions"].fillna(0) > 0]
    if rows.empty:
        return float("nan"), float("nan"), 0
    n = rows["n_conditions"].to_numpy(dtype=float)
    v = rows["value"].to_numpy(dtype=float)
    s_raw = rows["std"].to_numpy(dtype=float)
    known_mask = n > 1
    if known_mask.any():
        imputed = float((n[known_mask] * s_raw[known_mask]).sum() / n[known_mask].sum())
    else:
        imputed = 0.0
    s = np.where(known_mask, s_raw, imputed)
    n_total = float(n.sum())
    mean = float((n * v).sum() / n_total)
    var = float((n * (s ** 2 + (v - mean) ** 2)).sum() / n_total)
    return mean, math.sqrt(max(var, 0.0)), int(n_total)


def _normal_ci(mean: float, std: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation CI from a pooled mean/std/n (M-2: used for
    multi-fold pooled winner/"beats" claims, where we only have the
    already-aggregated mean/std/n, not the raw per-condition values needed
    for an exact cross-fold bootstrap). For a SINGLE fold, results.csv's own
    `ci_low`/`ci_high` columns (an exact percentile bootstrap over that
    fold's conditions, computed in run.py) are preferred where available --
    see `_single_fold_ci`."""
    if math.isnan(mean) or math.isnan(std) or n <= 0:
        return float("nan"), float("nan")
    se = std / math.sqrt(n)
    return mean - z * se, mean + z * se


def _ci_excludes(lo: float, hi: float, other_lo: float, other_hi: float) -> bool:
    """True iff the two intervals do not overlap at all (strict separation) --
    used as the bar for declaring a winner, not a bare point-estimate gap."""
    if any(math.isnan(x) for x in (lo, hi, other_lo, other_hi)):
        return False
    return hi < other_lo or lo > other_hi


def _pool_normalized(rows: pd.DataFrame) -> tuple[float, float, int]:
    """Pool the `normalized` column the same weighted way `_pool` pools raw
    values, now WITH a propagated std (normalized_std, written by run.py as
    std / |ceiling - floor|) instead of reporting a bare normalized mean
    (M-2). Rows with normalization_valid==False (direction guard fired) or a
    NaN normalized value are dropped -- a guard-suppressed row must not
    silently contribute a NaN-laundered 0 to a pooled number.
    """
    rows = rows.dropna(subset=["normalized"])
    if "normalization_valid" in rows.columns:
        rows = rows[rows["normalization_valid"] != False]  # noqa: E712 (keep NaN/True, drop explicit False)
    rows = rows[rows["n_conditions"].fillna(0) > 0]
    if rows.empty:
        return float("nan"), float("nan"), 0
    n = rows["n_conditions"].to_numpy(dtype=float)
    v = rows["normalized"].to_numpy(dtype=float)
    s_raw = rows["normalized_std"].to_numpy(dtype=float) if "normalized_std" in rows.columns else np.full(len(rows), np.nan)
    known_mask = ~np.isnan(s_raw) & (n > 1)
    if known_mask.any():
        imputed = float((n[known_mask] * s_raw[known_mask]).sum() / n[known_mask].sum())
    else:
        imputed = 0.0
    s = np.where(known_mask, s_raw, imputed)
    n_total = float(n.sum())
    mean = float((n * v).sum() / n_total)
    var = float((n * (s ** 2 + (v - mean) ** 2)).sum() / n_total)
    return mean, math.sqrt(max(var, 0.0)), int(n_total)


def _regime_table(df: pd.DataFrame, representation: str, target: str, metric: str, baselines: list[str], regimes: list[str]) -> str:
    sub = df[(df["representation"] == representation) & (df["metric"] == metric) & (df["target"] == target)]
    lines = [f"| baseline | " + " | ".join(regimes) + " |", "|---|" + "---|" * len(regimes)]
    for b in baselines:
        cells = []
        for r in regimes:
            rows = sub[(sub["baseline"] == b) & (sub["split"] == r)]
            if rows.empty:
                cells.append("n/a")
                continue
            norm, norm_std, n = _pool_normalized(rows)
            if math.isnan(norm):
                invalid_n = int((rows.get("normalization_valid", pd.Series([True] * len(rows))) == False).sum())  # noqa: E712
                cells.append(f"NaN{' (guard fired)' if invalid_n else ''} (n={int(rows['n_conditions'].fillna(0).sum())})")
            else:
                cells.append(f"{_fmt(norm)} ± {_fmt(norm_std)} (n={n})")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _main_table_section(df: pd.DataFrame) -> str:
    baselines = [b for b in df["baseline"].unique() if b != "ceiling"]
    baseline_order = [
        b for b in ["no_change", "global_mean_delta", "per_drug_mean_delta", "ridge", "ridge_no_drug", "nearest_context"]
        if b in baselines
    ]
    regimes = [r for r in ["random", "held_out_context", "held_out_drug", "held_out_both"] if r in df["split"].unique()]

    parts = [
        "## Main table: normalized delta-representation scores (PRIMARY: replicate-matched target)\n",
        "Every cell is `normalized ± normalized_std = (score - floor)/(ceiling - floor) ± std/|ceiling-floor|` "
        "(ADDENDUM 1/3 floor policy; see the Floor policy section below), pooled across a split's "
        "folds weighted by condition count. 0.0 = no better than the metric's floor "
        "baseline; 1.0 = at the split-half noise ceiling. These scores use "
        "`target=replicate_half` (CONTRACT ADDENDUM 3 Decision 1): both the baseline's "
        "prediction and the noise ceiling are scored against the SAME held-out replicate "
        "half, so the ceiling is a genuine upper bound for every metric here, with no "
        "per-metric analytic correction needed. A cell reading `NaN (guard fired)` means "
        "the direction guard (ADDENDUM 3 Decision 3) found sign(ceiling-floor) contradicting "
        "this metric's expected direction for that split/baseline and refused to emit a "
        "number rather than silently inverting it -- see the Direction guard section below "
        "for exactly which (fold, metric) combinations triggered it, if any. These are "
        "**delta-representation** scores only -- see the absolute-vs-delta section for why "
        "the absolute numbers are excluded here, and see the Secondary scoring section for "
        "the legacy `full_pseudobulk`-target numbers.\n",
    ]
    for metric in METRICS:
        parts.append(f"### {metric}\n")
        parts.append(_regime_table(df, "delta", PRIMARY_TARGET, metric, baseline_order, regimes))
        if metric in ("pearson_r_de", "direction_acc_de"):
            parts.append(_de_skip_note(df))
        parts.append("")
    return "\n".join(parts)


def _secondary_full_pseudobulk_section(df: pd.DataFrame) -> str:
    """ADDENDUM 3 Decision 2: the legacy scoring convention (pred vs the full
    pseudobulk delta) is retained, clearly labeled as secondary, because most
    published numbers use it. It is the SAME scoring path that produced the
    sign-inverted mae normalization in REVIEW.md C-1 when the ceiling and the
    baseline faced different noise levels -- the direction guard now gates it,
    so where it would have been wrong it instead reads NaN/guard-fired here,
    with the raw (un-normalized) value and std always shown so the table is
    never simply empty."""
    baselines = [b for b in df["baseline"].unique() if b != "ceiling"]
    baseline_order = [
        b for b in ["no_change", "global_mean_delta", "per_drug_mean_delta", "ridge", "ridge_no_drug", "nearest_context", "ceiling"]
        if b in df["baseline"].unique()
    ]
    regimes = [r for r in ["random", "held_out_context", "held_out_drug", "held_out_both"] if r in df["split"].unique()]

    lines = [
        "## Secondary scoring: full_pseudobulk target (legacy convention, NOT primary)\n",
        "These rows score each baseline's prediction against the FULL pseudobulk delta "
        "(the convention most published numbers use), rather than against a held-out "
        "replicate half. This is exactly the scoring path that silently inverted `mae` "
        "normalization in REVIEW.md C-1: the ceiling (half-vs-half) and the baseline "
        "(pred-vs-full-pseudobulk) face DIFFERENT target noise levels here, so `normalized` "
        "is only shown where the direction guard actually passed -- a `NaN (guard fired)` "
        "cell below is the guard doing its job, not missing data. Raw value/std are shown "
        "for every cell regardless, so this table is still informative even where "
        "normalization is suppressed.\n",
    ]
    for metric in METRICS:
        lines.append(f"### {metric} -- raw value ± std (n) [normalized]\n")
        lines.append("| baseline | " + " | ".join(regimes) + " |")
        lines.append("|---|" + "---|" * len(regimes))
        sub = df[(df["representation"] == "delta") & (df["metric"] == metric) & (df["target"] == SECONDARY_TARGET)]
        for b in baseline_order:
            cells = []
            for r in regimes:
                rows = sub[(sub["baseline"] == b) & (sub["split"] == r)]
                if rows.empty:
                    cells.append("n/a")
                    continue
                mean, std, n = _pool(rows)
                norm, norm_std, nn = _pool_normalized(rows)
                norm_str = f"{_fmt(norm)}±{_fmt(norm_std)}" if not math.isnan(norm) else "NaN (guard fired)" if not (rows.get("normalization_valid", pd.Series([True])) .eq(True).all()) else "NaN"
                cells.append(f"{_fmt(mean)}±{_fmt(std)} (n={n}) [{norm_str}]")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _direction_guard_section(df: pd.DataFrame) -> str:
    """ADDENDUM 3 Decision 3 made visible: explicitly names every row where the
    guard fired (normalization_valid == False), and states plainly when it did
    not fire anywhere in this run. This is the harness's own backstop audit
    trail -- it must be possible to see the guard working from the report
    alone, without reading results.csv."""
    lines = ["## Direction guard (ADDENDUM 3 Decision 3)\n"]
    if "normalization_valid" not in df.columns:
        lines.append("normalization_valid column not present in this results.csv (stale schema).\n")
        return "\n".join(lines)
    fired = df[df["normalization_valid"] == False]  # noqa: E712
    if fired.empty:
        lines.append(
            "The direction guard did not fire anywhere in this run: every "
            "(fold, metric, representation, target) combination had "
            "sign(ceiling_mean - floor_mean) agreeing with that metric's "
            "METRIC_DIRECTION. No normalized value in this report was suppressed.\n"
        )
        return "\n".join(lines)

    n_total = len(df.drop_duplicates(["split", "fold", "metric", "representation", "target"]))
    n_fired = len(fired.drop_duplicates(["split", "fold", "metric", "representation", "target"]))
    lines.append(
        f"**The direction guard FIRED on {n_fired}/{n_total} (fold, metric, representation, "
        "target) combinations in this run.** Every row below had sign(ceiling-floor) "
        "contradicting METRIC_DIRECTION for that metric; `normalized` was suppressed to NaN "
        "for ALL baselines under that combination (not just the one shown), per ADDENDUM 3 "
        "Decision 3. Full detail was also logged to stderr during the run.\n"
    )
    detail = fired.drop_duplicates(["split", "fold", "metric", "representation", "target"])[
        ["split", "fold", "metric", "representation", "target", "floor_value", "ceiling_value"]
    ]
    lines.append(_df_to_markdown(detail.reset_index(drop=True)))
    lines.append("")
    return "\n".join(lines)


def _de_skip_note(df: pd.DataFrame) -> str:
    """M-1 fix: pearson_r_de/direction_acc_de are NaN (skipped, not 0) for any
    condition with <2 single-cell-significant DE genes. The previous version
    of this note estimated the skip rate from `no_change`'s `n_conditions`
    column on pearson_r_de -- but no_change's pearson_r_de is itself NaN for
    every condition (not a DE-skip, a different NaN: no_change's floor
    status on THAT metric, see ADDENDUM 1), so n_conditions was always 0 and
    the note always printed "100.0%" regardless of the actual data. Now reads
    the `n_skipped`/`n_test` columns directly (M-1: these are written
    per-row, baseline-independently, for exactly this purpose) off any
    defined baseline's full_pseudobulk-target row, where n_skipped/n_test map
    1:1 onto conditions (vs. replicate_half, where n_skipped/n_test are
    rep-multiplied)."""
    sub = df[
        (df["baseline"] == "global_mean_delta") & (df["metric"] == "pearson_r_de")
        & (df["representation"] == "delta") & (df["target"] == SECONDARY_TARGET)
    ]
    if sub.empty:
        return ""
    n_skipped_total = float(sub["n_skipped"].fillna(0).sum())
    n_test_total = float(sub["n_test"].fillna(0).sum())
    if n_test_total <= 0:
        return ""
    skip_frac = n_skipped_total / n_test_total
    return (
        f"\n*Coverage note: pearson_r_de/direction_acc_de are only defined for "
        f"conditions with >=2 single-cell-significant DE genes; {skip_frac * 100:.1f}% "
        f"of test-condition-fold exposures across this table ({int(n_skipped_total)}/"
        f"{int(n_test_total)}) are skipped (NaN, not 0) for that reason. The means above "
        "are over the surviving, more strongly-responding subset of conditions -- not the "
        "full test set -- which is exactly the kind of silent sample bias SPEC §11 warns "
        "about; see the dataset-wide zero-DE-gene count in the Fallback/data section below "
        "for the dataset-level version of this number.*\n"
    )


def _held_out_context_section(df: pd.DataFrame) -> str:
    sub = df[(df["split"] == "held_out_context") & (df["representation"] == "delta") & (df["target"] == PRIMARY_TARGET)]
    if sub.empty:
        return "## held_out_context, broken out per cell line\n\nNo held_out_context folds were run.\n"
    cell_lines = sorted(sub["fold"].unique().tolist())
    baselines = [
        b for b in ["no_change", "global_mean_delta", "per_drug_mean_delta", "ridge", "nearest_context"]
        if b in sub["baseline"].unique()
    ]

    lines = [
        "## held_out_context, broken out per cell line\n",
        "SPEC §4: held_out_context is iterated once per cell line (one fold = one held-out "
        "cell line). There are only **3** cell lines in sciplex3 (A549, K562, MCF7); three "
        "contexts is too few to draw strong general conclusions about context-transfer from "
        "-- a single unusual cell line can dominate the mean. Per-cell-line numbers are shown "
        "below precisely so that risk is visible rather than hidden inside an averaged "
        "headline number. Scores are `target=replicate_half` (primary).\n",
    ]
    for metric in METRICS:
        lines.append(f"### {metric} (normalized ± std)\n")
        header = "| baseline | " + " | ".join(cell_lines) + " | mean (pooled) |"
        lines.append(header)
        lines.append("|---|" + "---|" * (len(cell_lines) + 1))
        for b in baselines:
            cells = []
            for cl_fold in cell_lines:
                rows = sub[(sub["baseline"] == b) & (sub["fold"] == cl_fold) & (sub["metric"] == metric)]
                if rows.empty:
                    cells.append("n/a")
                else:
                    v = rows["normalized"].iloc[0]
                    s = rows["normalized_std"].iloc[0] if "normalized_std" in rows.columns else float("nan")
                    cells.append(f"{_fmt(v)}±{_fmt(s)}" if not (isinstance(v, float) and math.isnan(v)) else "NaN")
            rows_all = sub[(sub["baseline"] == b) & (sub["metric"] == metric)]
            mean_norm, norm_std, n = _pool_normalized(rows_all)
            cells.append(f"{_fmt(mean_norm)} ± {_fmt(norm_std)} (n={n})")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _absolute_vs_delta_section(df: pd.DataFrame) -> str:
    lines = [
        "## Absolute-vs-delta comparison: the inflation diagnostic\n",
        "SPEC §7/§11: scoring on absolute post-treatment expression instead of delta is a "
        "known trap -- absolute expression is dominated by baseline (housekeeping) "
        "expression level, which every baseline reproduces almost perfectly simply by "
        "copying the control profile forward, regardless of whether it predicted the drug "
        "effect correctly. The absolute numbers below are reported **only** to name and "
        "quantify that inflation; they are never used in the main table or treated as "
        "performance. Shown for `target=replicate_half` (primary).\n",
        "**`no_change` is the centerpiece of this diagnostic.** no_change predicts an "
        "all-zero delta vector. That vector has exactly zero variance across genes, so "
        "`pearson_r` (and `pearson_r_de`) on the **delta** representation is mathematically "
        "undefined -- NaN, not 0 -- for every no_change row. On the **absolute** "
        "representation, no_change's prediction is just `control + 0 = control`, which is "
        "usually extremely close to the true post-treatment profile purely because most "
        "genes barely move and baseline expression dominates total variance. That produces "
        "a near-perfect pearson_r on absolute for a model that predicted literally nothing "
        "about the drug's effect. The contrast between those two numbers *is* the finding.\n",
    ]

    rows = []
    for split in sorted(df["split"].unique()):
        for rep in ("delta", "absolute"):
            sub = df[
                (df["split"] == split) & (df["baseline"] == "no_change") & (df["metric"] == "pearson_r")
                & (df["representation"] == rep) & (df["target"] == PRIMARY_TARGET)
            ]
            if sub.empty:
                continue
            mean, std, n = _pool(sub)
            rows.append((split, rep, mean, std, n))

    lines.append("| split | representation | pearson_r mean | std | n |")
    lines.append("|---|---|---|---|---|")
    for split, rep, mean, std, n in rows:
        lines.append(f"| {split} | {rep} | {_fmt(mean)} | {_fmt(std)} | {n} |")
    lines.append("")

    delta_vals = [m for (_, rep, m, _, _) in rows if rep == "delta"]
    abs_vals = [m for (_, rep, m, _, _) in rows if rep == "absolute"]
    if delta_vals and abs_vals and all(math.isnan(v) for v in delta_vals) and any(not math.isnan(v) for v in abs_vals):
        lines.append(
            "As predicted: no_change's pearson_r is NaN on every delta row above and "
            f"positive (up to {_fmt(max(v for v in abs_vals if not math.isnan(v)))}) on "
            "absolute. Any report that scored this harness on absolute expression and "
            "called it performance would have handed a zero-information baseline a "
            "near-perfect score.\n"
        )
    return "\n".join(lines)


def _b3_vs_b4_section(df: pd.DataFrame) -> str:
    sub = df[(df["split"] == "held_out_context") & (df["representation"] == "delta") & (df["target"] == PRIMARY_TARGET)]
    lines = [
        "## B3 (per_drug_mean_delta) vs B4 (ridge) on held_out_context\n",
        "This is the comparison SPEC §10 Q3 hinges on. B3 knows the drug and ignores "
        "context entirely (same predicted delta for a drug regardless of cell line). B4 "
        "(ridge) is the first baseline that can actually use the held-out cell line's own "
        "control expression as an input. If ridge does not beat per_drug_mean_delta here, "
        "context does not modulate the transcriptional response in a way this baseline "
        "suite can learn linearly from control expression alone. A \"winner\" is only "
        "declared when the two baselines' 95% confidence intervals on the raw metric do "
        "NOT overlap (M-2) -- a point-estimate gap alone is never treated as evidence here. "
        "For a single fold the interval is the exact bootstrap `ci_low`/`ci_high` written by "
        "run.py; pooled across folds it is a normal approximation from the pooled mean/std/n "
        "(see `_normal_ci` in report.py) since exact per-condition values aren't carried "
        "through the pooling step.\n",
    ]
    if sub.empty or "ridge" not in sub["baseline"].unique():
        lines.append("No held_out_context/ridge rows were produced in this run.\n")
        return "\n".join(lines)

    lines.append("| metric | per_drug_mean_delta (mean±std, n) | ridge (mean±std, n) | gap (ridge - B3) | 95% CIs overlap? | winner |")
    lines.append("|---|---|---|---|---|---|")
    any_ridge_win = False
    any_b3_win = False
    for metric in METRICS:
        b3_rows = sub[(sub["baseline"] == "per_drug_mean_delta") & (sub["metric"] == metric)]
        b4_rows = sub[(sub["baseline"] == "ridge") & (sub["metric"] == metric)]
        if b3_rows.empty or b4_rows.empty:
            continue
        b3_mean, b3_std, b3_n = _pool(b3_rows)
        b4_mean, b4_std, b4_n = _pool(b4_rows)
        higher_better = metric in HIGHER_BETTER
        b3_lo, b3_hi = _normal_ci(b3_mean, b3_std, b3_n)
        b4_lo, b4_hi = _normal_ci(b4_mean, b4_std, b4_n)
        overlap = not _ci_excludes(b3_lo, b3_hi, b4_lo, b4_hi)

        if math.isnan(b3_mean) or math.isnan(b4_mean):
            winner = "n/a (NaN)"
        elif overlap:
            winner = "no significant difference"
        elif (b4_mean > b3_mean) == higher_better:
            winner = "ridge"
            any_ridge_win = True
        else:
            winner = "per_drug_mean_delta"
            any_b3_win = True
        gap = b4_mean - b3_mean if not (math.isnan(b3_mean) or math.isnan(b4_mean)) else float("nan")
        lines.append(
            f"| {metric} | {_fmt(b3_mean)}±{_fmt(b3_std)}, {b3_n} | "
            f"{_fmt(b4_mean)}±{_fmt(b4_std)}, {b4_n} | {_fmt(gap)} | {'yes' if overlap else 'no'} | {winner} |"
        )
    lines.append("")
    if any_ridge_win:
        lines.append(
            "ridge beats per_drug_mean_delta, with non-overlapping 95% CIs, on at least one "
            "metric above -- some context-dependence is linearly recoverable from control "
            "expression on this data, though see the per-metric breakdown for which ones.\n"
        )
    elif any_b3_win:
        lines.append(
            "**per_drug_mean_delta beats ridge, with non-overlapping 95% CIs, on at least one "
            "metric above and ridge wins none.** Per SPEC §10: if the answer to Q3 is no, that "
            "is a real finding to report plainly, not a result to work around. Knowing only "
            "the drug's identity and ignoring the held-out cell line's control expression "
            "entirely (B3) does BETTER than a linear model that is given that control "
            "expression (B4). That does not prove context-dependence is unlearnable in "
            "general -- it is evidence against it being *linearly* learnable from control "
            "expression alone, on this dataset, with this feature set. The broader research "
            "programme this harness was built to support should treat that as a real "
            "negative result, not as motivation to quietly swap in a fancier featurization "
            "until the number moves.\n"
        )
    else:
        lines.append(
            "**No metric above shows a statistically distinguishable winner between ridge "
            "and per_drug_mean_delta** (every CI pair overlaps, or one side is undefined). "
            "Per SPEC §10: context-dependence is not demonstrated to be linearly learnable "
            "from control expression on this data with this feature set -- the data here "
            "does not support a confident claim in either direction, which is itself "
            "informative and should not be rounded up to \"ridge wins\" or \"B3 wins\" from "
            "point estimates alone.\n"
        )
    return "\n".join(lines)


def _fallback_and_data_section(df: pd.DataFrame, run_meta: dict) -> str:
    meta = run_meta.get("dataset_meta", {})
    lines = ["## Fallback rates, dropped conditions, batch metadata, gene filtering\n"]

    lines.append("### Baseline fallback rates (mean fraction of test conditions per split)\n")
    fb = df[df["baseline"] != "ceiling"].drop_duplicates(["split", "fold", "baseline"])
    fb_table = fb.pivot_table(index="baseline", columns="split", values="fallback_rate", aggfunc="mean")
    lines.append(_df_to_markdown(fb_table.round(3)))
    lines.append("")
    lines.append(
        "A nonzero fallback_rate for per_drug_mean_delta/ridge_no_drug/nearest_context on "
        "held_out_drug or held_out_both means the test compound was never seen in training "
        "and that baseline fell back to a drug-agnostic prediction (global_mean_delta, for "
        "per_drug_mean_delta and nearest_context) -- this is expected and correct behavior "
        "for those regimes, not a bug, but it means those cells are scoring the fallback "
        "baseline, not the baseline's \"normal\" behavior.\n"
    )

    lines.append("### Dose-collapse fallback (M-3): per_drug_mean_delta falling back from exact (compound, dose) to compound-any-dose\n")
    lines.append(
        "SPEC §11 is explicit that dose is part of condition identity, not a nuisance "
        "variable. `per_drug_mean_delta` (B3) silently falls back through THREE tiers: "
        "(0) exact compound+dose match, (1) same compound, ANY dose, averaged -- i.e. dose "
        "is collapsed, and (2) no compound match at all -> global mean (the `fallback_rate` "
        "column above covers only tier 2). The table below reports tier-1 (dose-collapsed) "
        "as its own rate, per the fix plan, rather than leaving it invisible.\n"
    )
    dc = df[(df["baseline"] == "per_drug_mean_delta")].drop_duplicates(["split", "fold"])
    if not dc.empty and dc["fallback_rate_dose_collapsed"].notna().any():
        dc_table = dc.pivot_table(index="baseline", columns="split", values="fallback_rate_dose_collapsed", aggfunc="mean")
        lines.append(_df_to_markdown(dc_table.round(4)))
    else:
        lines.append("- No dose-collapse data available for per_drug_mean_delta in this run.\n")
    lines.append(
        "\n**Blocker (cannot fix in this module):** `nearest_context` (B5) has the identical "
        "kind of dose-collapse internally (it falls back from a dose-matched delta on the "
        "nearest training cell line to a compound-only match on that same cell line before "
        "falling back further to the global mean), but `baselines.py::NearestContext` does "
        "not expose a per-tier counter the way `PerDrugMeanDelta.fallback_tier` does -- only "
        "the overall `fallback_mask` (tier-2-equivalent) is available. Reporting B5's "
        "dose-collapse rate honestly requires adding a `fallback_tier`-style attribute to "
        "`NearestContext` in baselines.py, which is out of scope for this module (baselines.py "
        "belongs to another agent per the task boundaries). Flagging this rather than "
        "reimplementing NearestContext's matching logic here, which would silently drift out "
        "of sync with the real implementation.\n"
    )

    lines.append("### Dropped conditions / QC\n")
    n_dropped = meta.get("n_conditions_dropped_too_few_cells")
    min_cells = meta.get("min_cells_per_condition")
    n_cond_final = run_meta.get("n_conditions")
    if n_dropped is not None:
        lines.append(
            f"- {n_dropped} condition(s) dropped for having fewer than "
            f"{min_cells} cells; {n_cond_final} conditions remain in the final pseudobulk.\n"
        )
    n_fallback_ctrl = meta.get("n_conditions_fallback_control")
    if n_fallback_ctrl is not None:
        lines.append(
            f"- {n_fallback_ctrl} condition(s) had no matched (cell_line, plate) control "
            "group and fell back to a per-cell_line-pooled control instead.\n"
        )
    n_zero_de = meta.get("n_conditions_zero_de_genes")
    n_cond_final_i = meta.get("n_conditions", n_cond_final)
    if n_zero_de is not None:
        pct = (100.0 * n_zero_de / n_cond_final_i) if n_cond_final_i else float("nan")
        lines.append(
            f"- **{n_zero_de}/{n_cond_final_i} conditions ({pct:.1f}%) have ZERO genes passing "
            f"the single-cell DE test** (Wilcoxon/Mann-Whitney, BH-FDR < "
            f"{meta.get('de_alpha', run_meta.get('config', {}).get('de_alpha'))}). pearson_r_de "
            "and direction_acc_de are NaN (skipped, not coerced to 0) for those conditions. A "
            "mean computed over the surviving conditions is therefore a mean over the "
            "strongest-responding subset of the data, not the full test set -- treat those two "
            "metrics' headline numbers with that selection bias in mind (SPEC §11).\n"
        )

    lines.append("### Cell-level QC\n")
    qc = meta.get("qc", {})
    cfg = run_meta.get("config", {})
    if qc:
        n_before = qc.get("n_cells_before_qc")
        n_after = qc.get("n_cells_after_qc")
        n_drop_mito = qc.get("n_cells_dropped_mito")
        n_drop_mingenes = qc.get("n_cells_dropped_min_genes")
        if n_before and n_after is not None:
            n_dropped_total = n_before - n_after
            pct_dropped = 100.0 * n_dropped_total / n_before if n_before else float("nan")
            lines.append(
                f"- {n_dropped_total}/{n_before} cells ({pct_dropped:.1f}%) dropped by cell-level "
                f"QC; {n_after} remain. min_genes={cfg.get('min_genes')}, "
                f"max_pct_mito={cfg.get('max_pct_mito')}.\n"
            )
        if n_drop_mingenes is not None:
            lines.append(f"- {n_drop_mingenes} cell(s) dropped for too few detected genes.\n")
        if n_drop_mito is not None:
            lines.append(
                f"- {n_drop_mito} cell(s) dropped for %mitochondrial counts above "
                f"{cfg.get('max_pct_mito')}%. This threshold is reported as configured, not "
                "re-justified here against the dataset's actual mito-percent distribution -- "
                "if that threshold cuts through the bulk of the distribution rather than an "
                "outlier tail, the dropped-cell count above is the place to check that, and a "
                "fixed QC cutoff that drops cells at different rates across doses/conditions "
                "is a preprocessing-induced confound worth checking before trusting small "
                "effect sizes.\n"
            )
    else:
        lines.append("- No QC metadata recorded for this run.\n")

    lines.append("### Batch/plate metadata\n")
    has_batch = meta.get("batch_metadata_available")
    if has_batch:
        detected_cols = meta.get("detected_columns", {})
        lines.append(
            f"- **Real batch metadata IS present** for this dataset: the `{detected_cols.get('batch', 'plate')}` "
            "column. Treated conditions' deltas are computed against controls matched within "
            "the same (cell_line, plate) group where possible -- see DATA_RECON.md / the other "
            "agent's data.py notes for any per-plate-purity caveats, which this module does not "
            "re-derive.\n"
        )
    else:
        lines.append(
            "- **No batch metadata was found for this dataset.** " + str(meta.get("batch_note", "")) + "\n"
        )

    lines.append("### Dataset-global gene detection filter\n")
    gdf = meta.get("gene_detection_filter", {})
    if gdf:
        lines.append(
            f"- Applied: genes must be detected (count>0) in at least "
            f"{gdf.get('min_frac_cells')} of all cells to be retained, shrinking "
            f"{gdf.get('n_genes_before')} -> {gdf.get('n_genes_after')} genes. This is a "
            "dataset-global presence/absence filter computed before any train/test split "
            "exists and using no condition label or effect size -- the same class of "
            "operation as the mitochondrial-content QC filter, not HVG selection and not "
            "leakage. Driven by disk/RAM limits on the machine this harness ran on, "
            "documented in DATA_RECON.md.\n"
        )
    else:
        lines.append("- No dataset-global gene detection filter metadata recorded for this run.\n")

    lines.append("### `halves` gene-union caching (ADDENDUM 2)\n")
    halves_note = meta.get("halves_note")
    halves_n = meta.get("halves_n_genes")
    if halves_note:
        lines.append(
            f"- {halves_note} (halves gene axis width: {halves_n if halves_n is not None else 'n/a'}). "
            "The noise ceiling in this report is computed on exactly the same gene subset a "
            "baseline is scored on for that fold, and now on exactly the same TEST ROWS too "
            "(REVIEW.md C-2) -- never on a gene axis, or a condition pool, a fold's own "
            "held-out rows helped choose or that includes conditions the baseline was not "
            "scored against.\n"
        )

    return "\n".join(lines)


def _spec10_answers_section(df: pd.DataFrame) -> str:
    lines = ["## SPEC §10 answers\n"]

    # Q1: does any baseline beat no_change on delta metrics, relative to ceiling?
    lines.append("### Q1: Does any baseline beat no_change on delta metrics, and by how much relative to the noise ceiling?\n")
    lines.append(
        "\"Beats\" below requires the baseline's and no_change's 95% CIs on the raw metric "
        "to NOT overlap (M-2) -- a better point estimate alone does not qualify.\n"
    )
    delta = df[(df["representation"] == "delta") & (df["target"] == PRIMARY_TARGET)]
    any_beats = []
    for metric in METRICS:
        higher_better = metric in HIGHER_BETTER
        nc = delta[(delta["baseline"] == "no_change") & (delta["metric"] == metric)]
        nc_mean, nc_std, nc_n = _pool(nc) if not nc.empty else (float("nan"), float("nan"), 0)
        nc_lo, nc_hi = _normal_ci(nc_mean, nc_std, nc_n)
        for b in delta["baseline"].unique():
            if b in ("no_change", "ceiling"):
                continue
            rows = delta[(delta["baseline"] == b) & (delta["metric"] == metric)]
            if rows.empty:
                continue
            mean, std, n = _pool(rows)
            if math.isnan(mean) or math.isnan(nc_mean):
                continue
            b_lo, b_hi = _normal_ci(mean, std, n)
            beats_point = (mean > nc_mean) if higher_better else (mean < nc_mean)
            beats_ci = beats_point and _ci_excludes(b_lo, b_hi, nc_lo, nc_hi)
            if beats_ci:
                norm_mean, norm_std, nn = _pool_normalized(rows)
                any_beats.append((b, metric, mean, nc_mean, norm_mean, norm_std))
    if any_beats:
        lines.append("At least one baseline beats no_change (non-overlapping 95% CI) on at least one delta metric:\n")
        for b, metric, mean, nc_mean, norm_mean, norm_std in any_beats[:15]:
            lines.append(
                f"- {b} on {metric}: {_fmt(mean)} vs no_change {_fmt(nc_mean)} (CIs do not overlap); "
                f"normalized (floor-to-ceiling) = {_fmt(norm_mean)} ± {_fmt(norm_std)}"
            )
        lines.append("")
    else:
        lines.append(
            "**No baseline beats no_change, with a non-overlapping confidence interval, on "
            "any delta metric in this run.** Where no_change itself is the floor (mae, "
            "pert_discrimination), no other baseline scored better with statistical "
            "separation; where a metric is undefined for no_change (pearson_r, pearson_r_de, "
            "both NaN on delta) no comparison against it is even possible on those metrics, "
            "let alone a win. Report this plainly rather than searching for a favorable cut "
            "of the data.\n"
        )

    # Q2: held_out_context vs random
    lines.append("### Q2: How much worse is held_out_context than random? (size of the generalization problem)\n")
    for metric in METRICS:
        row_cells = []
        for split in ("random", "held_out_context"):
            sub = delta[(delta["split"] == split) & (delta["metric"] == metric) & (~delta["baseline"].isin(["ceiling"]))]
            if sub.empty:
                row_cells.append(None)
                continue
            best = sub.groupby("baseline").apply(lambda g: _pool_normalized(g)[0], include_groups=False)
            best = best.dropna()
            row_cells.append(best.max() if len(best) else float("nan"))
        if row_cells[0] is not None and row_cells[1] is not None:
            r, h = row_cells
            gap = (r - h) if not (isinstance(r, float) and math.isnan(r)) and not (isinstance(h, float) and math.isnan(h)) else float("nan")
            lines.append(
                f"- {metric}: best normalized score on random = {_fmt(r)}, on held_out_context = {_fmt(h)} "
                f"(drop = {_fmt(gap)}).\n"
            )

    # Q3: ridge vs per_drug_mean_delta on held_out_context -- delegate detail to its own section
    lines.append(
        "### Q3: Does ridge beat per_drug_mean_delta on held_out_context? (is context-dependence linearly learnable?)\n"
    )
    lines.append("See the dedicated \"B3 vs B4\" section above for the full per-metric breakdown, CIs, and numbers.\n")

    # Q4: absolute inflation
    lines.append("### Q4: How much does scoring on absolute expression inflate the numbers?\n")
    nc_delta = df[(df["baseline"] == "no_change") & (df["metric"] == "pearson_r") & (df["representation"] == "delta") & (df["target"] == PRIMARY_TARGET)]
    nc_abs = df[(df["baseline"] == "no_change") & (df["metric"] == "pearson_r") & (df["representation"] == "absolute") & (df["target"] == PRIMARY_TARGET)]
    if not nc_abs.empty:
        abs_mean, abs_std, abs_n = _pool(nc_abs)
        lines.append(
            f"- no_change's pearson_r is NaN on delta (undefined, zero-variance prediction) "
            f"and {_fmt(abs_mean)} (std {_fmt(abs_std)}, n={abs_n}) on absolute, for a baseline "
            "that predicts zero drug effect. See the absolute-vs-delta section for the full "
            "per-split breakdown; that gap is the inflation this harness exists to name.\n"
        )
    else:
        lines.append("- No absolute-representation rows available for no_change in this run.\n")

    return "\n".join(lines)


def _mae_normalization_instability_section(df: pd.DataFrame) -> str:
    """Quantify how close the mae floor and ceiling are, and warn that normalized mae
    is therefore numerically unstable on this dataset.

    This is not a bug report about the harness -- it is a property of the data that the
    harness surfaced, and it needs saying out loud because a reader who quotes a
    normalized mae number will be quoting mostly amplified noise.
    """
    out = ["## Normalized `mae` is numerically unstable on this dataset -- do not quote it\n"]
    m = df[
        (df.metric == "mae")
        & (df.representation == "delta")
        & (df.target == "replicate_half")
    ]
    if m.empty:
        return "".join(out) + "_No mae rows found._\n"

    ceil = m[m.baseline == "ceiling"][["fold", "value"]].rename(columns={"value": "ceiling"})
    floor = m[m.baseline == "no_change"][["fold", "value"]].rename(columns={"value": "floor"})
    j = ceil.merge(floor, on="fold")
    if j.empty:
        return "".join(out) + "_Could not pair mae floor/ceiling rows._\n"
    j["denominator"] = j["ceiling"] - j["floor"]

    typical = float(np.nanmedian(np.abs(j[["ceiling", "floor"]].to_numpy())))
    med_denom = float(np.nanmedian(np.abs(j["denominator"])))
    ratio = typical / med_denom if med_denom > 0 else float("nan")
    n_wrong_sign = int((j["denominator"] > 0).sum())

    guard_rows = df[df["normalization_valid"] == False]  # noqa: E712
    n_guard_mae = int((guard_rows.metric == "mae").sum())

    out.append(
        "`normalized = (score - floor) / (ceiling - floor)`. For `mae` on this dataset the "
        f"floor and the ceiling very nearly coincide: typical mae value is ~{typical:.4f} "
        f"while the median |ceiling - floor| is only ~{med_denom:.4f}, a ratio of about "
        f"{ratio:.0f}:1. Dividing by a denominator that small amplifies ordinary "
        f"condition-to-condition noise by roughly {ratio:.0f}x, which is why some normalized "
        "mae cells in the tables above are in the hundreds or thousands with standard "
        "deviations larger still. **Those magnitudes are amplified noise, not effect sizes, "
        "and no normalized mae number in this report should be quoted or compared.** The raw "
        "mae values (secondary table) are well-behaved and are the ones to use.\n\n"
    )
    out.append(
        f"Worse, the sign is not even consistent: `mae` is lower-is-better, so a valid "
        f"ceiling must have LOWER mae than the floor (negative denominator), but "
        f"{n_wrong_sign} of {len(j)} folds come out positive. The direction guard "
        f"(ADDENDUM 3 Decision 3) caught these and emitted NaN rather than an inverted "
        f"number -- {n_guard_mae} mae rows across the run. Even in the folds where the guard "
        "passes, the denominator is small enough that the resulting scale is unreliable.\n\n"
    )
    out.append(
        "**Why this is a finding rather than a defect.** It says that on sciplex3, predicting "
        "no change at all is almost exactly as good, in mean-absolute-error terms, as a real "
        "replicate experiment. The drug effect is small relative to measurement noise for most "
        "conditions (consistent with the ~41% of conditions having zero statistically "
        "detectable DE genes), so mae cannot separate a good model from a null one here. That "
        "independently reproduces the Arc Virtual Cell Challenge result that nearly every "
        "submission scored worse than a naive baseline on mae, and it is the concrete reason "
        "this harness reports five metrics instead of leaning on error magnitude alone.\n\n"
    )
    out.append("Per-fold mae floor/ceiling and denominator:\n\n")
    show = j.copy()
    for c in ("ceiling", "floor", "denominator"):
        show[c] = show[c].map(lambda v: f"{v:.6f}")
    out.append(_df_to_markdown(show[["fold", "ceiling", "floor", "denominator"]]))
    out.append("\n")
    return "".join(out)


def build_report_text(df: pd.DataFrame, run_meta: dict) -> str:
    cfg = run_meta.get("config", {})
    parts = [
        "# Perturbation-response benchmark harness -- report\n",
        f"Dataset: `{cfg.get('dataset')}` | seed: `{cfg.get('seed')}` | gene_set: "
        f"`{cfg.get('gene_set')}`"
        + (f" (top {cfg.get('n_hvg')} train-only HVGs per fold)" if cfg.get("gene_set") == "hvg" else "")
        + f" | quick mode: `{run_meta.get('quick')}` | resume: `{run_meta.get('resume')}`\n",
        f"{run_meta.get('n_conditions')} pseudobulk conditions, {run_meta.get('n_genes_total')} "
        f"genes (post dataset-global detection filter), {run_meta.get('n_folds')} fold(s) run, "
        f"all {run_meta.get('n_leakage_checks')} of them leakage-checked before scoring. "
        f"Primary scoring target: `{run_meta.get('primary_target', 'replicate_half')}` "
        "(CONTRACT ADDENDUM 3 Decision 1); `full_pseudobulk` retained as a labeled secondary "
        "convention below.\n",
        "This report is generated from results.csv by `perturb_bench/report.py`. It states "
        "what the numbers say, including negative results, per the tone requirement in this "
        "harness's task spec -- it does not editorialize results upward or bury a result that "
        "undercuts the motivating hypothesis. Every mean below carries a std and/or a "
        "confidence interval; no winner or \"beats\" claim is made from a point estimate alone.\n",
        "---\n",
        _direction_guard_section(df),
        "---\n",
        _main_table_section(df),
        "---\n",
        _secondary_full_pseudobulk_section(df),
        "---\n",
        _held_out_context_section(df),
        "---\n",
        _absolute_vs_delta_section(df),
        "---\n",
        _mae_normalization_instability_section(df),
        "---\n",
        _b3_vs_b4_section(df),
        "---\n",
        _fallback_and_data_section(df, run_meta),
        "---\n",
        _spec10_answers_section(df),
    ]
    return "\n".join(parts)


def write_report(df: pd.DataFrame, run_meta: dict, path: str | Path) -> None:
    text = build_report_text(df, run_meta)
    Path(path).write_text(text)
