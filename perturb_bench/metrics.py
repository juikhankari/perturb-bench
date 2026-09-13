"""Metric suite for the perturbation-response benchmark harness.

SPEC §7 / CONTRACT.md. Implements the per-condition metric suite, aggregation with
mandatory variance reporting, and floor-to-ceiling normalization.

IMPORTANT — "absolute" representation is a DIAGNOSTIC, not a performance measure.
`score_conditions(..., representation="absolute")` scores (control + pred) against
(control + true) and exists ONLY to demonstrate metric inflation from baseline
expression dominating the signal (SPEC §7, §11). Never report numbers from the
"absolute" representation as model performance in a results table — they belong in
an explicitly-labeled inflation-diagnostic section only.

This module takes plain numpy arrays. It does not import `perturb_bench.data` at
module load time (data.py is owned by another module/agent and may be mid-edit);
any reference to PseudobulkData is confined to TYPE_CHECKING so this module always
imports cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, rankdata

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    from perturb_bench.data import PseudobulkData  # noqa: F401


METRIC_DIRECTION: dict[str, str] = {
    "pearson_r": "higher_better",
    "mae": "lower_better",
    "pearson_r_de": "higher_better",
    "direction_acc_de": "higher_better",
    "pert_discrimination": "lower_better",
}

_METRIC_NAMES = tuple(METRIC_DIRECTION.keys())


def _pearson_r_row(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D vectors. NaN if either is constant or too short."""
    if a.size < 2:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    r, _ = pearsonr(a, b)
    return float(r)


def _direction_acc(pred_de: np.ndarray, true_de: np.ndarray) -> float:
    """Fraction of DE genes where sign(pred) == sign(true).

    Zero handling: np.sign(0) == 0. A predicted exact zero only "matches" a true
    exact zero (sign 0 == sign 0); it does NOT match a true nonzero value (sign 0 !=
    sign +/-1). This is deliberate: an all-zero predictor (B1, no_change) should not
    be rewarded for "agreeing" with nonzero true DE effects just because it emits no
    sign at all. In the overwhelmingly common case that a true DE gene has a nonzero
    true delta, a zero prediction counts as a miss, so B1 scores at or below chance
    on this metric rather than being accidentally inflated.
    """
    pred_sign = np.sign(pred_de)
    true_sign = np.sign(true_de)
    return float(np.mean(pred_sign == true_sign))


def _pert_discrimination(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Normalized rank (in [0, 1]) of the correct true profile among all test true
    profiles, when ranking by distance to each prediction. 0 = prediction is nearest
    to its own true profile (perfect discrimination); ~0.5 = chance / indistinguishable.

    Distance metric: Euclidean (L2) distance in gene space on the delta (or absolute,
    per the caller's representation) vectors actually passed in. Euclidean is chosen
    over correlation-distance because it is not shift/scale invariant, so it actually
    penalizes predictions that get the magnitude of the effect wrong too, not just its
    shape — and it is computed here, not cosine/correlation, because we already score
    `pearson_r` separately and want this metric to carry different information.

    Tie handling: ranks are computed with average-rank-over-ties (scipy `rankdata`,
    method="average"). This matters because a baseline like B1 (no_change) predicts
    an IDENTICAL vector for every condition, so every row of `pred` is equal and every
    condition is equidistant from every other condition's true profile. A naive
    argsort-based rank would assign rank 0 to the first tied element it encounters,
    silently handing such a baseline a near-perfect score. With average-rank ties,
    a fully-tied distance matrix gives every condition the SAME average rank, which
    normalizes to ~0.5 (the middle of [0,1]) — chance-level, not perfect. This is
    exactly the behavior required by the B1 test: identical predictions must score
    ~0.5, not ~0.

    Returns an array of length n_test, one normalized rank per condition (the "row"
    being ranked is "distance from pred[i] to every true[j]"; its own true[i] rank is
    reported).

    VECTORIZED (fix #15 / REVIEW.md L-2/"fix 15"): the original implementation looped
    over i and materialized a full (n, n_genes) `true - pred[i]` broadcast temp per
    iteration -- a 36 MB temp at n=2233 that falls out of cache and makes this call
    O(n^2 * g) with a large constant (20.2s measured on the real cache). This computes
    the full (n, n) pairwise Euclidean distance matrix in one `cdist` call, then ranks
    EACH ROW independently with `rankdata(..., axis=1)` (scipy >= 1.10; pinned to 1.18
    here, see requirements.lock.txt) -- mathematically identical to the per-row loop,
    average-rank ties included, just computed as one vectorized pass instead of n
    separate ones.
    """
    n = pred.shape[0]
    if n <= 1:
        return np.full(n, np.nan)
    dists = cdist(pred, true, metric="euclidean")  # dists[i, j] = ||pred_i - true_j||
    ranks = rankdata(dists, method="average", axis=1)  # 1-indexed, ties averaged, per row
    own_rank = ranks[np.arange(n), np.arange(n)]  # rank of true_i among distances from pred_i
    # normalized rank in [0, 1]: 0 means nearest (best), 1 means farthest (worst)
    return (own_rank - 1.0) / (n - 1.0)


def score_conditions(
    pred: np.ndarray,
    true: np.ndarray,
    control: np.ndarray,
    de_mask: np.ndarray,
    representation: str,
) -> pd.DataFrame:
    """Score predicted vs true deltas condition-by-condition.

    Parameters
    ----------
    pred, true : (n_test, n_genes) predicted / true DELTA arrays.
    control    : (n_test, n_genes) matched control expression, used only to build the
                 "absolute" representation.
    de_mask    : (n_test, n_genes) bool, True where a gene is DE for that condition.
                 Must come from data.py's single-cell significance test — NEVER
                 recompute DE genes here by ranking true deltas (SPEC §11 trap).
    representation : "delta" -> score pred vs true directly (the performance numbers).
                      "absolute" -> score (control+pred) vs (control+true). DIAGNOSTIC
                      ONLY, see module docstring. Must never be reported as performance.

    Returns
    -------
    pd.DataFrame, one row per test condition (same order as input arrays), columns =
    metric names (`_METRIC_NAMES`). NaN where a metric is undefined for that condition
    (e.g. <2 DE genes for pearson_r_de/direction_acc_de).
    """
    if representation not in ("delta", "absolute"):
        raise ValueError(f"representation must be 'delta' or 'absolute', got {representation!r}")

    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    control = np.asarray(control, dtype=float)
    de_mask = np.asarray(de_mask, dtype=bool)

    n_test = pred.shape[0]
    if representation == "absolute":
        score_pred = control + pred
        score_true = control + true
    else:
        score_pred = pred
        score_true = true

    rows = []
    for i in range(n_test):
        p, t = score_pred[i], score_true[i]
        row = {}
        row["pearson_r"] = _pearson_r_row(p, t)
        row["mae"] = float(np.mean(np.abs(p - t)))

        mask = de_mask[i]
        n_de = int(mask.sum())
        if n_de < 2:
            row["pearson_r_de"] = np.nan
            row["direction_acc_de"] = np.nan
        else:
            row["pearson_r_de"] = _pearson_r_row(p[mask], t[mask])
            row["direction_acc_de"] = _direction_acc(p[mask], t[mask])
        rows.append(row)

    df = pd.DataFrame(rows, columns=list(_METRIC_NAMES[:4]))

    # pert_discrimination is computed jointly across all test conditions (it ranks
    # against the full held-out set), not per-row independently, so it's added after.
    df["pert_discrimination"] = _pert_discrimination(score_pred, score_true)

    return df[list(_METRIC_NAMES)]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a score_conditions() DataFrame to mean/std/n/n_skipped per metric.

    SPEC §11: never report a mean without its variance. std is mandatory here, not
    optional. n is the number of non-NaN conditions contributing to the mean; n_skipped
    is the number of conditions for which that metric was NaN (most commonly
    pearson_r_de / direction_acc_de skipped for <2 DE genes).

    Returns a DataFrame indexed by metric name with columns: mean, std, n, n_skipped.
    """
    n_total = len(df)
    out = {}
    for col in df.columns:
        vals = df[col].to_numpy(dtype=float)
        finite = vals[~np.isnan(vals)]
        n = finite.size
        out[col] = {
            "mean": float(np.mean(finite)) if n > 0 else np.nan,
            "std": float(np.std(finite, ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan),
            "n": n,
            "n_skipped": n_total - n,
        }
    return pd.DataFrame(out).T[["mean", "std", "n", "n_skipped"]]


def bootstrap_ci_mean(
    values: np.ndarray, seed: int, n_boot: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values` over CONDITIONS (M-2 / SPEC
    §11: "never report a mean across conditions without reporting variance" --
    `aggregate()` already gives std; this is the companion interval estimate used
    for winner/"beats" claims in report.py, where a point-estimate gap is not
    evidence of anything by itself).

    NaN entries are dropped before resampling (consistent with `aggregate()`'s
    n/n_skipped bookkeeping -- a metric that is undefined for a condition,
    e.g. <2 DE genes, is excluded, not coerced to 0). Deterministic given `seed`
    (np.random.default_rng(seed)), so callers must derive a stable seed per
    row/metric/fold if they need run-to-run byte-identical results.csv.

    Returns (nan, nan) if fewer than 2 finite values remain (cannot resample
    meaningfully); returns (value, value) for exactly 1 finite value (a
    degenerate, zero-width interval rather than fabricating spread from n=1).
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def normalize_score(score: float, floor: float, ceiling: float) -> float:
    """normalized = (score - floor) / (ceiling - floor), SPEC §7.

    Sign-agnostic: works unchanged whether the metric is higher-is-better (ceiling >
    floor, e.g. pearson_r) or lower-is-better (ceiling < floor, e.g. mae,
    pert_discrimination where a perfect score is 0 and floor/no_change sits higher).
    The formula brackets `score` between floor and ceiling from whichever side they
    fall on: score==floor -> 0.0, score==ceiling -> 1.0, regardless of which of
    floor/ceiling is numerically larger.

    Returns NaN (not inf) when ceiling == floor (degenerate: no dynamic range to
    normalize against).
    """
    denom = ceiling - floor
    if denom == 0:
        return float("nan")
    return float((score - floor) / denom)
