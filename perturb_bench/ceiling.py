"""Noise ceiling estimation — SPEC §6 (MANDATORY), CONTRACT.md.

The noise ceiling answers "what is the best achievable score given measurement
noise alone" by scoring one split-half pseudobulk against the other, with the SAME
metric suite on the SAME gene subset a baseline would be scored on. This is what
makes a raw correlation number interpretable (SPEC §6: "without it a correlation of
0.4 is uninterpretable") and what `normalize_score` in metrics.py uses as the
ceiling endpoint.

CONSERVATISM NOTE (read before using this number anywhere else in the harness):
Each half is built from only HALF the cells of a condition, so each half-pseudobulk
is noisier than the full pseudobulk that a baseline is actually scored against. That
means split-half agreement is systematically an UNDERESTIMATE of the true achievable
ceiling (the full-data-vs-full-data agreement would be higher). This estimate is
therefore CONSERVATIVE: a model that clears this ceiling would clear the true one
too, but a model capped below it cannot be cleanly declared "at the measurement
floor" without caveat.

A Spearman-Brown-style correction (reliability_full = 2*r_half / (1 + r_half)) is the
standard fix for correlation-type reliability estimates (it projects a half-sample
correlation up to what the full-sample correlation would be) and WOULD be applicable
to `pearson_r` / `pearson_r_de` here. We deliberately do NOT apply it.

CORRECTION (CONTRACT.md ADDENDUM 3, binding, withdraws a claim in the previous version
of this docstring): an earlier version of this module claimed there is "no known
analog" of a split-half correction for `mae`. THAT WAS WRONG. Under additive
measurement noise, `mae_ceiling = split_half_mae / 2` is the direct analog (each half
carries ~2x the variance of the full pseudobulk, so the half-vs-half MAE is ~2x the
true achievable MAE) -- see REVIEW.md C-1 / fix plan item 4 for the measured numbers
that exposed this (split-half MAE exceeded the `no_change` floor MAE on every real
fold, which an analytic /2 correction alone would not have caught either, since
`direction_acc_de`/`pert_discrimination` have no equivalent clean correction).

ADDENDUM 3 Decision 1 resolves this a different way, at the run.py level rather than
here: baselines are scored against a held-out replicate half (`halves[r,1]`) and the
ceiling is scored as `halves[r,0]` vs `halves[r,1]`, so model and ceiling face IDENTICAL
target noise and the ceiling is a true upper bound for EVERY metric with NO per-metric
analytic correction needed at all -- this supersedes applying a `mae`-specific /2 fix
here. `compute_ceiling` itself is unchanged by that decision (it still scores
half-vs-half exactly as before); what changed is what baselines are scored against
(see `run.py`'s replicate-matched scoring path and the `target` column in
results.csv), plus the mandatory direction guard (ADDENDUM 3 Decision 3, in run.py)
that refuses to emit a `normalized` value whose sign contradicts `METRIC_DIRECTION`,
regardless of which scoring path produced floor/ceiling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from perturb_bench.metrics import aggregate, score_conditions

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from perturb_bench.data import PseudobulkData  # noqa: F401


def compute_ceiling(
    halves: np.ndarray,
    control: np.ndarray,
    de_mask: np.ndarray,
    genes: np.ndarray,
    representation: str,
) -> pd.DataFrame:
    """Estimate the noise ceiling by scoring half[r,0] against half[r,1].

    Parameters
    ----------
    halves : (n_rep, 2, n_cond, n_genes_full) array of split-half pseudobulk DELTAS
             (each half already minus the same matched control), per CONTRACT.md.
             May be a np.memmap — reps are iterated one at a time so only one
             (2, n_cond, n_genes_full) slice needs to be materialized per rep.
    control : (n_cond, n_genes_full) matched control expression (full pseudobulk's
              control, same control used to build both halves' deltas). Only used to
              build the "absolute" representation.
    de_mask : (n_cond, n_genes_full) bool DE mask for the FULL condition (from data.py's
              single-cell test), restricted to `genes` below before scoring.
    genes : positional index array selecting the fold's gene subset (same array a
            baseline's `fit`/`predict` would be scored on for a fair comparison).
    representation : "delta" or "absolute" — same meaning as in metrics.score_conditions.
                      The "absolute" ceiling is the same diagnostic-only construct;
                      do not report it as a performance ceiling.

    Returns
    -------
    pd.DataFrame indexed by metric name with columns mean/std/n/n_skipped — i.e. the
    same shape `aggregate()` produces for a baseline's per-condition scores, averaged
    over BOTH reps and conditions, so run.py can drop this in as a pseudo-baseline row
    in results.csv (CONTRACT.md: "same output shape as a baseline's scores").
    """
    if representation not in ("delta", "absolute"):
        raise ValueError(f"representation must be 'delta' or 'absolute', got {representation!r}")

    genes = np.asarray(genes)
    control_sub = np.asarray(control)[:, genes]
    de_mask_sub = np.asarray(de_mask)[:, genes]

    n_rep = halves.shape[0]
    per_rep_frames = []
    for r in range(n_rep):
        half0 = np.asarray(halves[r, 0])[:, genes]
        half1 = np.asarray(halves[r, 1])[:, genes]
        df_r = score_conditions(
            pred=half0,
            true=half1,
            control=control_sub,
            de_mask=de_mask_sub,
            representation=representation,
        )
        per_rep_frames.append(df_r)

    all_scores = pd.concat(per_rep_frames, axis=0, ignore_index=True)
    return aggregate(all_scores)
