"""Single entry point for the perturbation-response benchmark harness.

    python -m perturb_bench.run --config configs/default.yaml [--quick] [--out-dir DIR] [--resume]

Produces outputs/results.csv and outputs/report.md (or under --out-dir). See
CONTRACT.md for the results.csv schema and ADDENDUM 1 (floor policy) /
ADDENDUM 2 (halves gene-union caching) / ADDENDUM 3 (ceiling commensurability,
replicate-matched scoring, the direction guard), and SPEC.md §8/§10 for what
report.md must contain.

Determinism: everything downstream of `load_pseudobulk` is a deterministic
function of (data, cfg.seed) -- `make_folds` is seeded, baselines are fit
only on deterministic inputs (Ridge's internal KFold CV is seeded with a
fixed random_state=0, not cfg.seed, but that is a constant, not a source of
run-to-run variance), the bootstrap CIs below are seeded off a stable hash of
(cfg.seed, split, fold, baseline, metric, representation, target) rather than
global RNG state, and the final DataFrame is sorted on stable keys before
being written. Two runs with the same config+seed must produce byte-identical
results.csv.

Leakage: `splits.assert_no_leakage` is called for EVERY fold, unconditionally,
before any gene selection or baseline touches that fold's data. It is allowed
to raise -- we do not catch it. That assertion firing is the harness doing its
job, not a bug to route around.

ADDENDUM 3 summary (read CONTRACT.md for the full text -- this is the part
that changed run.py's structure the most):
  - Decision 1: baselines are now ALSO scored against a held-out replicate
    half (`halves[r, 1]`), averaged over reps, with the ceiling scored as
    `halves[r, 0]` vs `halves[r, 1]` -- IDENTICAL target noise for model and
    ceiling. This is `target == "replicate_half"`, the PRIMARY scoring path.
  - Decision 2: the old behavior (score against the full pseudobulk delta) is
    retained as `target == "full_pseudobulk"`, a labeled SECONDARY path, and
    only gets a non-NaN `normalized` value if the guard below passes.
  - Decision 3 (non-negotiable, independent of 1/2): before emitting any
    `normalized` value, `_normalize_with_guard` checks
    sign(ceiling_mean - floor_mean) against METRIC_DIRECTION[metric]. On a
    contradiction it emits normalized=NaN, normalization_valid=False, and
    logs loudly to stderr naming the fold, metric, representation, target,
    and the two offending means. This stays on permanently.
  - Decision 4: `direction_acc_de`'s floor changes from `no_change` to
    `global_mean_delta` (amends ADDENDUM 1 -- see FLOOR_BASELINE below).

NOTE on `cond_id` / obs columns: per explicit instruction from the
orchestrator, this module and report.py must NEVER hardcode or regex-parse
the `cond_id` string format (it may grow a `time` component). Any condition
identity information used here comes from `data.obs` columns (cell_line,
compound, dose, ...), never by parsing `cond_id`.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_pseudobulk, select_hvgs
from .splits import REGIMES, assert_no_leakage, make_folds
from .baselines import BASELINES
from .metrics import (
    METRIC_DIRECTION,
    aggregate,
    bootstrap_ci_mean,
    normalize_score,
    score_conditions,
)
from .ceiling import compute_ceiling
from . import report as report_mod

REPRESENTATIONS = ("delta", "absolute")

# ADDENDUM 3 Decision 1/2: baselines (and the ceiling) are scored against two
# different targets. "replicate_half" is PRIMARY (matched target noise vs the
# ceiling, see module docstring); "full_pseudobulk" is SECONDARY/legacy and is
# retained only because most published numbers use that convention -- it does
# NOT get a normalized value unless the direction guard passes.
TARGETS = ("replicate_half", "full_pseudobulk")
PRIMARY_TARGET = "replicate_half"

# ADDENDUM 1 -- per-metric floor policy (binding orchestrator decision; see
# CONTRACT.md), amended by ADDENDUM 3 Decision 4 (direction_acc_de). no_change
# is NaN-on-delta for the correlation metrics (zero variance), so those two
# use global_mean_delta as the honest "no drug identity" floor instead.
# direction_acc_de's no_change floor of exactly 0.0 is a sign(0) artifact, not
# a genuine floor (H-5) -- global_mean_delta replaces it too. Never coerce any
# of this NaN to 0.
FLOOR_BASELINE: dict[str, str] = {
    "pearson_r": "global_mean_delta",
    "pearson_r_de": "global_mean_delta",
    "mae": "no_change",
    "direction_acc_de": "global_mean_delta",
    "pert_discrimination": "no_change",
}

# B4 ridge uses a one-hot drug encoding and is explicitly undefined for
# regimes where the test compound was never seen in training (SPEC §5). We do
# not silently change ridge's encoding -- we run the separately-registered,
# clearly-labeled drug-agnostic variant (ridge_no_drug) in those regimes
# instead, and never run plain "ridge" there at all.
_STANDARD_BASELINES = ["no_change", "global_mean_delta", "per_drug_mean_delta", "ridge", "nearest_context"]
_DRUG_AGNOSTIC_BASELINES = [
    "no_change", "global_mean_delta", "per_drug_mean_delta", "ridge_no_drug", "nearest_context",
]

_QUICK_MAX_FOLDS_PER_REGIME = 2
_QUICK_MAX_GENES = 200
_N_BOOT = 1000

RESULTS_COLUMNS = [
    "split", "fold", "baseline", "metric", "representation", "target",
    "value", "std", "n_conditions", "n_test", "n_skipped",
    "ci_low", "ci_high",
    "normalized", "normalized_std", "normalization_valid",
    "floor_source", "floor_value", "ceiling_value",
    "fallback_rate", "fallback_rate_dose_collapsed",
]


def _log(msg: str) -> None:
    print(f"[run] {msg}", file=sys.stderr, flush=True)


def _baselines_for_regime(regime: str) -> list[str]:
    if regime in ("held_out_drug", "held_out_both"):
        return _DRUG_AGNOSTIC_BASELINES
    return _STANDARD_BASELINES


def _select_genes(data, train_idx: np.ndarray, cfg: Config, quick: bool) -> np.ndarray:
    """Train-only gene selection for one fold. NEVER passed test_idx."""
    if cfg.gene_set == "all":
        genes = np.arange(data.X.shape[1])
    else:
        genes = select_hvgs(data, train_idx, cfg.n_hvg)
    if quick and len(genes) > _QUICK_MAX_GENES:
        # A further subset of an already train-only selection is still
        # train-only -- no test data is consulted here either.
        genes = genes[:_QUICK_MAX_GENES]
    return genes


def _halves_gene_positions(data, genes: np.ndarray) -> np.ndarray | None:
    """Translate FULL-axis gene positions into LOCAL positions within the
    `halves` gene-union axis (ADDENDUM 2), or return None if no translation is
    needed (halves already covers every gene -- true for small/synthetic
    data). Raises loudly (AssertionError) if any gene is outside the cached
    union rather than silently mis-indexing or dropping it.
    """
    halves_gene_idx = data.meta.get("halves_gene_idx")
    if halves_gene_idx is None or len(set(halves_gene_idx)) >= data.X.shape[1]:
        return None
    union = np.asarray(sorted(halves_gene_idx), dtype=int)
    pos = np.searchsorted(union, genes)
    pos_clipped = np.clip(pos, 0, len(union) - 1)
    valid = (pos < len(union)) & (union[pos_clipped] == genes)
    if not np.all(valid):
        missing = np.asarray(genes)[~valid]
        raise AssertionError(
            f"{missing.size} fold gene(s) not covered by the cached halves gene union "
            "(ADDENDUM 2). The halves cache is stale relative to this config/splits "
            "logic -- delete the cache dir and rebuild rather than silently skipping "
            "the ceiling for these genes."
        )
    return pos


def _map_genes_for_halves(
    data, control_rows: np.ndarray, de_mask_rows: np.ndarray, genes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given control/de_mask already row-sliced to a fold's test rows (full
    gene axis), return (control, de_mask, halves_col_idx) such that indexing
    `data.halves`'s last axis with `halves_col_idx` lines up column-for-column
    with `genes` (full axis, the same order a baseline's prediction uses),
    and `control`/`de_mask` are sliced down to whatever axis `halves_col_idx`
    addresses. Shared by `_ceiling_for_fold` and `_replicate_half1_for_fold`
    so the ADDENDUM 2 remap logic exists in exactly one place.
    """
    pos = _halves_gene_positions(data, genes)
    if pos is None:
        return control_rows, de_mask_rows, np.asarray(genes)
    halves_gene_idx = data.meta["halves_gene_idx"]
    union = np.asarray(sorted(halves_gene_idx), dtype=int)
    return control_rows[:, union], de_mask_rows[:, union], pos


def _ceiling_for_fold(data, test_idx: np.ndarray, genes: np.ndarray, representation: str) -> pd.DataFrame:
    """Score the split-half noise ceiling on the SAME test conditions and the
    SAME gene subset a baseline would be scored on for this fold.

    REVIEW FIX (see REVIEW.md §4, C-2): the ceiling was previously computed
    over ALL conditions in the dataset rather than the fold's test rows,
    which made the normalization denominator fold-mismatched (and made
    pert_discrimination rank against every condition instead of the held-out
    pool). Rows are sliced to `test_idx` for halves/control/de_mask before
    scoring.

    ADDENDUM 2: `data.halves` is cached only over the UNION of every fold's
    train-only HVG selection across all regimes (a much smaller axis than the
    full, detection-filtered gene axis that X/control/delta/de_mask live on).
    `genes` here is positional indices into the FULL gene axis; `_map_genes_for_halves`
    translates it into positions within the halves union axis and slices
    control/de_mask down to that same union axis so the three arrays stay
    index-consistent.
    """
    test_idx = np.asarray(test_idx)
    halves_test = np.asarray(data.halves[:, :, test_idx, :])
    control_test = data.control[test_idx]
    de_mask_test = data.de_mask[test_idx]
    control_m, de_mask_m, idx = _map_genes_for_halves(data, control_test, de_mask_test, genes)
    return compute_ceiling(halves_test, control_m, de_mask_m, idx, representation)


def _replicate_half1_for_fold(data, test_idx: np.ndarray, genes: np.ndarray) -> np.ndarray:
    """ADDENDUM 3 Decision 1 (PRIMARY scoring target): the second split-half
    delta (`halves[:, 1]`), restricted to this fold's test rows and remapped
    onto `genes` (full axis, the exact column order a baseline's prediction
    uses), for every ceiling rep. Returns (n_rep, n_test, n_genes).

    Baselines are scored against this instead of (or in addition to) the full
    pseudobulk delta so that model and ceiling face IDENTICAL target noise --
    the ceiling is `halves[:,0]` vs this same `halves[:,1]`. Costs nothing
    extra to fetch since halves are already cached over every fold's genes
    (ADDENDUM 2); the extra cost is re-scoring each baseline n_rep times
    instead of once, which is why `_pert_discrimination` being vectorized
    (fix #15) matters more now, not less.
    """
    test_idx = np.asarray(test_idx)
    half1_test = np.asarray(data.halves[:, 1, test_idx, :])  # (n_rep, n_test, n_genes_local)
    pos = _halves_gene_positions(data, genes)
    idx = np.asarray(genes) if pos is None else pos
    return half1_test[:, :, idx]


def _derived_seed(cfg_seed: int, *parts: str) -> int:
    """Deterministic, reproducible per-row seed for bootstrap resampling.
    Depends only on cfg.seed and a stable tuple of string keys (never on
    iteration order or global RNG state), so re-running with the same config
    and seed reproduces byte-identical ci_low/ci_high columns (determinism
    requirement, module docstring)."""
    h = zlib.crc32("|".join(parts).encode("utf-8"))
    return int((int(cfg_seed) * 2_147_483_647 + h) % (2**31 - 1))


def _normalize_with_guard(
    value: float,
    std: float,
    floor_mean: float,
    ceiling_mean: float,
    metric: str,
    fold_name: str,
    representation: str,
    target: str,
) -> tuple[float, float, bool]:
    """ADDENDUM 3 Decision 3 -- the permanent backstop.

    Checks sign(ceiling_mean - floor_mean) against METRIC_DIRECTION[metric]
    before emitting ANY normalized value. If they contradict, this is a
    degenerate or inverted ceiling/floor pair for this (fold, metric,
    representation, target) -- emit normalized=NaN, normalized_std=NaN,
    normalization_valid=False, and log loudly to stderr naming everything a
    reader would need to investigate. A normalized number whose scale is
    inverted must NEVER reach results.csv or report.md; this is the exact
    backstop that would have caught REVIEW.md C-1 (split-half mae ceiling
    worse than the no_change floor, silently inverting normalized mae to
    +1.07...+7.31 on every real fold).

    Returns (normalized, normalized_std, normalization_valid).
    normalized_std is the per-condition std propagated onto the normalized
    scale (std / |ceiling - floor|) -- a linear-propagation approximation
    (floor/ceiling themselves are treated as fixed, not as having their own
    sampling variance) used so that M-2 ("never report a mean without
    variance") applies to normalized cells too, not just raw ones.
    """
    if np.isnan(floor_mean) or np.isnan(ceiling_mean):
        return float("nan"), float("nan"), True

    denom = ceiling_mean - floor_mean
    if denom == 0:
        # Degenerate (no dynamic range), not a direction violation.
        return float("nan"), float("nan"), True

    expected_higher = METRIC_DIRECTION[metric] == "higher_better"
    actual_higher = denom > 0
    if actual_higher != expected_higher:
        _log(
            "DIRECTION GUARD FIRED (ADDENDUM 3 Decision 3) -- "
            f"fold={fold_name!r} metric={metric!r} representation={representation!r} "
            f"target={target!r}: ceiling_mean={ceiling_mean:.6g} floor_mean={floor_mean:.6g} "
            f"(sign(ceiling-floor)={'+' if actual_higher else '-'}, but "
            f"METRIC_DIRECTION[{metric!r}]={METRIC_DIRECTION[metric]!r} expects "
            f"{'+' if expected_higher else '-'}). Emitting normalized=NaN, "
            "normalization_valid=False. This number must NOT be trusted or quoted -- "
            "see CONTRACT.md ADDENDUM 3 Decision 3 / REVIEW.md C-1."
        )
        return float("nan"), float("nan"), False

    normalized = normalize_score(value, floor_mean, ceiling_mean)
    normalized_std = float(std / abs(denom)) if not np.isnan(std) else float("nan")
    return normalized, normalized_std, True


def _score_against(
    pred: np.ndarray,
    true_arr: np.ndarray,
    control_sub: np.ndarray,
    de_mask_sub: np.ndarray,
    representation: str,
) -> pd.DataFrame:
    """Thin wrapper so the per-rep concat pattern (replicate_half target and
    the ceiling) and the single-shot pattern (full_pseudobulk target) share
    one call path. Returns the raw per-condition score_conditions() output."""
    return score_conditions(pred, true_arr, control_sub, de_mask_sub, representation=representation)


def _scores_for_targets(
    pred: np.ndarray,
    true_delta: np.ndarray,
    half1: np.ndarray,
    control_sub: np.ndarray,
    de_mask_sub: np.ndarray,
    representation: str,
) -> dict[str, pd.DataFrame]:
    """Score one baseline's prediction against both targets (ADDENDUM 3):
    - "full_pseudobulk": pred vs the full pseudobulk delta (secondary, legacy).
    - "replicate_half": pred vs EACH rep's held-out half `halves[r,1]`,
      concatenated across reps before aggregation (mirrors how
      `compute_ceiling` pools over reps+conditions) -- this is the PRIMARY
      path and is what makes the ceiling a true bound (same target noise).

    Returns {target: per-condition (or per-rep-condition) score_conditions()
    DataFrame}, NOT yet aggregated -- callers aggregate separately so they
    can also bootstrap over the same raw rows.
    """
    out: dict[str, pd.DataFrame] = {}
    out["full_pseudobulk"] = _score_against(pred, true_delta, control_sub, de_mask_sub, representation)

    n_rep = half1.shape[0]
    per_rep = [
        _score_against(pred, half1[r], control_sub, de_mask_sub, representation)
        for r in range(n_rep)
    ]
    out["replicate_half"] = pd.concat(per_rep, axis=0, ignore_index=True)
    return out


def _ceiling_raw_for_fold(
    half0: np.ndarray,
    half1: np.ndarray,
    control_sub: np.ndarray,
    de_mask_sub: np.ndarray,
    representation: str,
    n_rep: int,
) -> pd.DataFrame:
    """Score `half0[r]` vs `half1[r]` for every rep and concatenate -- the
    ceiling's raw per-rep-condition rows, used for BOTH its aggregate
    (`aggregate(...)`) and its bootstrap CI. A separate, directly-patchable
    function (rather than inlining this loop in `run()`) so tests can plant
    a deliberately wrong ceiling end-to-end without needing the redundant
    second half-vs-half pass that calling `_ceiling_for_fold`/`compute_ceiling`
    here would cost (REVIEW.md §1.1: the ceiling dominates runtime)."""
    per_rep = [
        _score_against(half0[r], half1[r], control_sub, de_mask_sub, representation)
        for r in range(n_rep)
    ]
    return pd.concat(per_rep, axis=0, ignore_index=True)


def _git_state() -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.run(
                args, cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "head": head if head else "no commits / not a git repo",
        "dirty_files": dirty.splitlines() if dirty else [],
    }


def _versions() -> dict:
    import numpy, pandas, scipy, sklearn
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _write_run_meta(path: Path, cfg: Config, quick: bool, resume: bool, extra: dict | None = None) -> None:
    meta = {
        "config": cfg.to_dict(),
        "quick": quick,
        "resume": resume,
        "seed": cfg.seed,
        "versions": _versions(),
        "git": _git_state(),
    }
    if extra:
        meta.update(extra)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _existing_completed_folds(out_csv: Path) -> tuple[pd.DataFrame | None, set[tuple[str, str]]]:
    """For --resume: read an existing results.csv (if any) and return
    (existing_df, {(split, fold) pairs already present}). A (split, fold) pair
    appearing at all means that fold's full row-set was appended atomically
    (run() only appends once a fold's rows are fully computed in memory), so
    presence is a safe completeness signal."""
    if not out_csv.exists():
        return None, set()
    existing = pd.read_csv(out_csv)
    done = set(existing[["split", "fold"]].drop_duplicates().itertuples(index=False, name=None))
    return existing, done


def run(
    cfg: Config, quick: bool = False, out_dir: str | None = None, resume: bool = False
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    out_dir_path = Path(out_dir) if out_dir else Path(cfg.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir_path / "results.csv"
    meta_path = out_dir_path / "run_meta.json"

    # H-6: run_meta.json is written BEFORE the sweep starts (config, seed,
    # versions, git state), so a crash mid-sweep still leaves provenance on
    # disk for whatever partial results.csv exists.
    _write_run_meta(meta_path, cfg, quick, resume)

    existing_df, done_folds = (None, set())
    if resume:
        existing_df, done_folds = _existing_completed_folds(out_csv)
        if done_folds:
            _log(f"--resume: {len(done_folds)} fold(s) already present in {out_csv}, will skip them")

    _log(f"loading pseudobulk: dataset={cfg.dataset} cache_key={cfg.cache_key()} quick={quick}")
    data = load_pseudobulk(cfg)
    _log(
        f"loaded {data.X.shape[0]} conditions x {data.X.shape[1]} genes "
        f"in {time.time() - t0:.1f}s (batch_metadata_available="
        f"{data.meta.get('batch_metadata_available')})"
    )
    n_rep = int(data.halves.shape[0])

    all_rows: list[dict] = list(existing_df.to_dict("records")) if existing_df is not None else []
    n_folds_total = len(done_folds)
    n_leakage_checks = len(done_folds)

    # H-6: incremental write -- append each fold's rows to results.csv as soon
    # as they are computed (header once), so a crash at fold K does not lose
    # folds 1..K-1. We open in append mode and write the header only if the
    # file doesn't already exist (fresh run) or is empty.
    header_needed = not (out_csv.exists() and out_csv.stat().st_size > 0)

    for regime in REGIMES:
        folds = make_folds(data.obs, regime, cfg.seed)
        if quick and len(folds) > _QUICK_MAX_FOLDS_PER_REGIME:
            folds = folds[:_QUICK_MAX_FOLDS_PER_REGIME]
        baseline_names = _baselines_for_regime(regime)
        _log(f"regime={regime}: {len(folds)} fold(s), baselines={baseline_names}")

        for fold in folds:
            if (regime, fold.name) in done_folds:
                _log(f"  {fold.name}: skipped (--resume, already in {out_csv})")
                continue

            t_fold = time.time()

            # Non-negotiable: every fold is leakage-checked before anything
            # touches its data. Let this raise if it fires.
            assert_no_leakage(data.obs, fold)
            n_leakage_checks += 1

            genes = _select_genes(data, fold.train_idx, cfg, quick)

            true_delta = data.delta[np.ix_(fold.test_idx, genes)]
            control_sub = data.control[np.ix_(fold.test_idx, genes)]
            de_mask_sub = data.de_mask[np.ix_(fold.test_idx, genes)]
            half1 = _replicate_half1_for_fold(data, fold.test_idx, genes)  # (n_rep, n_test, n_genes)
            n_test = len(fold.test_idx)

            # fold_aggs[(bname, rep, target)] -> aggregate() DataFrame
            fold_aggs: dict[tuple[str, str, str], pd.DataFrame] = {}
            fold_raw: dict[tuple[str, str, str], pd.DataFrame] = {}
            fallback_rates: dict[str, float] = {}
            fallback_rates_dose_collapsed: dict[str, float] = {}

            for bname in baseline_names:
                baseline = BASELINES[bname]()
                baseline.fit(data, fold.train_idx, genes)
                pred = baseline.predict(data, fold.test_idx)
                fmask = baseline.fallback_mask
                fallback_rates[bname] = float(np.mean(fmask)) if len(fmask) else float("nan")

                # M-3: tier-1 dose-collapse fallback (same compound, any dose)
                # for PerDrugMeanDelta. SPEC §11: dose is part of condition
                # identity, so this silent collapse must be counted, not just
                # the tier-2 (all the way to global mean) fallback_rate above.
                # NearestContext (B5) has the same kind of collapse internally
                # (dose-matched cd_map vs compound-only c_map) but does not
                # expose a per-tier counter -- see final report for that as a
                # blocker (baselines.py is not this module's file to edit).
                tiers = getattr(baseline, "fallback_tier", None)
                if tiers is not None and len(tiers):
                    fallback_rates_dose_collapsed[bname] = float(np.mean(np.asarray(tiers) == 1))
                else:
                    fallback_rates_dose_collapsed[bname] = float("nan")

                for rep in REPRESENTATIONS:
                    raw_by_target = _scores_for_targets(pred, true_delta, half1, control_sub, de_mask_sub, rep)
                    for target, raw_df in raw_by_target.items():
                        fold_raw[(bname, rep, target)] = raw_df
                        fold_aggs[(bname, rep, target)] = aggregate(raw_df)

            # Ceiling: score half0 vs half1 directly (rather than calling
            # `_ceiling_for_fold`/`compute_ceiling`, which would do this exact
            # per-rep loop internally a SECOND time) so the raw per-rep rows
            # are available once for both the aggregate and the bootstrap CI
            # -- REVIEW.md flagged the ceiling as the dominant runtime cost
            # (fix plan item 15 / §1.1), so this path must not duplicate it.
            # Routed through `_ceiling_raw_for_fold` (a thin, separately
            # patchable seam) purely so tests can plant a deliberately WRONG
            # ceiling end-to-end (see test_run.py's L-2-replacement test)
            # without needing to fight this function's internal call graph.
            half0 = _replicate_half0_for_fold(data, fold.test_idx, genes)
            ceiling_aggs: dict[str, pd.DataFrame] = {}
            ceiling_raw: dict[str, pd.DataFrame] = {}
            for rep in REPRESENTATIONS:
                raw = _ceiling_raw_for_fold(half0, half1, control_sub, de_mask_sub, rep, n_rep)
                ceiling_raw[rep] = raw
                ceiling_aggs[rep] = aggregate(raw)

            fold_rows: list[dict] = []
            for rep in REPRESENTATIONS:
                for target in TARGETS:
                    for metric in METRIC_DIRECTION:
                        floor_name = FLOOR_BASELINE[metric]
                        floor_mean = fold_aggs[(floor_name, rep, target)].loc[metric, "mean"]
                        ceiling_mean = ceiling_aggs[rep].loc[metric, "mean"]

                        for bname in baseline_names:
                            agg = fold_aggs[(bname, rep, target)]
                            value = agg.loc[metric, "mean"]
                            std = agg.loc[metric, "std"]
                            n_cond = agg.loc[metric, "n"]
                            n_skipped = agg.loc[metric, "n_skipped"]
                            normalized, normalized_std, valid = _normalize_with_guard(
                                value, std, floor_mean, ceiling_mean, metric, fold.name, rep, target,
                            )
                            seed = _derived_seed(cfg.seed, regime, fold.name, bname, metric, rep, target)
                            ci_low, ci_high = bootstrap_ci_mean(
                                fold_raw[(bname, rep, target)][metric].to_numpy(dtype=float),
                                seed=seed, n_boot=_N_BOOT,
                            )
                            fold_rows.append(dict(
                                split=regime, fold=fold.name, baseline=bname, metric=metric,
                                representation=rep, target=target, value=value, std=std,
                                n_conditions=n_cond, n_test=n_test, n_skipped=n_skipped,
                                ci_low=ci_low, ci_high=ci_high,
                                normalized=normalized, normalized_std=normalized_std,
                                normalization_valid=valid,
                                floor_source=floor_name, floor_value=floor_mean, ceiling_value=ceiling_mean,
                                fallback_rate=fallback_rates[bname],
                                fallback_rate_dose_collapsed=fallback_rates_dose_collapsed[bname],
                            ))

                        c_value = ceiling_aggs[rep].loc[metric, "mean"]
                        c_std = ceiling_aggs[rep].loc[metric, "std"]
                        c_n = ceiling_aggs[rep].loc[metric, "n"]
                        c_n_skipped = ceiling_aggs[rep].loc[metric, "n_skipped"]
                        c_normalized, c_normalized_std, c_valid = _normalize_with_guard(
                            c_value, c_std, floor_mean, ceiling_mean, metric, fold.name, rep, target,
                        )
                        seed = _derived_seed(cfg.seed, regime, fold.name, "ceiling", metric, rep, target)
                        c_ci_low, c_ci_high = bootstrap_ci_mean(
                            ceiling_raw[rep][metric].to_numpy(dtype=float), seed=seed, n_boot=_N_BOOT,
                        )
                        fold_rows.append(dict(
                            split=regime, fold=fold.name, baseline="ceiling", metric=metric,
                            representation=rep, target=target, value=c_value, std=c_std,
                            n_conditions=c_n, n_test=n_test, n_skipped=c_n_skipped,
                            ci_low=c_ci_low, ci_high=c_ci_high,
                            normalized=c_normalized, normalized_std=c_normalized_std,
                            normalization_valid=c_valid,
                            floor_source=floor_name, floor_value=floor_mean, ceiling_value=ceiling_mean,
                            fallback_rate=float("nan"), fallback_rate_dose_collapsed=float("nan"),
                        ))

            # Append this fold's rows to disk NOW (H-6) -- atomic from the
            # --resume perspective: either all of a fold's rows are on disk,
            # or none are.
            fold_df = pd.DataFrame(fold_rows, columns=RESULTS_COLUMNS)
            fold_df.to_csv(out_csv, mode="a", header=header_needed, index=False)
            header_needed = False

            all_rows.extend(fold_rows)
            n_folds_total += 1
            _log(
                f"  {fold.name}: {len(baseline_names)} baselines scored in "
                f"{time.time() - t_fold:.1f}s (n_train={len(fold.train_idx)}, "
                f"n_test={n_test}, n_genes={len(genes)}, n_rep={n_rep})"
            )

    results = pd.DataFrame(all_rows, columns=RESULTS_COLUMNS)
    results = results.sort_values(
        ["split", "fold", "baseline", "metric", "representation", "target"]
    ).reset_index(drop=True)

    # Final rewrite: the incremental appends above are in fold-completion
    # order (crash-safety); once the sweep finishes cleanly we rewrite the
    # whole file once, sorted, so the on-disk artifact is the same
    # deterministic, stably-ordered file the old single-shot writer produced
    # (determinism test requires byte-identical output across two full runs
    # with the same seed).
    results.to_csv(out_csv, index=False)
    _log(
        f"wrote {out_csv} ({len(results)} rows); {n_leakage_checks} fold(s) "
        f"leakage-checked; total elapsed {time.time() - t0:.1f}s"
    )

    run_meta = {
        "config": cfg.to_dict(),
        "quick": quick,
        "resume": resume,
        "seed": cfg.seed,
        "versions": _versions(),
        "git": _git_state(),
        "dataset_meta": data.meta,
        "n_conditions": int(data.X.shape[0]),
        "n_genes_total": int(data.X.shape[1]),
        "n_ceiling_reps": n_rep,
        "n_folds": n_folds_total,
        "n_leakage_checks": n_leakage_checks,
        "floor_policy": FLOOR_BASELINE,
        "targets": list(TARGETS),
        "primary_target": PRIMARY_TARGET,
    }
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    report_path = out_dir_path / "report.md"
    report_mod.write_report(results, run_meta, report_path)
    _log(f"wrote {report_path}")

    return results, run_meta


def _replicate_half0_for_fold(data, test_idx: np.ndarray, genes: np.ndarray) -> np.ndarray:
    """Same as `_replicate_half1_for_fold` but for `halves[:, 0]` -- the other
    side of the split-half ceiling pairing (`halves[:,0]` vs `halves[:,1]`).
    `run()` scores this directly (half0 vs half1, per rep) to get the
    ceiling's raw per-rep rows once, for both its aggregate and its bootstrap
    CI, rather than calling `_ceiling_for_fold`/`compute_ceiling` (which would
    do the identical per-rep loop internally a second time -- see the comment
    at the ceiling-scoring call site in `run()`)."""
    test_idx = np.asarray(test_idx)
    half0_test = np.asarray(data.halves[:, 0, test_idx, :])
    pos = _halves_gene_positions(data, genes)
    idx = np.asarray(genes) if pos is None else pos
    return half0_test[:, :, idx]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the perturbation-response benchmark harness.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--quick", action="store_true", help="Subset folds/genes for fast iteration.")
    parser.add_argument("--out-dir", default=None, help="Override cfg.out_dir.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folds already present in an existing results.csv at out-dir (H-6).",
    )
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    run(cfg, quick=args.quick, out_dir=args.out_dir, resume=args.resume)


if __name__ == "__main__":
    main()
