"""Split regimes for the perturbation-response benchmark harness.

Defined at the CONDITION level: (cell_line, compound, dose). Dose is part of
condition identity and is never collapsed over.

See CONTRACT.md for the exact `Fold` shape and SPEC.md §4 for the regime
definitions and the leakage checklist this module must enforce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REGIMES: list[str] = [
    "random",
    "held_out_context",
    "held_out_drug",
    "held_out_both",
]

_N_DRUG_FOLDS = 5
_RANDOM_HOLDOUT_FRAC = 0.2


@dataclass
class Fold:
    regime: str
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _require_columns(obs: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise ValueError(f"obs is missing required columns: {missing}")


def _all_idx(obs: pd.DataFrame) -> np.ndarray:
    return np.arange(len(obs))


def make_folds(obs: pd.DataFrame, regime: str, seed: int) -> list[Fold]:
    """Build the list of folds for a given regime.

    `obs` is the PseudobulkData.obs frame (row order == positional index into
    the data arrays). `seed` controls all randomness so the split is
    reproducible.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; must be one of {REGIMES}")
    _require_columns(obs, ["cell_line", "compound", "dose"])

    if regime == "random":
        return _make_random_folds(obs, seed)
    if regime == "held_out_context":
        return _make_held_out_context_folds(obs, seed)
    if regime == "held_out_drug":
        return _make_held_out_drug_folds(obs, seed)
    if regime == "held_out_both":
        return _make_held_out_both_folds(obs, seed)
    raise AssertionError("unreachable")


def _make_random_folds(obs: pd.DataFrame, seed: int) -> list[Fold]:
    rng = np.random.default_rng(seed)
    n = len(obs)
    idx = _all_idx(obs)
    n_test = max(1, int(round(n * _RANDOM_HOLDOUT_FRAC)))
    test_idx = rng.choice(idx, size=n_test, replace=False)
    test_idx = np.sort(test_idx)
    train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
    fold = Fold(
        regime="random",
        name="random/holdout",
        train_idx=train_idx,
        test_idx=test_idx,
        meta={
            "held_out_cell_lines": [],
            "held_out_compounds": [],
            "frac": _RANDOM_HOLDOUT_FRAC,
        },
    )
    return [fold]


def _make_held_out_context_folds(obs: pd.DataFrame, seed: int) -> list[Fold]:
    idx = _all_idx(obs)
    cell_lines = sorted(obs["cell_line"].unique().tolist())
    folds = []
    for cl in cell_lines:
        is_held = (obs["cell_line"] == cl).to_numpy()
        test_idx = idx[is_held]
        train_idx = idx[~is_held]
        folds.append(
            Fold(
                regime="held_out_context",
                name=f"held_out_context/{cl}",
                train_idx=train_idx,
                test_idx=test_idx,
                meta={"held_out_cell_lines": [cl], "held_out_compounds": []},
            )
        )
    return folds


def _compound_groups(obs: pd.DataFrame) -> tuple[dict[str, list[str]], bool]:
    """Group compounds by target annotation.

    Returns (group_name -> list of compounds, target_annotation_available).
    Falls back to treating every compound as its own singleton group (so the
    caller can randomly bucket them) when target is absent or all "unknown".
    """
    compounds = sorted(obs["compound"].unique().tolist())
    if "target" not in obs.columns:
        return {c: [c] for c in compounds}, False

    # map compound -> target (first observed value)
    compound_to_target = (
        obs.groupby("compound")["target"].first().to_dict()
    )
    targets_present = {
        t for t in compound_to_target.values() if t is not None and str(t).lower() != "unknown" and str(t) != ""
    }
    if not targets_present:
        return {c: [c] for c in compounds}, False

    groups: dict[str, list[str]] = {}
    for c in compounds:
        t = compound_to_target.get(c, "unknown")
        if t is None or str(t).lower() == "unknown" or str(t) == "":
            # unknown-target compounds each form their own group so they
            # don't silently cluster with real annotated groups
            key = f"__unknown__{c}"
        else:
            key = str(t)
        groups.setdefault(key, []).append(c)
    return groups, True


def _bucket_groups_into_folds(
    group_names: list[str], n_folds: int, rng: np.random.Generator
) -> list[list[str]]:
    """Shuffle group names and deal them round-robin into n_folds buckets."""
    order = list(group_names)
    rng.shuffle(order)
    n_folds = max(1, min(n_folds, len(order))) if order else 1
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(order):
        buckets[i % n_folds].append(g)
    return [b for b in buckets if b]


def _make_held_out_drug_folds(obs: pd.DataFrame, seed: int) -> list[Fold]:
    rng = np.random.default_rng(seed)
    idx = _all_idx(obs)
    groups, target_available = _compound_groups(obs)
    group_names = sorted(groups.keys())
    buckets = _bucket_groups_into_folds(group_names, _N_DRUG_FOLDS, rng)

    folds = []
    for i, bucket in enumerate(buckets):
        held_out_compounds = sorted(
            {c for gname in bucket for c in groups[gname]}
        )
        is_held = obs["compound"].isin(held_out_compounds).to_numpy()
        test_idx = idx[is_held]
        train_idx = idx[~is_held]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append(
            Fold(
                regime="held_out_drug",
                name=f"held_out_drug/fold{i}",
                train_idx=train_idx,
                test_idx=test_idx,
                meta={
                    "held_out_cell_lines": [],
                    "held_out_compounds": held_out_compounds,
                    "target_annotation_available": target_available,
                },
            )
        )
    return folds


def _make_held_out_both_folds(obs: pd.DataFrame, seed: int) -> list[Fold]:
    rng = np.random.default_rng(seed)
    idx = _all_idx(obs)
    cell_lines = sorted(obs["cell_line"].unique().tolist())
    groups, target_available = _compound_groups(obs)
    group_names = sorted(groups.keys())
    # Reuse the same compound bucketing scheme as held_out_drug, then cross
    # each compound-bucket with each held-out cell line.
    buckets = _bucket_groups_into_folds(group_names, _N_DRUG_FOLDS, rng)

    folds = []
    for cl in cell_lines:
        for i, bucket in enumerate(buckets):
            held_out_compounds = sorted(
                {c for gname in bucket for c in groups[gname]}
            )
            line_held = (obs["cell_line"] == cl).to_numpy()
            compound_held = obs["compound"].isin(held_out_compounds).to_numpy()

            test_mask = line_held & compound_held  # intersection
            # training excludes every condition touching EITHER held-out axis
            train_mask = ~(line_held | compound_held)

            test_idx = idx[test_mask]
            train_idx = idx[train_mask]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            folds.append(
                Fold(
                    regime="held_out_both",
                    name=f"held_out_both/{cl}__fold{i}",
                    train_idx=train_idx,
                    test_idx=test_idx,
                    meta={
                        "held_out_cell_lines": [cl],
                        "held_out_compounds": held_out_compounds,
                        "target_annotation_available": target_available,
                    },
                )
            )
    return folds


def assert_no_leakage(obs: pd.DataFrame, fold: Fold) -> None:
    """Programmatic leakage checklist from SPEC §4.

    - train/test index sets are disjoint
    - together they only cover valid rows (no out-of-range / duplicate indices)
    - no held-out cell line appears in any training condition
    - no held-out compound appears in any training condition

    H-2 fix (REVIEW.md): the checks below used to derive "what is held out" ONLY
    from `fold.meta` (`held_out_cell_lines` / `held_out_compounds`), which the fold
    itself declares. That meant a fold with a genuine leak AND an empty/wrong meta
    (e.g. `regime="held_out_context"`, cell line A present in both train and test,
    `meta={}`) passed silently -- the check was just "did make_folds contradict
    itself", not an independent leakage check. Now the held-out sets are DERIVED
    from `fold.regime` + the actual train/test rows in `obs`, and cross-checked
    against `fold.meta` (a mismatch between declared and derived is itself an
    error, since it means meta lied about what this fold holds out).
    """
    n = len(obs)
    train_idx = np.asarray(fold.train_idx)
    test_idx = np.asarray(fold.test_idx)

    assert train_idx.size == len(set(train_idx.tolist())), (
        f"{fold.name}: train_idx contains duplicates"
    )
    assert test_idx.size == len(set(test_idx.tolist())), (
        f"{fold.name}: test_idx contains duplicates"
    )

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    overlap = train_set & test_set
    assert not overlap, (
        f"{fold.name}: train/test index sets are not disjoint; overlap={sorted(overlap)[:10]}"
    )

    if train_idx.size:
        assert train_idx.min() >= 0 and train_idx.max() < n, (
            f"{fold.name}: train_idx out of bounds for obs of length {n}"
        )
    if test_idx.size:
        assert test_idx.min() >= 0 and test_idx.max() < n, (
            f"{fold.name}: test_idx out of bounds for obs of length {n}"
        )

    train_lines = set(obs["cell_line"].iloc[train_idx].unique().tolist()) if train_idx.size else set()
    test_lines = set(obs["cell_line"].iloc[test_idx].unique().tolist()) if test_idx.size else set()
    train_compounds = set(obs["compound"].iloc[train_idx].unique().tolist()) if train_idx.size else set()
    test_compounds = set(obs["compound"].iloc[test_idx].unique().tolist()) if test_idx.size else set()

    declared_held_out_cell_lines = set(fold.meta.get("held_out_cell_lines", []))
    declared_held_out_compounds = set(fold.meta.get("held_out_compounds", []))

    # --- regime-derived independent checks (H-2: do NOT trust meta alone) ---
    if fold.regime in ("held_out_context", "held_out_both"):
        # The defining property of these regimes: the test set's cell lines must
        # be completely absent from train, independent of what meta claims.
        leaked_lines = test_lines & train_lines
        assert not leaked_lines, (
            f"{fold.name} (regime={fold.regime}): cell line(s) {leaked_lines} appear "
            f"in BOTH train and test -- this is the defining leakage check for "
            f"{fold.regime} and does not depend on fold.meta being correct."
        )
        # Cross-check: meta's declaration must match what the data actually shows.
        assert test_lines == declared_held_out_cell_lines, (
            f"{fold.name}: fold.meta['held_out_cell_lines']={sorted(declared_held_out_cell_lines)} "
            f"does not match the cell line(s) actually present in test_idx: "
            f"{sorted(test_lines)}. meta is lying about what this fold holds out."
        )

    if fold.regime in ("held_out_drug", "held_out_both"):
        leaked_compounds = test_compounds & train_compounds
        assert not leaked_compounds, (
            f"{fold.name} (regime={fold.regime}): compound(s) {leaked_compounds} "
            f"appear in BOTH train and test -- this is the defining leakage check "
            f"for {fold.regime} and does not depend on fold.meta being correct."
        )
        assert test_compounds <= declared_held_out_compounds, (
            f"{fold.name}: fold.meta['held_out_compounds'] does not cover the "
            f"compound(s) actually present in test_idx: "
            f"{sorted(test_compounds - declared_held_out_compounds)} missing from "
            f"the declared set. meta is lying about what this fold holds out."
        )

    # --- legacy meta-declared checks, kept as an additional (not sole) layer ---
    if declared_held_out_cell_lines and train_idx.size:
        leaked_lines = train_lines & declared_held_out_cell_lines
        assert not leaked_lines, (
            f"{fold.name}: held-out cell line(s) {leaked_lines} appear in training conditions"
        )

    if declared_held_out_compounds and train_idx.size:
        leaked_compounds = train_compounds & declared_held_out_compounds
        assert not leaked_compounds, (
            f"{fold.name}: held-out compound(s) {leaked_compounds} appear in training conditions"
        )

    # Test set sanity for the regimes that define explicit held-out axes.
    if declared_held_out_cell_lines and test_idx.size:
        assert test_lines <= declared_held_out_cell_lines, (
            f"{fold.name}: test set contains cell line(s) outside the declared held-out set: "
            f"{test_lines - declared_held_out_cell_lines}"
        )

    if declared_held_out_compounds and test_idx.size and fold.regime in ("held_out_drug", "held_out_both"):
        assert test_compounds <= declared_held_out_compounds, (
            f"{fold.name}: test set contains compound(s) outside the declared held-out set: "
            f"{test_compounds - declared_held_out_compounds}"
        )
