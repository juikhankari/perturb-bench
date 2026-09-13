"""Data loading, QC, pseudobulk, delta, DE masking, split-halves, and caching.

See SPEC.md §3 and CONTRACT.md for the authoritative interface. This module owns
EXACTLY the `PseudobulkData` container and the loader registry. HVG selection is
deliberately NOT performed here -- the cache always stores ALL (post-QC, post
detection-filter) genes; HVG selection happens per-fold, train-only, downstream
(see `select_hvgs`).

IMPORTANT disk/memory correction (see DATA_RECON.md and the commit history of this
file): sciplex3's raw h5ad has 799,317 cells x 110,983 genes/features. Caching
`halves` at shape (n_rep, 2, n_cond, n_genes) over ALL 110,983 genes would be
terabytes and does not fit this machine's ~24GB free disk / ~26GB RAM. Two fixes
are applied, both documented in meta and DATA_RECON.md:
  1. A dataset-global gene DETECTION filter (genes expressed in a tiny fraction of
     cells are dropped) shrinks "all genes" from 110,983 to ~12.9k before anything
     else happens. This uses only presence/absence across the WHOLE dataset, not any
     condition label or effect size, so it is not a leakage concern -- it is the same
     class of operation as the mitochondrial-% QC filter.
  2. `halves` (the expensive one: it's per-rep, per-half, per-condition) is cached
     ONLY over the UNION of gene indices that `select_hvgs` would pick, train-only,
     across every fold of every split regime. This is NOT leakage: each fold still
     only ever scores on genes selected from ITS OWN training rows. The cache merely
     happens to also hold other folds' selections; no fold's selection is informed by
     its own held-out data, which is the actual invariant. `meta["halves_gene_idx"]`
     records which positional gene indices (into the full, detection-filtered gene
     axis) the halves array covers, and `load_pseudobulk` asserts every fold's
     HVG selection is a subset of it.
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import Config

RNG_SEED_SALT_HALVES = 1_000_003
GENE_DETECTION_MIN_FRAC = 0.01  # a gene must be detected (count>0) in >=1% of cells


# ---------------------------------------------------------------------------
# Core container
# ---------------------------------------------------------------------------

@dataclass
class PseudobulkData:
    X: np.ndarray        # (n_cond, n_genes) float32
    control: np.ndarray  # (n_cond, n_genes) float32
    delta: np.ndarray    # (n_cond, n_genes) float32, == X - control
    de_mask: np.ndarray  # (n_cond, n_genes) bool
    halves: np.ndarray   # (n_rep, 2, n_cond, n_halves_genes) float32, may be memmap.
                         # See module docstring: n_halves_genes is a fold-HVG-union
                         # subset of n_genes, not n_genes itself. meta["halves_gene_idx"]
                         # gives the mapping.
    obs: pd.DataFrame    # index = cond_id
    var: np.ndarray      # (n_genes,) gene symbols, str
    meta: dict


class DatasetLoader(Protocol):
    name: str

    def load(self, cfg: Config) -> PseudobulkData: ...


_LOADER_REGISTRY: dict[str, "DatasetLoader"] = {}


def register_loader(loader: "DatasetLoader") -> None:
    _LOADER_REGISTRY[loader.name] = loader


def get_loader(name: str) -> "DatasetLoader":
    if name == "tahoe":
        raise NotImplementedError(
            "The 'tahoe' dataset loader is not implemented yet. This loader registry "
            "exists so a second dataset can be plugged in without touching splits.py, "
            "baselines.py, or metrics.py -- implement a DatasetLoader for tahoe and "
            "call register_loader(TahoeLoader()) to add it."
        )
    if name not in _LOADER_REGISTRY:
        raise KeyError(
            f"No loader registered for dataset {name!r}. Registered: "
            f"{sorted(_LOADER_REGISTRY)}"
        )
    return _LOADER_REGISTRY[name]


# ---------------------------------------------------------------------------
# Matrix adapters: give build_pseudobulk_core a uniform way to (a) get the mean
# expression over a set of row indices and (b) get a dense (len(idx), n_genes)
# array for a set of row indices -- without ever materializing the full cell x gene
# matrix. The dense, in-memory adapter is what tests use (tiny synthetic data); the
# backed adapter is what the real sciplex3 loader uses (streams from the h5ad file,
# normalizing+log1p-ing on the fly, one condition's cells at a time).
# ---------------------------------------------------------------------------

class _DenseMatrixAdapter:
    """Wraps an already QC'd + normalized + log1p'd dense (n_cells, n_genes) array
    held fully in memory. Used for small/synthetic data (tests)."""

    def __init__(self, X: np.ndarray):
        self.X = X
        self.n_genes = X.shape[1]

    def mean(self, idx: np.ndarray) -> np.ndarray:
        return self.X[idx].mean(axis=0)

    def dense(self, idx: np.ndarray, gene_idx: Optional[np.ndarray] = None) -> np.ndarray:
        sub = self.X[idx]
        return sub if gene_idx is None else sub[:, gene_idx]


class _BackedMatrixAdapter:
    """Streams rows from a backed AnnData's sparse X, restricted to a pre-selected
    set of "kept" gene columns, normalizing (by the row's FULL-transcriptome total
    count, computed once up front) and log1p-ing on the fly. Never holds more than
    one condition's / control group's cells densely at once."""

    def __init__(
        self,
        adata_backed,
        gene_keep_idx: np.ndarray,   # original column indices kept, len == n_genes
        total_counts: np.ndarray,    # (n_cells_total,) full-transcriptome per-cell total
        median_total: float,
    ):
        self.adata = adata_backed
        self.gene_keep_idx = gene_keep_idx
        self.total_counts = total_counts
        self.median_total = median_total
        self.n_genes = len(gene_keep_idx)

    def _fetch_raw(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        order = np.argsort(idx)
        idx_sorted = idx[order]
        chunk = self.adata.X[idx_sorted][:, self.gene_keep_idx]
        dense = chunk.toarray().astype(np.float64)
        inv = np.argsort(order)
        dense = dense[inv]
        totals = self.total_counts[idx][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = np.where(totals > 0, dense / totals * self.median_total, 0.0)
        normed = np.log1p(normed)
        return normed.astype(np.float32)

    def mean(self, idx: np.ndarray) -> np.ndarray:
        return self._fetch_raw(idx).mean(axis=0)

    def dense(self, idx: np.ndarray, gene_idx: Optional[np.ndarray] = None) -> np.ndarray:
        d = self._fetch_raw(idx)
        return d if gene_idx is None else d[:, gene_idx]


# ---------------------------------------------------------------------------
# Generic, testable pipeline pieces
# ---------------------------------------------------------------------------

def detect_mito_genes(gene_names: np.ndarray) -> Optional[np.ndarray]:
    """Return a bool mask of mitochondrial genes by symbol prefix, or None if none
    found (caller should then skip the mito QC filter rather than crash)."""
    gene_names = np.asarray(gene_names, dtype=str)
    mask = np.array(
        [g.upper().startswith("MT-") or g.upper().startswith("MT.") for g in gene_names]
    )
    if mask.sum() == 0:
        return None
    return mask


def qc_filter(
    X: sp.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    min_genes: int,
    max_pct_mito: float,
) -> tuple[np.ndarray, dict]:
    """Small-data QC filter (used by tests / tiny in-memory AnnData). Returns
    (bool keep-mask over cells, meta dict describing what was applied)."""
    meta: dict = {}
    if sp.issparse(X):
        n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        n_genes_per_cell = (X > 0).sum(axis=1)
    keep = n_genes_per_cell >= min_genes
    meta["n_cells_before_qc"] = int(X.shape[0])
    meta["n_cells_dropped_min_genes"] = int((~keep).sum())

    mito_mask = detect_mito_genes(gene_names)
    if mito_mask is None:
        meta["mito_filter_applied"] = False
        meta["mito_filter_note"] = (
            "No mitochondrial genes identifiable from gene symbols (no 'MT-' prefix "
            "found); skipping the mito-percent QC filter rather than crashing."
        )
    else:
        meta["mito_filter_applied"] = True
        total = np.asarray(X.sum(axis=1)).ravel()
        if sp.issparse(X):
            mito_total = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
        else:
            mito_total = X[:, mito_mask].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_mito = np.where(total > 0, 100.0 * mito_total / total, 0.0)
        mito_keep = pct_mito <= max_pct_mito
        meta["n_cells_dropped_mito"] = int((~mito_keep & keep).sum())
        keep = keep & mito_keep

    meta["n_cells_after_qc"] = int(keep.sum())
    return keep, meta


def ensure_lognorm(X: sp.spmatrix | np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect whether X looks like raw counts (near-integer, unnormalized) or is
    already log-normalized; normalize_total + log1p only in the raw-counts case.
    Returns (dense float32 array, meta dict documenting what was detected/done). Used
    for small/synthetic data only -- the real sciplex3 loader normalizes on the fly
    per-chunk via `_BackedMatrixAdapter` instead of densifying the whole matrix."""
    meta: dict = {}
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    Xd = Xd.astype(np.float64)

    sample = Xd[: min(500, Xd.shape[0])]
    nonzero = sample[sample > 0]
    looks_integer = nonzero.size > 0 and np.allclose(nonzero, np.round(nonzero), atol=1e-6)
    max_val = float(Xd.max()) if Xd.size else 0.0

    if looks_integer and max_val > 30:
        meta["detected_input"] = "raw_counts"
        totals = Xd.sum(axis=1, keepdims=True)
        median_total = np.median(totals[totals > 0]) if np.any(totals > 0) else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            Xn = np.where(totals > 0, Xd / totals * median_total, 0.0)
        Xn = np.log1p(Xn)
        meta["normalization_applied"] = "normalize_total+log1p"
    else:
        meta["detected_input"] = "already_normalized_or_log"
        meta["normalization_applied"] = "none (detected pre-normalized input)"
        Xn = Xd

    return Xn.astype(np.float32), meta


def _mannwhitney_de_for_condition(
    treated: np.ndarray, control: np.ndarray, alpha: float
) -> np.ndarray:
    """Vectorized (over genes) Mann-Whitney U test of treated vs control cells for a
    single condition, BH-FDR corrected. Returns bool (n_genes,) DE mask.

    Kept for tests / the single-plate case. Real sciplex3 conditions go through
    `_stratified_de_pvalues` below (H-3 fix) so treated cells are never tested
    against a different plate's control cells."""
    n_genes = treated.shape[1]
    if treated.shape[0] < 2 or control.shape[0] < 2:
        return np.zeros(n_genes, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.mannwhitneyu(treated, control, axis=0, alternative="two-sided")
            pvals = np.asarray(res.pvalue, dtype=np.float64)
        except Exception:
            pvals = np.full(n_genes, np.nan)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return reject


def _plate_z_from_mwu(treated: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One plate's contribution to a stratified DE test: signed z-scores (direction
    from the sign of the per-gene mean difference) and a per-gene weight (sqrt of the
    treated-cell count on this plate). Returns (z, weight), both (n_genes,); z is NaN
    where undefined (too few cells) and must be masked out by the caller."""
    n_genes = treated.shape[1]
    if treated.shape[0] < 2 or control.shape[0] < 2:
        return np.full(n_genes, np.nan), np.zeros(n_genes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.mannwhitneyu(treated, control, axis=0, alternative="two-sided")
            pvals = np.asarray(res.pvalue, dtype=np.float64)
        except Exception:
            pvals = np.full(n_genes, np.nan)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    pvals = np.clip(pvals, 1e-300, 1.0)
    direction = np.sign(treated.mean(axis=0) - control.mean(axis=0))
    direction = np.where(direction == 0, 1.0, direction)
    z = stats.norm.isf(pvals / 2.0) * direction
    weight = np.full(n_genes, np.sqrt(treated.shape[0]))
    return z, weight


def _stratified_de_mask(
    plate_pairs: list[tuple[np.ndarray, np.ndarray]], alpha: float, n_genes: int
) -> np.ndarray:
    """Stratified (per-plate) Mann-Whitney DE test, combined across plates via a
    Stouffer z-score combination weighted by sqrt(n_treated_on_plate). H-3 fix:
    treated cells are tested ONLY against their OWN plate's matched control cells --
    never pooled with, or compared against, another plate's controls. With exactly
    one plate this reduces exactly to the plain two-sided Mann-Whitney p-value (the
    p->z->p round trip is exact), so single-plate conditions are unaffected.
    `plate_pairs` is a list of (treated_dense, control_dense) arrays, one per usable
    plate. BH-FDR is applied across genes on the combined p-value."""
    if not plate_pairs:
        return np.zeros(n_genes, dtype=bool)
    zs = np.zeros((len(plate_pairs), n_genes))
    ws = np.zeros((len(plate_pairs), n_genes))
    for i, (treated, control) in enumerate(plate_pairs):
        z, w = _plate_z_from_mwu(treated, control)
        valid = ~np.isnan(z)
        zs[i] = np.where(valid, z, 0.0)
        ws[i] = np.where(valid, w, 0.0)
    denom = np.sqrt(np.sum(ws ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        combined_z = np.where(denom > 0, np.sum(ws * zs, axis=0) / denom, 0.0)
    has_any = denom > 0
    pvals = np.where(has_any, 2.0 * stats.norm.sf(np.abs(combined_z)), 1.0)
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return reject & has_any


@dataclass
class _HalvesContext:
    """Everything needed to compute split-half noise-ceiling deltas for an arbitrary
    gene subset, without re-deriving condition groupings or control matches.

    H-3 fix: cells are grouped PER PLATE, not pooled across a condition's plates --
    `cond_plate_cell_idx[ck]` maps plate-key -> that plate's (usable) treated cell
    indices for condition `ck`, and `cond_plate_control_key[ck]` maps the same
    plate-key -> the control-group key (into `control_profile`) for THAT plate. Only
    plates with a matched vehicle-cell group are present at all (plates without one
    are dropped upstream in `build_pseudobulk_core`, counted in meta, never silently
    pooled with another plate's control)."""
    matrix: object  # _DenseMatrixAdapter | _BackedMatrixAdapter
    cond_plate_cell_idx: dict[str, dict[str, np.ndarray]]   # cond_id -> plate_key -> idx
    cond_plate_control_key: dict[str, dict[str, str]]       # cond_id -> plate_key -> control key
    control_profile: dict[str, np.ndarray]     # control key -> mean profile (n_genes,)
    obs_index: list[str]                       # cond_id order matching PseudobulkData rows
    n_genes: int
    seed: int
    n_rep: int


def compute_halves(ctx: _HalvesContext, gene_idx: np.ndarray) -> np.ndarray:
    """Compute split-half pseudobulk DELTAS for the given gene subset (positional
    indices into the full, detection-filtered gene axis). Shape
    (n_rep, 2, n_cond, len(gene_idx)) float32.

    H-3 fix: each condition's cells are split into two halves WITHIN each plate
    separately (never pooling cells across plates before the split), and each half is
    then a cell-count-weighted average of that half's per-plate deltas against that
    SAME plate's matched control -- i.e. every cell in every half is scored against
    its own plate's control, exactly mirroring how the main `delta` array is built."""
    n_cond = len(ctx.obs_index)
    n_g = len(gene_idx)
    halves = np.zeros((ctx.n_rep, 2, n_cond, n_g), dtype=np.float32)
    rng = np.random.default_rng(ctx.seed + RNG_SEED_SALT_HALVES)
    for i, ck in enumerate(ctx.obs_index):
        plate_idx_map = ctx.cond_plate_cell_idx[ck]
        plate_ctrl_map = ctx.cond_plate_control_key[ck]
        # Pre-fetch per-plate cell-dense arrays and matched control profiles once;
        # reused across all n_rep permutations of this condition.
        plate_keys = list(plate_idx_map.keys())
        plate_dense = {pk: ctx.matrix.dense(plate_idx_map[pk], gene_idx) for pk in plate_keys}
        plate_ctrl = {pk: ctx.control_profile[plate_ctrl_map[pk]][gene_idx] for pk in plate_keys}
        plate_n = {pk: plate_idx_map[pk].size for pk in plate_keys}
        for r in range(ctx.n_rep):
            sum1 = np.zeros(n_g, dtype=np.float64)
            sum2 = np.zeros(n_g, dtype=np.float64)
            w1 = 0
            w2 = 0
            for pk in plate_keys:
                n = plate_n[pk]
                perm = rng.permutation(n)
                split = n // 2
                h1, h2 = perm[:split], perm[split:]
                if h1.size == 0 or h2.size == 0:
                    continue
                dense = plate_dense[pk]
                ctrl = plate_ctrl[pk]
                sum1 += dense[h1].mean(axis=0) * h1.size
                sum2 += dense[h2].mean(axis=0) * h2.size
                # subtract the SAME plate's control, weighted the same way the mean
                # is weighted, so weighted-avg(half) - weighted-avg(control) == the
                # weighted-avg of per-plate (half - control), same identity used for
                # the main `delta` array.
                sum1 -= ctrl * h1.size
                sum2 -= ctrl * h2.size
                w1 += h1.size
                w2 += h2.size
            if w1 > 0:
                halves[r, 0, i] = (sum1 / w1).astype(np.float32)
            if w2 > 0:
                halves[r, 1, i] = (sum2 / w2).astype(np.float32)
        del plate_dense
    return halves


def build_pseudobulk_core(
    matrix,                            # _DenseMatrixAdapter | _BackedMatrixAdapter
    cell_line: np.ndarray,             # (n_cells,) str, over ALL cells matrix indexes
    compound: np.ndarray,              # (n_cells,) str
    dose: np.ndarray,                  # (n_cells,) float
    is_control: np.ndarray,            # (n_cells,) bool
    batch: Optional[np.ndarray],       # (n_cells,) str, or None if no batch metadata
    gene_names: np.ndarray,
    cfg: Config,
    meta_extra: Optional[dict] = None,
    want_halves: bool = True,
    time: Optional[np.ndarray] = None,  # (n_cells,) float; H-4 fix, see module docstring
) -> tuple[PseudobulkData, _HalvesContext]:
    """The core, dataset-agnostic pipeline: pseudobulk, matched controls, delta,
    single-cell DE mask, and (optionally, immediately) split-half noise-ceiling
    halves over ALL genes. For the real sciplex3 loader we call this with
    want_halves=False to get X/control/delta/de_mask cheaply, then compute halves
    separately over a much smaller fold-HVG-union gene subset via `compute_halves`
    (see module docstring) using the returned `_HalvesContext`.

    H-4 fix (REVIEW.md): `time` is now part of condition identity. `cond_id` format
    changed from `cell_line|compound|dose` to `cell_line|compound|dose|time`. If
    `time` is not supplied, every cell is treated as time=0.0 (a single constant),
    which makes this a no-op for callers/tests that don't have a time axis --
    cond_id still gets a trailing `|0.0` though, so any caller that parses cond_id
    by position (none do today; verified by grep) would need updating. See
    meta["time_identity"] for counts.

    H-3 fix (REVIEW.md): controls are matched PER PLATE, not to a single modal
    plate. For each condition, cells are grouped by plate; a plate is "usable" only
    if a matched (cell_line, plate) vehicle-cell control group exists for it --
    otherwise that plate's cells are DROPPED from the condition (not the whole
    condition) and counted in meta["plate_matching"]. The condition's X/control/delta
    are then a cell-count-weighted average over its usable plates' per-plate
    pseudobulk/control/delta, which guarantees `delta == X - control` exactly (see
    the comment above the averaging code) while every cell that contributes is
    compared only to ITS OWN plate's vehicle cells -- 100% plate-matched by
    construction, not merely for the modal plate's share."""
    meta: dict = dict(meta_extra or {})

    n_cells = len(cell_line)
    n_genes = matrix.n_genes
    has_batch = batch is not None
    meta["batch_metadata_available"] = bool(has_batch)
    if not has_batch:
        meta["batch_note"] = (
            "No batch/plate/well metadata column found for this dataset. Falling back "
            "to per-cell_line controls (pooling all vehicle cells for a cell line into "
            "a single control profile), which means deltas may be contaminated by "
            "batch effects if treated and vehicle cells were not processed together. "
            "This is a real limitation of the source data annotation, not a shortcut "
            "taken silently -- flagged here and surfaced in the report."
        )
        batch = np.array(["__no_batch__"] * n_cells, dtype=object)

    cell_line = np.asarray(cell_line, dtype=object)
    compound = np.asarray(compound, dtype=object)
    dose = np.asarray(dose, dtype=float)
    batch = np.asarray(batch, dtype=object)
    has_time = time is not None
    time = np.asarray(time, dtype=float) if has_time else np.zeros(n_cells, dtype=float)
    meta["time_identity"] = {
        "added_to_cond_id": True,
        "time_metadata_available": bool(has_time),
        "note": (
            "cond_id = cell_line|compound|dose|time (was cell_line|compound|dose). "
            "H-4 fix: time is part of condition identity, same as dose -- 24h and 72h "
            "cells of the same (cell_line, compound, dose) are now two SEPARATE "
            "conditions and are never pooled into one pseudobulk/delta."
        ),
    }

    # --- control groups per (cell_line, plate) ---
    control_global_idx = np.where(is_control)[0]
    control_key_for_idx = np.array(
        [f"{cl}||{b}" for cl, b in zip(cell_line[control_global_idx], batch[control_global_idx])],
        dtype=object,
    )
    control_idx_by_key: dict[str, np.ndarray] = {}
    control_profile: dict[str, np.ndarray] = {}
    for key in np.unique(control_key_for_idx):
        idx = control_global_idx[control_key_for_idx == key]
        control_idx_by_key[key] = idx
        control_profile[key] = matrix.mean(idx)

    cellline_fallback_idx: dict[str, np.ndarray] = {}
    cellline_fallback_profile: dict[str, np.ndarray] = {}
    for cl in np.unique(cell_line[control_global_idx]):
        idx = control_global_idx[cell_line[control_global_idx] == cl]
        cellline_fallback_idx[cl] = idx
        cellline_fallback_profile[cl] = matrix.mean(idx)

    # --- treated conditions: (cell_line, compound, dose, time) ---
    treated_mask = ~is_control
    treated_global_idx = np.where(treated_mask)[0]
    cond_key = np.array(
        [
            f"{cl}|{co}|{d}|{t}"
            for cl, co, d, t in zip(
                cell_line[treated_global_idx], compound[treated_global_idx],
                dose[treated_global_idx], time[treated_global_idx],
            )
        ],
        dtype=object,
    )

    unique_conds = np.unique(cond_key)
    rows_X, rows_control, rows_obs = [], [], []
    n_dropped_too_few_cells = 0
    n_fallback_control = 0
    cond_plate_cell_idx: dict[str, dict[str, np.ndarray]] = {}
    cond_plate_control_key: dict[str, dict[str, str]] = {}

    # Plate-purity diagnostics (H-3): fraction of a condition's USABLE cells sitting
    # on its single largest plate, BEFORE the fix this was the fraction correctly
    # matched; AFTER the fix every plate is correctly matched regardless, so this is
    # now purely descriptive of how multi-plate conditions typically are.
    modal_plate_purity: list[float] = []
    n_multi_plate_conditions = 0
    n_cells_dropped_plate_no_control = 0
    n_condition_plate_pairs_dropped_no_control = 0

    for ck in unique_conds:
        idx_all = treated_global_idx[cond_key == ck]
        cl, co, d, t = ck.split("|")
        d = float(d)
        t = float(t)

        plate_vals_all, inverse = np.unique(batch[idx_all], return_inverse=True)
        usable_plate_idx: dict[str, np.ndarray] = {}
        usable_plate_key: dict[str, str] = {}
        for pi, pv in enumerate(plate_vals_all):
            plate_cells = idx_all[inverse == pi]
            ckey = f"{cl}||{pv}"
            if ckey in control_profile:
                usable_plate_idx[str(pv)] = plate_cells
                usable_plate_key[str(pv)] = ckey
            else:
                n_cells_dropped_plate_no_control += plate_cells.size
                n_condition_plate_pairs_dropped_no_control += 1

        if not usable_plate_idx:
            # No plate of this condition has a matched vehicle-cell group at all.
            # Last-resort fallback: per-cell-line control (still better than
            # dropping the whole condition), applied to ALL of the condition's
            # cells as a single pseudo-plate.
            if cl in cellline_fallback_profile:
                used_key = f"__cellline_fallback__{cl}"
                control_profile.setdefault(used_key, cellline_fallback_profile[cl])
                control_idx_by_key.setdefault(used_key, cellline_fallback_idx[cl])
                usable_plate_idx = {"__cellline_fallback__": idx_all}
                usable_plate_key = {"__cellline_fallback__": used_key}
                n_fallback_control += 1
            else:
                n_dropped_too_few_cells += 1
                continue

        idx_usable = np.concatenate(list(usable_plate_idx.values()))
        if idx_usable.size < cfg.min_cells_per_condition:
            n_dropped_too_few_cells += 1
            continue

        plate_sizes = {pk: v.size for pk, v in usable_plate_idx.items()}
        total_n = idx_usable.size
        if len(plate_sizes) > 1:
            n_multi_plate_conditions += 1
        modal_plate_purity.append(max(plate_sizes.values()) / total_n)

        # control_c = cell-count-weighted average of each usable plate's OWN matched
        # control profile, weighted by how many of THIS condition's cells sit on
        # that plate. X_c is the plain mean over idx_usable (equivalent to the same
        # cell-count weighting of per-plate means). delta_c = X_c - control_c is
        # then EXACTLY the cell-count-weighted average of each plate's
        # (X_plate - control_plate) -- i.e. every cell's contribution to the
        # condition's delta only ever involves its own plate's control. See
        # module/function docstring.
        control_c = np.zeros(n_genes, dtype=np.float64)
        for pk, n_p in plate_sizes.items():
            control_c += (n_p / total_n) * control_profile[usable_plate_key[pk]]
        X_c = matrix.mean(idx_usable)

        cond_plate_cell_idx[ck] = usable_plate_idx
        cond_plate_control_key[ck] = usable_plate_key
        rows_X.append(X_c)
        rows_control.append(control_c.astype(np.float32))
        rows_obs.append(
            dict(
                cond_id=ck,
                cell_line=cl,
                compound=co,
                dose=d,
                log_dose=float(np.log1p(d)),
                time=t,
                batch=",".join(sorted(plate_sizes.keys())),
                n_cells=int(idx_usable.size),
                is_control=False,
            )
        )

    meta["n_conditions_dropped_too_few_cells"] = int(n_dropped_too_few_cells)
    meta["n_conditions_fallback_control"] = int(n_fallback_control)
    meta["min_cells_per_condition"] = cfg.min_cells_per_condition
    meta["plate_matching"] = {
        "method": (
            "per-(condition, plate) pseudobulk delta against that plate's own "
            "matched vehicle-cell control, cell-count-weighted-averaged across the "
            "condition's plates. 100% of a condition's (usable) cells are compared "
            "only to their OWN plate's control, by construction -- not just the "
            "modal plate's share as before (H-3 fix)."
        ),
        "n_conditions_multi_plate": int(n_multi_plate_conditions),
        "n_conditions_total": int(len(rows_obs)),
        "modal_plate_purity_mean": float(np.mean(modal_plate_purity)) if modal_plate_purity else None,
        "modal_plate_purity_median": float(np.median(modal_plate_purity)) if modal_plate_purity else None,
        "modal_plate_purity_note": (
            "Fraction of each condition's usable cells sitting on its single "
            "largest plate -- descriptive of how multi-plate sciplex3 conditions "
            "are; NOT a measure of mismatch, since every plate (not just the modal "
            "one) is now matched to its own control."
        ),
        "pct_cells_plate_matched": 100.0,
        "n_cells_dropped_plate_no_control": int(n_cells_dropped_plate_no_control),
        "n_condition_plate_pairs_dropped_no_control": int(n_condition_plate_pairs_dropped_no_control),
    }

    n_cond = len(rows_obs)
    Xb = np.asarray(rows_X, dtype=np.float32) if n_cond else np.zeros((0, n_genes), np.float32)
    Cb = np.asarray(rows_control, dtype=np.float32) if n_cond else np.zeros((0, n_genes), np.float32)
    delta = (Xb - Cb).astype(np.float32)

    obs = pd.DataFrame(rows_obs)
    if "target" not in obs.columns:
        obs["target"] = "unknown"
    if n_cond:
        obs = obs.set_index("cond_id")
    else:
        obs = obs.set_index(pd.Index([], name="cond_id"))

    # --- single-cell DE mask, per condition, stratified per plate so treated cells
    # are never tested against another plate's control cells (H-3 fix) ---
    de_mask = np.zeros((n_cond, n_genes), dtype=bool)
    n_zero_de = 0
    for i, ck in enumerate(obs.index):
        plate_pairs = []
        for pk, idx in cond_plate_cell_idx[ck].items():
            used_key = cond_plate_control_key[ck][pk]
            ctrl_idx = control_idx_by_key.get(used_key, np.array([], dtype=int))
            if idx.size == 0 or ctrl_idx.size == 0:
                continue
            treated_dense = matrix.dense(idx)
            ctrl_dense = matrix.dense(ctrl_idx)
            plate_pairs.append((treated_dense, ctrl_dense))
        mask = _stratified_de_mask(plate_pairs, cfg.de_alpha, n_genes)
        de_mask[i] = mask
        if mask.sum() == 0:
            n_zero_de += 1
        del plate_pairs
    meta["n_conditions_zero_de_genes"] = int(n_zero_de)
    meta["n_conditions"] = int(n_cond)

    ctx = _HalvesContext(
        matrix=matrix,
        cond_plate_cell_idx=cond_plate_cell_idx,
        cond_plate_control_key=cond_plate_control_key,
        control_profile=control_profile,
        obs_index=list(obs.index),
        n_genes=n_genes,
        seed=cfg.seed,
        n_rep=cfg.n_ceiling_reps,
    )

    if want_halves:
        halves = compute_halves(ctx, gene_idx=np.arange(n_genes))
        meta["halves_gene_idx"] = list(range(n_genes))
        meta["halves_note"] = "halves computed over all genes (small dataset)"
    else:
        halves = np.zeros((0, 2, 0, 0), dtype=np.float32)

    meta.setdefault("dataset", cfg.dataset)
    meta.setdefault("seed", cfg.seed)

    data = PseudobulkData(
        X=Xb, control=Cb, delta=delta, de_mask=de_mask, halves=halves,
        obs=obs, var=np.asarray(gene_names, dtype=str), meta=meta,
    )
    return data, ctx


# ---------------------------------------------------------------------------
# sciplex3 loader
# ---------------------------------------------------------------------------

_CELL_LINE_CANDIDATES = ["cell_line", "celltype", "cell_type", "cell_line_short"]
_COMPOUND_CANDIDATES = ["compound", "perturbation", "product_name", "drug"]
_DOSE_CANDIDATES = ["dose_value", "dose", "concentration"]
_BATCH_CANDIDATES = ["plate", "batch", "culture_plate", "hash_plate", "well", "rt_well", "pcr_well", "replicate"]
_TARGET_CANDIDATES = ["target", "pathway", "pathway_level_1", "moa", "gene_target"]
_TIME_CANDIDATES = ["time", "timepoint", "time_point", "hours", "treatment_time"]
_CONTROL_VALUES = {"vehicle", "control", "dmso", "none", "untreated", "ctrl"}


def _pick_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


class Sciplex3Loader:
    name = "sciplex3"

    def _raw_path(self, cfg: Config) -> Path:
        return Path(cfg.cache_dir) / "sciplex3_raw.h5ad"

    def _load_raw_backed(self, cfg: Config):
        import anndata as ad

        raw_path = self._raw_path(cfg)
        if not raw_path.exists():
            import pertpy

            print("Downloading sciplex3 via pertpy (multi-GB, be patient)...", flush=True)
            adata_full = pertpy.data.srivatsan_2020_sciplex3()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            adata_full.write_h5ad(raw_path)
            del adata_full
            gc.collect()
            source = "pertpy.data.srivatsan_2020_sciplex3() [freshly downloaded]"
        else:
            source = f"cached_h5ad:{raw_path}"
        adata = ad.read_h5ad(raw_path, backed="r")
        return adata, source

    def _streaming_qc_and_gene_counts(
        self, adata, mito_mask: Optional[np.ndarray], cfg: Config, chunk_size: int = 100_000
    ):
        n_cells, n_genes = adata.shape
        n_genes_detected = np.zeros(n_cells, dtype=np.int32)
        total_counts = np.zeros(n_cells, dtype=np.float64)
        mito_counts = np.zeros(n_cells, dtype=np.float64) if mito_mask is not None else None
        gene_nnz = np.zeros(n_genes, dtype=np.int64)

        t0 = time.time()
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = adata.X[start:end]
            n_genes_detected[start:end] = np.asarray((chunk > 0).sum(axis=1)).ravel()
            total_counts[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
            if mito_mask is not None:
                mito_counts[start:end] = np.asarray(chunk[:, mito_mask].sum(axis=1)).ravel()
            gene_nnz += np.asarray((chunk > 0).sum(axis=0)).ravel()
            print(f"  QC streaming pass: {end}/{n_cells} cells ({time.time()-t0:.1f}s)", flush=True)

        keep = n_genes_detected >= cfg.min_genes
        qc_meta = {
            "n_cells_before_qc": int(n_cells),
            "n_cells_dropped_min_genes": int((~keep).sum()),
        }
        pct_mito = None
        if mito_mask is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_mito = np.where(total_counts > 0, 100.0 * mito_counts / total_counts, 0.0)
            mito_keep = pct_mito <= cfg.max_pct_mito
            qc_meta["mito_filter_applied"] = True
            qc_meta["n_cells_dropped_mito"] = int((~mito_keep & keep).sum())
            # §1.2 / fix-plan #9: record the FULL pct-mito distribution (quantiles),
            # independent of the chosen threshold, so report.py can show the
            # threshold is a real outlier trim and not a cut through the middle of
            # the distribution.
            qc_meta["mito_quantiles"] = {
                "median": float(np.median(pct_mito)),
                "p25": float(np.percentile(pct_mito, 25)),
                "p75": float(np.percentile(pct_mito, 75)),
                "p90": float(np.percentile(pct_mito, 90)),
                "p99": float(np.percentile(pct_mito, 99)),
            }
            qc_meta["max_pct_mito_threshold"] = float(cfg.max_pct_mito)
            qc_meta["frac_cells_dropped_by_mito_overall"] = float((~mito_keep & keep).sum() / max(1, keep.sum()))
            keep = keep & mito_keep
        else:
            qc_meta["mito_filter_applied"] = False
            qc_meta["mito_filter_note"] = (
                "No mitochondrial genes identifiable from gene symbols; skipping the "
                "mito-percent QC filter rather than crashing."
            )
        qc_meta["n_cells_after_qc"] = int(keep.sum())
        return keep, total_counts, gene_nnz, qc_meta, pct_mito, n_genes_detected

    def load(self, cfg: Config) -> PseudobulkData:
        adata, source = self._load_raw_backed(cfg)
        columns = list(adata.obs.columns)

        cl_col = _pick_column(columns, _CELL_LINE_CANDIDATES)
        co_col = _pick_column(columns, _COMPOUND_CANDIDATES)
        dose_col = _pick_column(columns, _DOSE_CANDIDATES)
        batch_col = _pick_column(columns, _BATCH_CANDIDATES)
        target_col = _pick_column(columns, _TARGET_CANDIDATES)
        time_col = _pick_column(columns, _TIME_CANDIDATES)

        if cl_col is None or co_col is None or dose_col is None:
            raise RuntimeError(
                f"Sciplex3Loader could not auto-detect required obs columns among "
                f"{columns}. Found cell_line={cl_col}, compound={co_col}, dose={dose_col}. "
                f"See DATA_RECON.md and update _*_CANDIDATES in data.py."
            )

        gene_names_full = np.asarray(adata.var_names, dtype=str)
        mito_mask = detect_mito_genes(gene_names_full)

        print("Running streaming QC + gene-detection pass over all cells...", flush=True)
        cell_keep, total_counts, gene_nnz, qc_meta, pct_mito_full, n_genes_detected_full = (
            self._streaming_qc_and_gene_counts(adata, mito_mask, cfg)
        )

        n_cells_total = adata.shape[0]
        gene_keep_mask = gene_nnz >= max(3, int(np.ceil(GENE_DETECTION_MIN_FRAC * n_cells_total)))
        gene_keep_idx = np.where(gene_keep_mask)[0]
        gene_names = gene_names_full[gene_keep_idx]

        median_total = float(np.median(total_counts[total_counts > 0])) if np.any(total_counts > 0) else 1.0

        matrix = _BackedMatrixAdapter(adata, gene_keep_idx, total_counts, median_total)

        compound_vals_full = adata.obs[co_col].astype(str).to_numpy()
        is_control_full = np.array([c.lower() in _CONTROL_VALUES for c in compound_vals_full])
        cell_line_full = adata.obs[cl_col].astype(str).to_numpy()
        dose_full = pd.to_numeric(adata.obs[dose_col], errors="coerce").fillna(0.0).to_numpy()
        batch_full = adata.obs[batch_col].astype(str).to_numpy() if batch_col else None
        if time_col is not None:
            time_full = pd.to_numeric(adata.obs[time_col], errors="coerce").fillna(0.0).to_numpy()
        else:
            time_full = np.zeros(n_cells_total, dtype=float)

        # §1.2 / fix-plan #9: mito drop rate per cell line AND per dose, so the
        # report can show the filter is no longer dose-/context-biased. Computed
        # among cells that pass min_genes (the `min_genes_keep` mask below),
        # independent of gene-detection filtering.
        if pct_mito_full is not None:
            min_genes_keep = n_genes_detected_full >= cfg.min_genes  # pre-mito keep mask
            mito_fail = pct_mito_full > cfg.max_pct_mito
            by_line = {}
            for cl in np.unique(cell_line_full):
                denom_mask = min_genes_keep & (cell_line_full == cl)
                denom = int(denom_mask.sum())
                by_line[cl] = float(mito_fail[denom_mask].sum() / denom) if denom else None
            dose_bucket_full = np.where(is_control_full, -1.0, dose_full)
            by_dose = {}
            for dv in np.unique(dose_bucket_full):
                denom_mask = min_genes_keep & (dose_bucket_full == dv)
                denom = int(denom_mask.sum())
                label = "vehicle" if dv == -1.0 else str(dv)
                by_dose[label] = float(mito_fail[denom_mask].sum() / denom) if denom else None
            qc_meta["mito_drop_rate_by_cell_line"] = by_line
            qc_meta["mito_drop_rate_by_dose"] = by_dose

        keep_idx = np.where(cell_keep)[0]
        cell_line = cell_line_full[keep_idx]
        compound_vals = compound_vals_full[keep_idx]
        dose = dose_full[keep_idx]
        is_control = is_control_full[keep_idx]
        batch = batch_full[keep_idx] if batch_full is not None else None
        time_vals = time_full[keep_idx]

        # Remap matrix adapter's row space: build_pseudobulk_core works with positional
        # indices into the KEPT-cell arrays above, but the matrix adapter's `.dense`/
        # `.mean` fetch against the ORIGINAL (pre-QC) row space. Wrap with a translation.
        class _ReindexedMatrix:
            def __init__(self, inner, keep_idx):
                self.inner = inner
                self.keep_idx = keep_idx
                self.n_genes = inner.n_genes

            def mean(self, idx):
                return self.inner.mean(self.keep_idx[idx])

            def dense(self, idx, gene_idx=None):
                return self.inner.dense(self.keep_idx[idx], gene_idx)

        reindexed_matrix = _ReindexedMatrix(matrix, keep_idx)

        meta_extra = {
            "dataset": "sciplex3",
            "source": source,
            "pertpy_load_fn": "pertpy.data.srivatsan_2020_sciplex3",
            "qc": qc_meta,
            "normalization": {
                "detected_input": "raw_counts",
                "normalization_applied": "normalize_total (per-cell, full-transcriptome total, "
                                          "scaled to dataset median total) + log1p, computed "
                                          "on the fly per fetched chunk (streamed, never densified "
                                          "for the full matrix)",
            },
            "gene_detection_filter": {
                "applied": True,
                "min_frac_cells": GENE_DETECTION_MIN_FRAC,
                "n_genes_before": int(len(gene_names_full)),
                "n_genes_after": int(len(gene_names)),
                "note": (
                    "Dataset-global filter: a gene must be detected (count>0) in at "
                    "least 1% of cells (post-QC... actually computed pre-cell-QC for "
                    "efficiency, a superset) to be retained. This uses only detection "
                    "presence/absence across the WHOLE dataset, never condition labels "
                    "or effect sizes, so it is not leakage -- same class of operation "
                    "as the mitochondrial-content QC filter. Needed because the raw "
                    "feature space (110,983) makes per-condition pseudobulk and the "
                    "split-half noise ceiling exceed available disk (~24GB free) and "
                    "RAM (~26GB) on this machine."
                ),
            },
            "detected_columns": {
                "cell_line": cl_col,
                "compound": co_col,
                "dose": dose_col,
                "batch": batch_col,
                "target": target_col,
                "time": time_col,
            },
        }

        print("Building pseudobulk (X, control, delta, de_mask) -- skipping halves for now...", flush=True)
        data, ctx = build_pseudobulk_core(
            matrix=reindexed_matrix,
            cell_line=cell_line,
            compound=compound_vals,
            dose=dose,
            is_control=is_control,
            batch=batch,
            gene_names=gene_names,
            cfg=cfg,
            meta_extra=meta_extra,
            want_halves=False,
            time=time_vals,
        )

        # §1.2 H-4 sanity counts (real numbers, not estimates): how many cells/
        # conditions in the surviving pseudobulk involve time != 24 at all, now
        # that time is part of cond_id so they can never be pooled with 24h cells
        # of the same (cell_line, compound, dose).
        if len(data.obs):
            is_72 = (data.obs["time"] != 24.0) & (data.obs["time"] != 0.0)
            data.meta["time_identity"]["n_conditions_time_ne_24"] = int(is_72.sum())
            data.meta["time_identity"]["n_cells_in_time_ne_24_conditions"] = int(
                data.obs.loc[is_72, "n_cells"].sum()
            )

        if target_col is not None and len(data.obs):
            target_lookup = (
                pd.DataFrame({"compound": compound_vals, "target": adata.obs[target_col].astype(str).to_numpy()[keep_idx]})
                .drop_duplicates("compound")
                .set_index("compound")["target"]
            )
            data.obs["target"] = data.obs["compound"].map(target_lookup).fillna("unknown")

        # --- gene-union halves (see module docstring for why) ---
        # M-4 fix: gene_set="all" means every fold's gene selection IS all genes
        # (run.py's _select_genes returns arange(n_genes) for gene_set="all"), so the
        # halves cache must cover all genes too -- the fold-HVG-union logic below
        # would otherwise still only union each fold's *HVG* selection (computed via
        # select_hvgs regardless of cfg.gene_set) and raise a stale-cache assertion
        # the first time a gene_set="all" run actually tries to score a ceiling.
        if cfg.gene_set == "all":
            print("gene_set='all': caching halves over ALL genes (not the HVG union).", flush=True)
            union_idx = np.arange(data.X.shape[1])
        else:
            print("Computing fold-HVG gene union across all split regimes for halves caching...", flush=True)
            union_idx = _compute_fold_hvg_union(data, cfg)
        n_rep = cfg.n_ceiling_reps
        est_bytes = n_rep * 2 * data.X.shape[0] * len(union_idx) * 4
        if est_bytes > 2 * 1024 ** 3 and n_rep > 5:
            n_rep = 5
            data.meta["n_ceiling_reps_reduced"] = {
                "from": cfg.n_ceiling_reps, "to": n_rep,
                "reason": f"estimated halves size {est_bytes/1e9:.2f}GB exceeded 2GB budget",
            }
            ctx.n_rep = n_rep
        print(f"Computing halves over {len(union_idx)} union genes, n_rep={n_rep}...", flush=True)
        halves = compute_halves(ctx, gene_idx=union_idx)
        data.halves = halves
        data.meta["halves_gene_idx"] = union_idx.tolist()
        data.meta["halves_n_genes"] = int(len(union_idx))
        data.meta["halves_size_bytes"] = int(halves.nbytes)
        data.meta["halves_note"] = (
            "halves cached only over the union of per-fold, train-only HVG-selected "
            "genes across every split regime -- not all genes. See module docstring. "
            "Not leakage: each fold scores only on its own train-selected genes."
        )

        return data


def _compute_fold_hvg_union(data: PseudobulkData, cfg: Config) -> np.ndarray:
    """Run make_folds for every regime, select_hvgs (train-only) per fold, and union
    the resulting gene indices. Falls back to a plain top-n_hvg-by-global-variance
    selection if splits.py isn't importable yet (keeps this module independently
    testable/usable even before splits.py exists)."""
    n_hvg = cfg.n_hvg if cfg.n_hvg is not None else data.X.shape[1]
    try:
        from .splits import make_folds, REGIMES
    except Exception as e:  # pragma: no cover
        print(f"  (splits.py unavailable ({e}); falling back to single global HVG selection)", flush=True)
        return select_hvgs(data, np.arange(data.X.shape[0]), n_hvg)

    union = set()
    for regime in REGIMES:
        folds = make_folds(data.obs.reset_index(), regime, cfg.seed)
        for fold in folds:
            if fold.train_idx.size == 0:
                continue
            genes = select_hvgs(data, fold.train_idx, n_hvg)
            union.update(genes.tolist())
    if not union:
        return select_hvgs(data, np.arange(data.X.shape[0]), n_hvg)
    return np.sort(np.array(sorted(union), dtype=int))


register_loader(Sciplex3Loader())


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------

def _cache_paths(cfg: Config) -> dict[str, Path]:
    base = Path(cfg.cache_dir) / f"pseudobulk_{cfg.dataset}_{cfg.cache_key()}"
    return {
        "dir": base,
        "X": base / "X.npy",
        "control": base / "control.npy",
        "delta": base / "delta.npy",
        "de_mask": base / "de_mask.npy",
        "halves": base / "halves.dat",
        "halves_shape": base / "halves_shape.json",
        "var": base / "var.npy",
        "obs": base / "obs.parquet",
        "meta": base / "meta.json",
    }


def _save_pseudobulk(data: PseudobulkData, cfg: Config) -> None:
    paths = _cache_paths(cfg)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    np.save(paths["X"], data.X)
    np.save(paths["control"], data.control)
    np.save(paths["delta"], data.delta)
    np.save(paths["de_mask"], data.de_mask)

    halves = np.asarray(data.halves, dtype=np.float32)
    mm = np.memmap(paths["halves"], dtype=np.float32, mode="w+", shape=halves.shape)
    mm[:] = halves[:]
    mm.flush()
    del mm
    with open(paths["halves_shape"], "w") as f:
        json.dump({"shape": list(halves.shape)}, f)

    np.save(paths["var"], data.var)
    data.obs.to_parquet(paths["obs"])
    with open(paths["meta"], "w") as f:
        json.dump(data.meta, f, indent=2, default=str)


def _load_pseudobulk_cached(cfg: Config) -> Optional[PseudobulkData]:
    paths = _cache_paths(cfg)
    required = [paths["X"], paths["control"], paths["delta"], paths["de_mask"],
                paths["halves"], paths["halves_shape"], paths["var"], paths["obs"], paths["meta"]]
    if not all(p.exists() for p in required):
        return None

    X = np.load(paths["X"])
    control = np.load(paths["control"])
    delta = np.load(paths["delta"])
    de_mask = np.load(paths["de_mask"])
    var = np.load(paths["var"], allow_pickle=False)
    obs = pd.read_parquet(paths["obs"])
    with open(paths["meta"]) as f:
        meta = json.load(f)
    with open(paths["halves_shape"]) as f:
        shape = tuple(json.load(f)["shape"])
    halves = np.memmap(paths["halves"], dtype=np.float32, mode="r", shape=shape) if shape[0] else np.zeros(shape, dtype=np.float32)

    return PseudobulkData(
        X=X, control=control, delta=delta, de_mask=de_mask, halves=halves,
        obs=obs, var=var, meta=meta,
    )


def load_pseudobulk(cfg: Config) -> PseudobulkData:
    """Cached entry point. Loads from cfg.cache_dir if a pseudobulk with a matching
    cache key already exists there; otherwise computes it via the registered loader
    for cfg.dataset and caches the result. Cache stores ALL (detection-filtered)
    genes for X/control/delta/de_mask; `halves` is restricted to a fold-HVG-union
    gene subset (see module docstring) -- this function asserts that invariant holds
    for every fold of every regime before returning."""
    cached = _load_pseudobulk_cached(cfg)
    if cached is None:
        loader = get_loader(cfg.dataset)
        cached = loader.load(cfg)
        _save_pseudobulk(cached, cfg)

    _assert_halves_gene_coverage(cached, cfg)
    return cached


def _assert_halves_gene_coverage(data: PseudobulkData, cfg: Config) -> None:
    halves_gene_idx = data.meta.get("halves_gene_idx")
    if halves_gene_idx is None:
        return  # e.g. a synthetic/test build that computed halves over all genes
    # L-1 fix (REVIEW.md): assert HERE, once, at the loader boundary, that
    # halves_gene_idx is sorted ascending and therefore matches the column order of
    # `halves` (which is always written in the order this list was computed in --
    # see `_compute_fold_hvg_union`'s `np.sort` and the `gene_set="all"`
    # `np.arange` branch above, both already sorted). Downstream consumers
    # (run.py's `_ceiling_for_fold`) can then `searchsorted` directly against this
    # list instead of re-sorting it themselves every call.
    assert list(halves_gene_idx) == sorted(halves_gene_idx), (
        "meta['halves_gene_idx'] is not sorted ascending, but callers assume its "
        "order matches the column order of the cached `halves` array and index into "
        "it with searchsorted without re-sorting. Stale or hand-built cache -- "
        "delete the cache dir and recompute."
    )
    halves_gene_set = set(halves_gene_idx)
    if len(halves_gene_set) == data.X.shape[1]:
        return  # halves covers all genes, trivially a superset of any selection
    try:
        from .splits import make_folds, REGIMES
    except Exception:
        return
    n_hvg = cfg.n_hvg if cfg.n_hvg is not None else data.X.shape[1]
    for regime in REGIMES:
        for fold in make_folds(data.obs.reset_index(), regime, cfg.seed):
            if fold.train_idx.size == 0:
                continue
            genes = select_hvgs(data, fold.train_idx, n_hvg)
            missing = set(genes.tolist()) - halves_gene_set
            assert not missing, (
                f"Fold {fold.name} selected {len(missing)} genes not covered by the "
                f"cached halves gene union. The cache is stale relative to splits.py; "
                f"delete the cache dir and recompute."
            )


# ---------------------------------------------------------------------------
# HVG helper (train-only; callers pass training-row indices)
# ---------------------------------------------------------------------------

def select_hvgs(data: PseudobulkData, train_idx: np.ndarray, n_hvg: int) -> np.ndarray:
    """Select the top-`n_hvg` most variable genes using ONLY the pseudobulk X rows at
    `train_idx` (e.g. training conditions in a fold). Returns positional gene indices
    (into data.var / data.X columns), sorted ascending. Callers are responsible for
    only ever calling this with training rows -- fitting HVG selection on held-out
    data is a leakage trap this harness exists to prevent."""
    if n_hvg is None or n_hvg >= data.X.shape[1]:
        return np.arange(data.X.shape[1])
    X_train = data.X[train_idx]
    var = X_train.var(axis=0)
    top = np.argsort(var)[::-1][:n_hvg]
    return np.sort(top)
