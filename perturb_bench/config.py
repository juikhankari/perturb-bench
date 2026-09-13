"""Configuration for the perturbation-response benchmark harness.

Config is a plain dataclass, loadable from YAML, with a stable hash over only the
fields that affect the cached pseudobulk (QC / preprocessing-relevant). Fields like
out_dir or cache_dir do not affect what gets computed, so they are excluded from the
cache key — otherwise moving the cache dir would invalidate a perfectly good cache.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

import yaml

# Fields that determine the CONTENT of the cached pseudobulk data. Anything not in
# this list (out_dir, cache_dir, n_ceiling_reps... wait n_ceiling_reps affects halves
# shape so it IS content-relevant) is excluded from the cache key.
_CACHE_KEY_FIELDS = (
    "dataset",
    "seed",
    "min_genes",
    "max_pct_mito",
    "min_cells_per_condition",
    "de_alpha",
    "n_ceiling_reps",
    # n_hvg / gene_set ARE included (REVIEW.md M-4): X/control/delta/de_mask always
    # store ALL genes regardless of these two fields, but `halves` does NOT — it is
    # cached only over the fold-HVG-union gene subset (ADDENDUM 2 to CONTRACT.md),
    # and that union depends on n_hvg, and on gene_set ("all" caches halves over
    # every gene instead of the HVG union). A cache built with n_hvg=2000 is not a
    # valid cache for a run with n_hvg=500 or gene_set="all" even though QC params
    # are identical, so both fields must be part of the key.
    "n_hvg",
    "gene_set",
)


@dataclass
class Config:
    dataset: str = "sciplex3"
    seed: int = 0
    min_genes: int = 200
    # REVIEW.md §1.2 / fix-plan #9: 20% is the 10x whole-cell convention and cuts
    # through the MIDDLE of sciplex3's pct-mito distribution (median 16.95%, p75
    # 23.51%, p90 30.27%), dropping 37% of cells overall and doing so dose- and
    # cell-line-dependently (A549 46.6% vs K562 16.2%; vehicle 47.2% vs treated
    # 36-38%) -- i.e. it preferentially deletes stressed high-dose cells and biases
    # exactly the held_out_context axis. sci-Plex is combinatorial indexing (sci-
    # RNA-seq3), where elevated mito fraction is normal chemistry, not damage; high-
    # mito cells here have HIGHER median library size, not lower. 50% is a real
    # outlier trim (drops 0.68% overall, flat 0.5-0.9% across both dose and cell
    # line -- see meta["qc"]["mito_quantiles"] / ["mito_drop_rate_by_*"] recorded by
    # the loader for the actual measured numbers on each run).
    max_pct_mito: float = 50.0
    min_cells_per_condition: int = 30
    n_hvg: Optional[int] = 2000
    gene_set: str = "hvg"  # "hvg" | "all"
    n_ceiling_reps: int = 10
    cache_dir: str = "cache"
    out_dir: str = "outputs"
    de_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.gene_set not in ("hvg", "all"):
            raise ValueError(f"gene_set must be 'hvg' or 'all', got {self.gene_set!r}")
        if self.gene_set == "hvg" and self.n_hvg is None:
            raise ValueError("gene_set='hvg' requires n_hvg to be set")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_key(self) -> str:
        """Stable hash over QC/preprocessing-relevant fields only (not seed-independent
        bookkeeping fields like out_dir/cache_dir). Used to key the on-disk pseudobulk
        cache so changing e.g. out_dir doesn't invalidate a perfectly good cache, but
        changing QC thresholds does."""
        payload = {k: getattr(self, k) for k in _CACHE_KEY_FIELDS}
        payload["dataset"] = self.dataset
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
