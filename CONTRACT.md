# Internal interface contract — perturbation response benchmark harness

All modules MUST conform to this. Do not change a signature without saying so loudly.
Package lives in `perturb_bench/`. Deps: numpy, pandas, scipy, scanpy/anndata,
scikit-learn, statsmodels. NO deep learning libraries. Ever.

## Core container (`perturb_bench/data.py`)

```python
@dataclass
class PseudobulkData:
    X:       np.ndarray  # (n_cond, n_genes) float32. Treated pseudobulk, mean log1p-normalized expr.
    control: np.ndarray  # (n_cond, n_genes) float32. Matched control profile for that condition.
    delta:   np.ndarray  # (n_cond, n_genes) float32. == X - control.
    de_mask: np.ndarray  # (n_cond, n_genes) bool. True = gene is DE for this condition,
                         #   from a SINGLE-CELL Wilcoxon of treated vs matched control,
                         #   BH-FDR < 0.05. NEVER from ranking true deltas.
    halves:  np.ndarray  # (n_rep, 2, n_cond, n_genes) float32 split-half pseudobulk DELTAS
                         #   (each half minus the same matched control). n_rep=10.
                         #   May be a np.memmap. Used only for the noise ceiling.
    obs:     pd.DataFrame  # index = cond_id, row order matches axis 0 of X/control/delta/de_mask.
                           # Required columns:
                           #   cell_line (str), compound (str), dose (float), log_dose (float),
                           #   batch (str), n_cells (int), is_control (bool, always False here),
                           #   target (str or "unknown")  <- pathway/target annotation if available
    var:     np.ndarray  # (n_genes,) gene symbols, str
    meta:    dict        # provenance: dataset name, source, version, QC params, counts dropped,
                         #   whether real batch metadata existed (key: "batch_metadata_available": bool)
```

`cond_id` is the string `f"{cell_line}|{compound}|{dose}"`. Dose is part of condition
identity — never collapse over it.

Loader entry point (interface so a 2nd dataset can be added later):

```python
class DatasetLoader(Protocol):
    name: str
    def load(self, cfg: Config) -> PseudobulkData: ...

def get_loader(name: str) -> DatasetLoader   # "sciplex3" registered; "tahoe" raises NotImplementedError
def load_pseudobulk(cfg) -> PseudobulkData   # cached to cfg.cache_dir, keyed by hash of QC params
```

Cache stores ALL genes. Gene selection (HVG) happens downstream, per fold, train-only.

## Splits (`perturb_bench/splits.py`)

```python
@dataclass
class Fold:
    regime: str            # "random" | "held_out_context" | "held_out_drug" | "held_out_both"
    name: str              # e.g. "held_out_context/A549"
    train_idx: np.ndarray  # int positional indices into PseudobulkData rows
    test_idx:  np.ndarray
    meta: dict             # e.g. {"held_out_cell_lines": [...], "held_out_compounds": [...]}

def make_folds(obs: pd.DataFrame, regime: str, seed: int) -> list[Fold]
def assert_no_leakage(obs, fold) -> None   # raises AssertionError; called for every fold
```

`held_out_context` yields one fold per cell line (3 folds).
`held_out_drug` groups compounds by `obs.target` so siblings don't leak; falls back to
random compound groups and sets `meta["target_annotation_available"]=False`.
`held_out_both` = unseen cell line x unseen compound (test set is the intersection).

## Baselines (`perturb_bench/baselines.py`)

```python
class Baseline(Protocol):
    name: str
    def fit(self, data: PseudobulkData, train_idx: np.ndarray, genes: np.ndarray) -> None: ...
    def predict(self, data: PseudobulkData, test_idx: np.ndarray) -> np.ndarray:
        """returns (n_test, n_genes) predicted DELTA, genes in the order given to fit()"""
    @property
    def fallback_mask(self) -> np.ndarray:  # bool (n_test,), True where the baseline
                                            # could not be applied and fell back
```

`genes` is the positional index array of the fold's gene selection. Baselines must only
ever touch `train_idx` rows during `fit`. B1 no_change, B2 global_mean_delta,
B3 per_drug_mean_delta, B4 ridge, B5 nearest_context. Registry: `BASELINES: dict[str, type]`.

## Metrics (`perturb_bench/metrics.py`)

```python
def score_conditions(
    pred: np.ndarray,       # (n_test, n_genes) predicted DELTA
    true: np.ndarray,       # (n_test, n_genes) true DELTA
    control: np.ndarray,    # (n_test, n_genes) matched control, to build the absolute rep
    de_mask: np.ndarray,    # (n_test, n_genes) bool
    representation: str,    # "delta" -> score pred vs true
                            # "absolute" -> score (control+pred) vs (control+true)
) -> pd.DataFrame:
    """one row per test condition, columns = metric names. NaN where undefined."""
```

Metrics: `pearson_r`, `mae`, `pearson_r_de`, `direction_acc_de`, `pert_discrimination`
(normalized rank of the correct true profile when ranking all held-out true profiles by
distance to the prediction; 0 = perfect, so report it and normalize accordingly).

Aggregation must carry variance: `aggregate(df) -> mean, std, n, n_skipped_no_de`.

## Noise ceiling (`perturb_bench/ceiling.py`)
Scores `halves[r,0]` against `halves[r,1]` with the SAME metric suite on the SAME gene
subset, averaged over reps and conditions. Same output shape as a baseline's scores.

## Normalized score
`normalized = (score - floor) / (ceiling - floor)`, floor = the `no_change` baseline on
that metric/fold/representation. For metrics where lower is better (mae,
pert_discrimination) the same formula still works because floor/ceiling bracket it —
implement it sign-agnostically and unit-test that.

## Outputs (`perturb_bench/run.py`)
- `results.csv`: long, one row per (split, fold, baseline, metric, representation) with
  columns: split, fold, baseline, metric, representation, value, std, n_conditions,
  normalized, fallback_rate. `no_change` floor and `ceiling` appear as pseudo-baselines.
- `report.md`: auto-generated, see spec §8.
- Single command: `python -m perturb_bench.run --config configs/default.yaml`
- Everything seeded. Full sweep target < 10 min given a warm pseudobulk cache.

## ADDENDUM 1 — per-metric floor policy (orchestrator decision, binding)

`no_change` (B1) predicts an all-zero vector. That vector has zero variance, so on the
`delta` representation `pearson_r` and `pearson_r_de` are mathematically UNDEFINED (NaN),
not 0. metrics.py correctly returns NaN. But SPEC §7 normalizes against a floor, and you
cannot normalize against NaN. SPEC §7 explicitly allows "the no_change baseline OR the
context-mean baseline, whichever is more appropriate per metric". Therefore:

| metric | floor | rationale |
|---|---|---|
| pearson_r | `global_mean_delta` (B2) | B1 undefined; B2 is the honest "generic response, no drug identity" floor |
| pearson_r_de | `global_mean_delta` (B2) | same |
| mae | `no_change` (B1) | well-defined and is exactly the "predict nothing" floor |
| direction_acc_de | `no_change` (B1) | well-defined (B1 scores ~0, at/below chance) |
| pert_discrimination | `no_change` (B1) | well-defined (B1 ties to ~0.5 chance) |

run.py MUST record which baseline supplied the floor for each metric in a `floor_source`
column of results.csv, and report.md MUST state the substitution and why. Do not silently
coerce NaN to 0 anywhere — that would hand B1 a fake score and defeat the entire point of
having an inflation detector.

Both B1 and B2 still appear as ordinary rows in results.csv regardless of floor duty, and
the ceiling appears as a pseudo-baseline row.

## ADDENDUM 2 — `halves` gene-union caching (orchestrator correction, binding)

The original contract said cache `halves` over ALL genes. That was WRONG: at
(n_rep=10, 2, n_cond~600, n_genes) it is terabytes at sciplex3's raw gene count and ~14 GB
even at 30k genes, against 25 GB of free disk. Superseded by:

- X / control / delta / de_mask stay cached at ALL post-QC genes (these are only
  (n_cond, n_genes) and are affordable).
- `halves` is cached ONLY over a gene UNION SUPERSET: the union, across every fold of all
  four regimes, of that fold's TRAIN-ONLY HVG selection. Companion array `halves_gene_idx`
  maps the union back into the full gene axis.

WHY THIS IS NOT LEAKAGE: each fold still scores only on its own train-selected genes. The
cache merely holds a superset spanning other folds' selections. No fold's gene selection is
informed by its own held-out data — that is the invariant SPEC §3/§11 actually requires.
Selecting genes *using test rows* would be leakage; this is not that. Loader asserts every
fold's gene set is a subset of the union.

If any dataset-global gene filter (e.g. detection rate) is applied to keep the full-gene
arrays tractable, it MUST be recorded in meta with its threshold and surfaced in report.md
as a dataset-global (not train-only) preprocessing step, with the rationale that it uses
only detection and never condition labels or effect sizes.

## ADDENDUM 3 — ceiling commensurability (supersedes part of ADDENDUM 1). BINDING.

ADDENDUM 1 claimed: "For metrics where lower is better (mae, pert_discrimination) the same
formula still works because floor/ceiling bracket it." **THAT SENTENCE IS FALSE AND IS
WITHDRAWN.** Measured on real folds: the split-half ceiling MAE (0.0247-0.0394) is WORSE
than the no_change floor MAE (0.0230-0.0325), because each half-pseudobulk carries ~2x the
variance of the full pseudobulk and their difference ~4x. normalize_score then divides by a
denominator of the wrong sign and reports baselines that are worse than predicting nothing
as normalized +1.07 to +7.31. The root cause is that the model and the ceiling were being
scored against TARGETS WITH DIFFERENT NOISE LEVELS.

### Decision 1 — replicate-matched scoring becomes primary
Score both baselines and the ceiling against the SAME noisy target:
- baseline score: pred (fit on train) vs `halves[r,1]` as the target
- ceiling score:  `halves[r,0]` vs `halves[r,1]`
averaged over reps r. Model and ceiling then face identical target noise, so the ceiling is
a true upper bound for EVERY metric with no per-metric analytic correction, and
`direction_acc_de`/`pert_discrimination` — which have no clean analytic correction — are
handled by the same mechanism as the rest. This is also exactly the "data from a real
replicate experiment" convention SPEC §7 cites for the Arc 2026 Virtual Cell Challenge,
which SPEC §7 explicitly wants us comparable to.

### Decision 2 — full-pseudobulk scoring is retained as a labeled secondary
Scoring against the full pseudobulk delta is still reported, in a clearly-labeled secondary
table, because it is what most published numbers use. It must NOT carry a normalized score
unless the guard in Decision 3 passes. Add a `target` column to results.csv with values
`replicate_half` (primary) and `full_pseudobulk` (secondary).

### Decision 3 — direction guard, non-negotiable, independent of Decisions 1-2
Before emitting any `normalized` value, check `sign(ceiling_mean - floor_mean)` against
`METRIC_DIRECTION[metric]`. If it contradicts, emit `normalized = NaN`, set a new
`normalization_valid = False` column, and log loudly to stderr. A normalized number whose
scale is inverted must NEVER reach results.csv or report.md. This guard stays permanently
even once Decisions 1-2 make inversion unlikely — it is the backstop that would have caught
this bug.

### Decision 4 — direction_acc_de floor (amends ADDENDUM 1's table)
`direction_acc_de` floor changes from `no_change` to `global_mean_delta`. `no_change` scores
exactly 0.0 there only because `sign(0) != sign(x)` — an artifact of the zero vector, not a
floor, and normalizing against it inflates every score (B3 reads 0.64-0.69 against the
artifact vs ~0.02-0.28 against the honest no-drug-identity floor). This is the same reason
ADDENDUM 1 already uses B2 for `pearson_r`. `pert_discrimination`'s no_change floor of
exactly 0.5 IS genuine chance and stays.

Updated floor table:
| metric | floor | why |
|---|---|---|
| pearson_r | global_mean_delta | no_change undefined (zero variance) |
| pearson_r_de | global_mean_delta | same |
| mae | no_change | genuine "predict nothing" floor |
| direction_acc_de | global_mean_delta | no_change's 0.0 is a sign(0) artifact, not a floor |
| pert_discrimination | no_change | 0.5 is genuine chance |
