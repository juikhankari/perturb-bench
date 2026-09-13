# perturb-bench

A perturbation-response benchmark harness for predicting transcriptional response
to drug perturbation in cellular contexts a model has not seen. This is
measurement infrastructure plus deliberately simple baselines, **not** a model --
see `SPEC.md` §0/§2 for why (neural models are explicitly out of scope).

The harness exists because evaluation, not modeling, is the usual failure mode in
this literature (`SPEC.md` §1): metrics on absolute expression are dominated by
baseline expression, splits leak held-out cell lines/compounds into training, and
nobody reports a noise ceiling. This harness makes those three mistakes hard to
make by accident.

## Quick start

```bash
.venv/bin/python -m perturb_bench.run --config configs/default.yaml
```

Produces `outputs/results.csv`, `outputs/report.md`, and `outputs/run_meta.json`.
Add `--quick` for a fast iteration pass (fewer folds, a capped gene subset --
still train-only, still leakage-checked) and `--out-dir DIR` to write elsewhere.

Requires a warm pseudobulk cache under `cache/` (see `perturb_bench/data.py`);
the first run against a given config will build it, which involves downloading
and streaming through the full sciplex3 h5ad and can take a long time and
significant disk. Do not run two builds concurrently against the same cache dir.

## What gets run

For every split regime in `perturb_bench.splits.REGIMES` (`random`,
`held_out_context`, `held_out_drug`, `held_out_both`) and every fold in that
regime:

1. `splits.assert_no_leakage` is called and allowed to raise -- this is
   non-negotiable, not a warning.
2. Genes are selected **train-only** for that fold (`data.select_hvgs`, or all
   genes if `gene_set: all`).
3. Every baseline in `perturb_bench.baselines.BASELINES` appropriate to that
   regime (`ridge` is swapped for the explicitly-labeled `ridge_no_drug` variant
   on `held_out_drug`/`held_out_both`, where a one-hot drug encoding is undefined
   for an unseen compound -- SPEC §5) is fit on `train_idx` and scored on
   `test_idx`, on both the `delta` and `absolute` representations.
4. The split-half noise ceiling (`ceiling.compute_ceiling`) is computed on the
   *same* gene subset, both representations.
5. Every metric is normalized against a per-metric floor/ceiling pair (see
   "Floor policy" below) and written as one row per
   (split, fold, baseline, metric, representation) to `results.csv`.

`report.md` is generated from `results.csv` afterward; see "What the report
covers" below.

## Floor policy (binding, see `CONTRACT.md` ADDENDUM 1)

`no_change` predicts an all-zero delta vector, which has zero variance, so
`pearson_r`/`pearson_r_de` are mathematically undefined (NaN) for it on the
`delta` representation -- not 0. Those two metrics are therefore normalized
against `global_mean_delta` instead; `mae`, `direction_acc_de`, and
`pert_discrimination` are normalized against `no_change`. `results.csv` records
which baseline supplied the floor for each metric in the `floor_source` column.
NaN floors/scores are never coerced to 0.

## The `halves` gene-axis subtlety (see `CONTRACT.md` ADDENDUM 2)

`X`/`control`/`delta`/`de_mask` are cached over every post-QC, post-detection-filter
gene. `halves` (used only for the noise ceiling) is cached over a *much smaller*
axis: the union of every fold's train-only HVG selection across all four regimes,
because caching split-half pseudobulk at the full gene width would be many GB to
low-TB (see `DATA_RECON.md`). `run.py`'s `_ceiling_for_fold` translates a fold's
gene selection (full-axis positions) into that smaller axis's local positions by
value lookup (`np.searchsorted` against the sorted union, with an explicit
assertion that every fold gene is actually covered) before calling
`ceiling.compute_ceiling` -- never by assuming the two axes share column order.
See the comment on that function and `tests/test_run.py`'s
`test_ceiling_for_fold_maps_full_axis_genes_to_local_halves_positions` (which uses
a deliberately non-contiguous, non-identity gene union specifically so an
off-by-identity mapping bug cannot pass it by accident).

## What the report covers

`report.md` (SPEC §8/§10) contains, in order: the main normalized-delta table
(baselines x splits, one sub-table per metric); `held_out_context` broken out per
cell line plus the pooled mean, with an explicit caveat that three cell lines is
too few to generalize from; the absolute-vs-delta inflation diagnostic, centered
on `no_change` (NaN `pearson_r` on delta vs. a near-perfect score on absolute for
a baseline that predicts zero drug effect); the `per_drug_mean_delta` (B3) vs.
`ridge` (B4) comparison on `held_out_context`, which is the single number SPEC §10
Q3 hinges on; fallback rates, dropped-condition counts, cell-level QC counts,
batch/plate-matching status, the dataset-global gene-detection filter and its
threshold, and the `halves` gene-union decision; and finally explicit answers to
SPEC §10's four questions, stated plainly including if they are negative.

Every mean in the report carries its std and n (SPEC §11) -- there is no bare
mean anywhere in `report.py`.

## Module map

| file | owns |
|---|---|
| `perturb_bench/config.py` | `Config` dataclass, YAML loading, cache-key hashing |
| `perturb_bench/data.py` | loading, QC, pseudobulk/delta/DE-mask, split-halves, on-disk cache |
| `perturb_bench/splits.py` | the four split regimes, `Fold`, `assert_no_leakage` |
| `perturb_bench/baselines.py` | B1-B5 baselines behind a common `fit`/`predict` interface |
| `perturb_bench/metrics.py` | the metric suite, aggregation (mean/std/n, never a bare mean), normalization |
| `perturb_bench/ceiling.py` | split-half noise ceiling, same metric suite, same gene subset as a baseline |
| `perturb_bench/run.py` | orchestration: the single `python -m perturb_bench.run` entry point |
| `perturb_bench/report.py` | turns `results.csv` into `report.md` |

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

All tests run against small synthetic data -- none require the real ~12GB
sciplex3 download/cache. `tests/test_run.py` additionally verifies: the full
pipeline runs end-to-end and writes the expected `results.csv` columns with no
NaN in structural columns; two runs with the same config+seed produce a
byte-identical `results.csv`; `assert_no_leakage` is called exactly once per
fold (spied, not just plausibly-probably called); `normalized` is ~0 for the
floor baseline and ~1 for the `ceiling` pseudo-baseline row; a deliberately
sabotaged (leaky) fold makes the run raise rather than silently score; and the
`halves` gene-axis translation described above is correct under a
deliberately-adversarial (non-identity) gene union.

## Known limitations (carried over from `DATA_RECON.md` / `CONTRACT.md`, not fixed here)

- Cell-level QC (`min_genes`, `max_pct_mito`) and the dataset-global gene
  detection filter are both dataset-wide preprocessing steps, applied before any
  split exists, using no condition label or effect size -- not leakage, but real
  preprocessing choices whose cell/gene counts are surfaced in `report.md` rather
  than silently applied.
- `time` (24h vs. 72h) is present in sciplex3's obs but not folded into `cond_id`;
  a small 72h subset is pooled with 24h cells of the same (cell_line, compound,
  dose) where both exist. Flagged in `DATA_RECON.md`, not addressed here.
