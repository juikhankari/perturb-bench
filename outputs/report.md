# Perturbation-response benchmark harness -- report

Dataset: `sciplex3` | seed: `0` | gene_set: `hvg` (top 2000 train-only HVGs per fold) | quick mode: `False` | resume: `False`

2424 pseudobulk conditions, 12871 genes (post dataset-global detection filter), 24 fold(s) run, all 24 of them leakage-checked before scoring. Primary scoring target: `replicate_half` (CONTRACT ADDENDUM 3 Decision 1); `full_pseudobulk` retained as a labeled secondary convention below.

This report is generated from results.csv by `perturb_bench/report.py`. It states what the numbers say, including negative results, per the tone requirement in this harness's task spec -- it does not editorialize results upward or bury a result that undercuts the motivating hypothesis. Every mean below carries a std and/or a confidence interval; no winner or "beats" claim is made from a point estimate alone.

---

## Direction guard (ADDENDUM 3 Decision 3)

**The direction guard FIRED on 90/480 (fold, metric, representation, target) combinations in this run.** Every row below had sign(ceiling-floor) contradicting METRIC_DIRECTION for that metric; `normalized` was suppressed to NaN for ALL baselines under that combination (not just the one shown), per ADDENDUM 3 Decision 3. Full detail was also logged to stderr during the run.

| index | split | fold | metric | representation | target | floor_value | ceiling_value |
|---|---|---|---|---|---|---|---|
| 0 | held_out_both | held_out_both/A549__fold0 | mae | absolute | full_pseudobulk | 0.024 | 0.028 |
| 1 | held_out_both | held_out_both/A549__fold0 | mae | delta | full_pseudobulk | 0.024 | 0.028 |
| 2 | held_out_both | held_out_both/A549__fold1 | mae | absolute | full_pseudobulk | 0.023 | 0.029 |
| 3 | held_out_both | held_out_both/A549__fold1 | mae | absolute | replicate_half | 0.028 | 0.029 |
| 4 | held_out_both | held_out_both/A549__fold1 | mae | delta | full_pseudobulk | 0.023 | 0.029 |
| 5 | held_out_both | held_out_both/A549__fold1 | mae | delta | replicate_half | 0.028 | 0.029 |
| 6 | held_out_both | held_out_both/A549__fold1 | pearson_r | absolute | full_pseudobulk | 0.984 | 0.983 |
| 7 | held_out_both | held_out_both/A549__fold2 | mae | absolute | full_pseudobulk | 0.026 | 0.027 |
| 8 | held_out_both | held_out_both/A549__fold2 | mae | delta | full_pseudobulk | 0.026 | 0.027 |
| 9 | held_out_both | held_out_both/A549__fold3 | mae | absolute | full_pseudobulk | 0.020 | 0.028 |
| 10 | held_out_both | held_out_both/A549__fold3 | mae | absolute | replicate_half | 0.025 | 0.028 |
| 11 | held_out_both | held_out_both/A549__fold3 | mae | delta | full_pseudobulk | 0.020 | 0.028 |
| 12 | held_out_both | held_out_both/A549__fold3 | mae | delta | replicate_half | 0.025 | 0.028 |
| 13 | held_out_both | held_out_both/A549__fold3 | pearson_r | absolute | full_pseudobulk | 0.988 | 0.985 |
| 14 | held_out_both | held_out_both/A549__fold4 | mae | absolute | full_pseudobulk | 0.020 | 0.027 |
| 15 | held_out_both | held_out_both/A549__fold4 | mae | absolute | replicate_half | 0.024 | 0.027 |
| 16 | held_out_both | held_out_both/A549__fold4 | mae | delta | full_pseudobulk | 0.020 | 0.027 |
| 17 | held_out_both | held_out_both/A549__fold4 | mae | delta | replicate_half | 0.024 | 0.027 |
| 18 | held_out_both | held_out_both/A549__fold4 | pearson_r | absolute | full_pseudobulk | 0.990 | 0.986 |
| 19 | held_out_both | held_out_both/A549__fold4 | pearson_r | absolute | replicate_half | 0.986 | 0.986 |
| 20 | held_out_both | held_out_both/K562__fold0 | mae | absolute | full_pseudobulk | 0.022 | 0.030 |
| 21 | held_out_both | held_out_both/K562__fold0 | mae | absolute | replicate_half | 0.027 | 0.030 |
| 22 | held_out_both | held_out_both/K562__fold0 | mae | delta | full_pseudobulk | 0.022 | 0.030 |
| 23 | held_out_both | held_out_both/K562__fold0 | mae | delta | replicate_half | 0.027 | 0.030 |
| 24 | held_out_both | held_out_both/K562__fold0 | pearson_r | absolute | full_pseudobulk | 0.985 | 0.979 |
| 25 | held_out_both | held_out_both/K562__fold0 | pearson_r | absolute | replicate_half | 0.979 | 0.979 |
| 26 | held_out_both | held_out_both/K562__fold1 | mae | absolute | full_pseudobulk | 0.022 | 0.031 |
| 27 | held_out_both | held_out_both/K562__fold1 | mae | absolute | replicate_half | 0.027 | 0.031 |
| 28 | held_out_both | held_out_both/K562__fold1 | mae | delta | full_pseudobulk | 0.022 | 0.031 |
| 29 | held_out_both | held_out_both/K562__fold1 | mae | delta | replicate_half | 0.027 | 0.031 |
| 30 | held_out_both | held_out_both/K562__fold1 | pearson_r | absolute | full_pseudobulk | 0.985 | 0.977 |
| 31 | held_out_both | held_out_both/K562__fold1 | pearson_r | absolute | replicate_half | 0.979 | 0.977 |
| 32 | held_out_both | held_out_both/K562__fold2 | mae | absolute | full_pseudobulk | 0.026 | 0.031 |
| 33 | held_out_both | held_out_both/K562__fold2 | mae | absolute | replicate_half | 0.031 | 0.031 |
| 34 | held_out_both | held_out_both/K562__fold2 | mae | delta | full_pseudobulk | 0.026 | 0.031 |
| 35 | held_out_both | held_out_both/K562__fold2 | mae | delta | replicate_half | 0.031 | 0.031 |
| 36 | held_out_both | held_out_both/K562__fold3 | mae | absolute | full_pseudobulk | 0.021 | 0.029 |
| 37 | held_out_both | held_out_both/K562__fold3 | mae | absolute | replicate_half | 0.026 | 0.029 |
| 38 | held_out_both | held_out_both/K562__fold3 | mae | delta | full_pseudobulk | 0.021 | 0.029 |
| 39 | held_out_both | held_out_both/K562__fold3 | mae | delta | replicate_half | 0.026 | 0.029 |
| 40 | held_out_both | held_out_both/K562__fold3 | pearson_r | absolute | full_pseudobulk | 0.985 | 0.979 |
| 41 | held_out_both | held_out_both/K562__fold3 | pearson_r | absolute | replicate_half | 0.980 | 0.979 |
| 42 | held_out_both | held_out_both/K562__fold4 | mae | absolute | full_pseudobulk | 0.020 | 0.028 |
| 43 | held_out_both | held_out_both/K562__fold4 | mae | absolute | replicate_half | 0.024 | 0.028 |
| 44 | held_out_both | held_out_both/K562__fold4 | mae | delta | full_pseudobulk | 0.020 | 0.028 |
| 45 | held_out_both | held_out_both/K562__fold4 | mae | delta | replicate_half | 0.024 | 0.028 |
| 46 | held_out_both | held_out_both/K562__fold4 | pearson_r | absolute | full_pseudobulk | 0.988 | 0.981 |
| 47 | held_out_both | held_out_both/K562__fold4 | pearson_r | absolute | replicate_half | 0.983 | 0.981 |
| 48 | held_out_both | held_out_both/MCF7__fold0 | mae | absolute | full_pseudobulk | 0.016 | 0.018 |
| 49 | held_out_both | held_out_both/MCF7__fold0 | mae | delta | full_pseudobulk | 0.016 | 0.018 |
| 50 | held_out_both | held_out_both/MCF7__fold1 | mae | absolute | full_pseudobulk | 0.017 | 0.019 |
| 51 | held_out_both | held_out_both/MCF7__fold1 | mae | delta | full_pseudobulk | 0.017 | 0.019 |
| 52 | held_out_both | held_out_both/MCF7__fold3 | mae | absolute | full_pseudobulk | 0.016 | 0.018 |
| 53 | held_out_both | held_out_both/MCF7__fold3 | mae | delta | full_pseudobulk | 0.016 | 0.018 |
| 54 | held_out_both | held_out_both/MCF7__fold4 | mae | absolute | full_pseudobulk | 0.014 | 0.017 |
| 55 | held_out_both | held_out_both/MCF7__fold4 | mae | absolute | replicate_half | 0.017 | 0.017 |
| 56 | held_out_both | held_out_both/MCF7__fold4 | mae | delta | full_pseudobulk | 0.014 | 0.017 |
| 57 | held_out_both | held_out_both/MCF7__fold4 | mae | delta | replicate_half | 0.017 | 0.017 |
| 58 | held_out_context | held_out_context/A549 | mae | absolute | full_pseudobulk | 0.023 | 0.028 |
| 59 | held_out_context | held_out_context/A549 | mae | absolute | replicate_half | 0.027 | 0.028 |
| 60 | held_out_context | held_out_context/A549 | mae | delta | full_pseudobulk | 0.023 | 0.028 |
| 61 | held_out_context | held_out_context/A549 | mae | delta | replicate_half | 0.027 | 0.028 |
| 62 | held_out_context | held_out_context/K562 | mae | absolute | full_pseudobulk | 0.022 | 0.030 |
| 63 | held_out_context | held_out_context/K562 | mae | absolute | replicate_half | 0.027 | 0.030 |
| 64 | held_out_context | held_out_context/K562 | mae | delta | full_pseudobulk | 0.022 | 0.030 |
| 65 | held_out_context | held_out_context/K562 | mae | delta | replicate_half | 0.027 | 0.030 |
| 66 | held_out_context | held_out_context/K562 | pearson_r | absolute | full_pseudobulk | 0.984 | 0.979 |
| 67 | held_out_context | held_out_context/MCF7 | mae | absolute | full_pseudobulk | 0.017 | 0.018 |
| 68 | held_out_context | held_out_context/MCF7 | mae | delta | full_pseudobulk | 0.017 | 0.018 |
| 69 | held_out_drug | held_out_drug/fold0 | mae | absolute | full_pseudobulk | 0.026 | 0.029 |
| 70 | held_out_drug | held_out_drug/fold0 | mae | delta | full_pseudobulk | 0.026 | 0.029 |
| 71 | held_out_drug | held_out_drug/fold0 | pearson_r | absolute | full_pseudobulk | 0.985 | 0.984 |
| 72 | held_out_drug | held_out_drug/fold1 | mae | absolute | full_pseudobulk | 0.025 | 0.030 |
| 73 | held_out_drug | held_out_drug/fold1 | mae | absolute | replicate_half | 0.029 | 0.030 |
| 74 | held_out_drug | held_out_drug/fold1 | mae | delta | full_pseudobulk | 0.025 | 0.030 |
| 75 | held_out_drug | held_out_drug/fold1 | mae | delta | replicate_half | 0.029 | 0.030 |
| 76 | held_out_drug | held_out_drug/fold1 | pearson_r | absolute | full_pseudobulk | 0.984 | 0.983 |
| 77 | held_out_drug | held_out_drug/fold3 | mae | absolute | full_pseudobulk | 0.023 | 0.029 |
| 78 | held_out_drug | held_out_drug/fold3 | mae | absolute | replicate_half | 0.027 | 0.029 |
| 79 | held_out_drug | held_out_drug/fold3 | mae | delta | full_pseudobulk | 0.023 | 0.029 |
| 80 | held_out_drug | held_out_drug/fold3 | mae | delta | replicate_half | 0.027 | 0.029 |
| 81 | held_out_drug | held_out_drug/fold3 | pearson_r | absolute | full_pseudobulk | 0.987 | 0.984 |
| 82 | held_out_drug | held_out_drug/fold4 | mae | absolute | full_pseudobulk | 0.021 | 0.028 |
| 83 | held_out_drug | held_out_drug/fold4 | mae | absolute | replicate_half | 0.026 | 0.028 |
| 84 | held_out_drug | held_out_drug/fold4 | mae | delta | full_pseudobulk | 0.021 | 0.028 |
| 85 | held_out_drug | held_out_drug/fold4 | mae | delta | replicate_half | 0.026 | 0.028 |
| 86 | held_out_drug | held_out_drug/fold4 | pearson_r | absolute | full_pseudobulk | 0.989 | 0.986 |
| 87 | held_out_drug | held_out_drug/fold4 | pearson_r | absolute | replicate_half | 0.986 | 0.986 |
| 88 | random | random/holdout | mae | absolute | full_pseudobulk | 0.026 | 0.029 |
| 89 | random | random/holdout | mae | delta | full_pseudobulk | 0.026 | 0.029 |

---

## Main table: normalized delta-representation scores (PRIMARY: replicate-matched target)

Every cell is `normalized ± normalized_std = (score - floor)/(ceiling - floor) ± std/|ceiling-floor|` (ADDENDUM 1/3 floor policy; see the Floor policy section below), pooled across a split's folds weighted by condition count. 0.0 = no better than the metric's floor baseline; 1.0 = at the split-half noise ceiling. These scores use `target=replicate_half` (CONTRACT ADDENDUM 3 Decision 1): both the baseline's prediction and the noise ceiling are scored against the SAME held-out replicate half, so the ceiling is a genuine upper bound for every metric here, with no per-metric analytic correction needed. A cell reading `NaN (guard fired)` means the direction guard (ADDENDUM 3 Decision 3) found sign(ceiling-floor) contradicting this metric's expected direction for that split/baseline and refused to emit a number rather than silently inverting it -- see the Direction guard section below for exactly which (fold, metric) combinations triggered it, if any. These are **delta-representation** scores only -- see the absolute-vs-delta section for why the absolute numbers are excluded here, and see the Secondary scoring section for the legacy `full_pseudobulk`-target numbers.

### pearson_r

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | NaN (n=0) | NaN (n=0) | NaN (n=0) | NaN (n=0) |
| global_mean_delta | 0.000 ± 0.503 (n=4850) | 0.000 ± 0.372 (n=24240) | 0.000 ± 0.424 (n=24240) | 0.000 ± 0.327 (n=24240) |
| per_drug_mean_delta | -0.104 ± 0.464 (n=4850) | 0.084 ± 0.418 (n=24240) | 0.000 ± 0.424 (n=24240) | 0.000 ± 0.327 (n=24240) |
| ridge | 0.948 ± 0.442 (n=4850) | -0.197 ± 0.333 (n=24240) | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.742 ± 0.418 (n=24240) | -0.201 ± 0.312 (n=24240) |
| nearest_context | -0.096 ± 0.488 (n=4850) | 0.072 ± 0.396 (n=24240) | 0.000 ± 0.424 (n=24240) | 0.000 ± 0.327 (n=24240) |

### mae

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | 0.000 ± 13.765 (n=4850) | 0.000 ± 5.889 (n=7440) | 0.000 ± 21.074 (n=10450) | 0.000 ± 542.645 (n=9610) |
| global_mean_delta | -0.175 ± 13.154 (n=4850) | -0.551 ± 5.514 (n=7440) | -0.278 ± 20.230 (n=10450) | -23.206 ± 517.787 (n=9610) |
| per_drug_mean_delta | -5.542 ± 14.424 (n=4850) | -3.744 ± 5.631 (n=7440) | -0.278 ± 20.230 (n=10450) | -23.206 ± 517.787 (n=9610) |
| ridge | 1.874 ± 11.332 (n=4850) | -27.508 ± 3.221 (n=7440) | n/a | n/a |
| ridge_no_drug | n/a | n/a | 1.193 ± 20.416 (n=10450) | -1328.178 ± 2601.512 (n=9610) |
| nearest_context | -7.192 ± 14.808 (n=4850) | -5.485 ± 6.404 (n=7440) | -0.278 ± 20.230 (n=10450) | -23.206 ± 517.787 (n=9610) |

### pearson_r_de

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | NaN (n=0) | NaN (n=0) | NaN (n=0) | NaN (n=0) |
| global_mean_delta | 0.000 ± 1.144 (n=2360) | 0.000 ± 0.865 (n=11500) | 0.000 ± 0.994 (n=11970) | 0.000 ± 0.833 (n=11510) |
| per_drug_mean_delta | -0.113 ± 1.050 (n=2360) | 0.199 ± 0.794 (n=11500) | 0.000 ± 0.994 (n=11970) | 0.000 ± 0.833 (n=11510) |
| ridge | 0.562 ± 1.015 (n=2360) | -0.512 ± 0.996 (n=11500) | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.268 ± 0.919 (n=11970) | -0.591 ± 0.960 (n=11510) |
| nearest_context | -0.124 ± 1.051 (n=2360) | 0.171 ± 0.810 (n=11500) | 0.000 ± 0.994 (n=11970) | 0.000 ± 0.833 (n=11510) |

*Coverage note: pearson_r_de/direction_acc_de are only defined for conditions with >=2 single-cell-significant DE genes; 51.9% of test-condition-fold exposures across this table (4023/7757) are skipped (NaN, not 0) for that reason. The means above are over the surviving, more strongly-responding subset of conditions -- not the full test set -- which is exactly the kind of silent sample bias SPEC §11 warns about; see the dataset-wide zero-DE-gene count in the Fallback/data section below for the dataset-level version of this number.*


### direction_acc_de

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | -2.441 ± 0.004 (n=2360) | -1.924 ± 0.090 (n=11500) | -2.366 ± 0.216 (n=11970) | -1.834 ± 0.328 (n=11510) |
| global_mean_delta | 0.000 ± 0.809 (n=2360) | 0.000 ± 0.687 (n=11500) | 0.000 ± 0.751 (n=11970) | 0.000 ± 0.656 (n=11510) |
| per_drug_mean_delta | -0.065 ± 0.656 (n=2360) | 0.158 ± 0.612 (n=11500) | 0.000 ± 0.751 (n=11970) | 0.000 ± 0.656 (n=11510) |
| ridge | 0.611 ± 0.462 (n=2360) | -0.239 ± 0.729 (n=11500) | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.338 ± 0.595 (n=11970) | -0.331 ± 0.717 (n=11510) |
| nearest_context | -0.116 ± 0.712 (n=2360) | 0.112 ± 0.612 (n=11500) | 0.000 ± 0.751 (n=11970) | 0.000 ± 0.656 (n=11510) |

*Coverage note: pearson_r_de/direction_acc_de are only defined for conditions with >=2 single-cell-significant DE genes; 51.9% of test-condition-fold exposures across this table (4023/7757) are skipped (NaN, not 0) for that reason. The means above are over the surviving, more strongly-responding subset of conditions -- not the full test set -- which is exactly the kind of silent sample bias SPEC §11 warns about; see the dataset-wide zero-DE-gene count in the Fallback/data section below for the dataset-level version of this number.*


### pert_discrimination

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | 0.000 ± 0.654 (n=4850) | 0.000 ± 0.604 (n=24240) | 0.000 ± 0.658 (n=24240) | 0.000 ± 0.609 (n=24240) |
| global_mean_delta | 0.000 ± 0.654 (n=4850) | 0.000 ± 0.604 (n=24240) | 0.000 ± 0.658 (n=24240) | 0.000 ± 0.609 (n=24240) |
| per_drug_mean_delta | 0.109 ± 0.648 (n=4850) | 0.216 ± 0.580 (n=24240) | 0.000 ± 0.658 (n=24240) | 0.000 ± 0.609 (n=24240) |
| ridge | 0.492 ± 0.684 (n=4850) | 0.052 ± 0.606 (n=24240) | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.379 ± 0.690 (n=24240) | 0.006 ± 0.611 (n=24240) |
| nearest_context | 0.209 ± 0.640 (n=4850) | 0.284 ± 0.576 (n=24240) | 0.000 ± 0.658 (n=24240) | 0.000 ± 0.609 (n=24240) |

---

## Secondary scoring: full_pseudobulk target (legacy convention, NOT primary)

These rows score each baseline's prediction against the FULL pseudobulk delta (the convention most published numbers use), rather than against a held-out replicate half. This is exactly the scoring path that silently inverted `mae` normalization in REVIEW.md C-1: the ceiling (half-vs-half) and the baseline (pred-vs-full-pseudobulk) face DIFFERENT target noise levels here, so `normalized` is only shown where the direction guard actually passed -- a `NaN (guard fired)` cell below is the guard doing its job, not missing data. Raw value/std are shown for every cell regardless, so this table is still informative even where normalization is suppressed.

### pearson_r -- raw value ± std (n) [normalized]

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | NaN±NaN (n=0) [NaN] | NaN±NaN (n=0) [NaN] | NaN±NaN (n=0) [NaN] | NaN±NaN (n=0) [NaN] |
| global_mean_delta | 0.155±0.166 (n=485) [0.000±0.559] | 0.089±0.146 (n=2424) [0.000±0.412] | 0.137±0.151 (n=2424) [0.000±0.477] | 0.079±0.136 (n=2424) [0.000±0.366] |
| per_drug_mean_delta | 0.117±0.155 (n=485) [-0.128±0.520] | 0.123±0.160 (n=2424) [0.098±0.456] | 0.137±0.151 (n=2424) [0.000±0.477] | 0.079±0.136 (n=2424) [0.000±0.366] |
| ridge | 0.509±0.136 (n=485) [1.187±0.457] | 0.007±0.134 (n=2424) [-0.227±0.377] | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.427±0.148 (n=2424) [0.939±0.495] | -0.007±0.125 (n=2424) [-0.232±0.355] |
| nearest_context | 0.120±0.164 (n=485) [-0.118±0.549] | 0.118±0.154 (n=2424) [0.083±0.432] | 0.137±0.151 (n=2424) [0.000±0.477] | 0.079±0.136 (n=2424) [0.000±0.366] |
| ceiling | 0.453±0.228 (n=4850) [1.000±0.763] | 0.438±0.217 (n=24240) [1.000±0.598] | 0.448±0.221 (n=24240) [1.000±0.695] | 0.438±0.216 (n=24240) [1.000±0.562] |

### mae -- raw value ± std (n) [normalized]

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | 0.026±0.016 (n=485) [NaN (guard fired)] | 0.021±0.012 (n=2424) [NaN (guard fired)] | 0.025±0.014 (n=2424) [0.000±14.601] | 0.021±0.011 (n=2424) [0.000±3.892] |
| global_mean_delta | 0.026±0.016 (n=485) [NaN (guard fired)] | 0.022±0.011 (n=2424) [NaN (guard fired)] | 0.025±0.014 (n=2424) [-0.009±14.256] | 0.022±0.011 (n=2424) [-0.157±3.745] |
| per_drug_mean_delta | 0.033±0.017 (n=485) [NaN (guard fired)] | 0.028±0.013 (n=2424) [NaN (guard fired)] | 0.025±0.014 (n=2424) [-0.009±14.256] | 0.022±0.011 (n=2424) [-0.157±3.745] |
| ridge | 0.023±0.013 (n=485) [NaN (guard fired)] | 0.072±0.008 (n=2424) [NaN (guard fired)] | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.023±0.014 (n=2424) [1.605±14.328] | 0.074±0.011 (n=2424) [-12.207±2.147] |
| nearest_context | 0.035±0.018 (n=485) [NaN (guard fired)] | 0.032±0.015 (n=2424) [NaN (guard fired)] | 0.025±0.014 (n=2424) [-0.009±14.256] | 0.022±0.011 (n=2424) [-0.157±3.745] |
| ceiling | 0.029±0.008 (n=4850) [NaN (guard fired)] | 0.025±0.007 (n=24240) [NaN (guard fired)] | 0.029±0.008 (n=24240) [1.000±5.633] | 0.025±0.007 (n=24240) [1.000±0.414] |

### pearson_r_de -- raw value ± std (n) [normalized]

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | NaN±NaN (n=0) [NaN] | NaN±NaN (n=0) [NaN] | NaN±NaN (n=0) [NaN] | NaN±NaN (n=0) [NaN] |
| global_mean_delta | 0.392±0.492 (n=236) [0.000±1.180] | 0.280±0.486 (n=1150) [0.000±0.881] | 0.373±0.466 (n=1197) [0.000±1.025] | 0.259±0.482 (n=1151) [0.000±0.846] |
| per_drug_mean_delta | 0.345±0.446 (n=236) [-0.111±1.068] | 0.400±0.426 (n=1150) [0.212±0.799] | 0.373±0.466 (n=1197) [0.000±1.025] | 0.259±0.482 (n=1151) [0.000±0.846] |
| ridge | 0.634±0.436 (n=236) [0.581±1.044] | 0.011±0.512 (n=1150) [-0.529±1.026] | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.502±0.424 (n=1197) [0.279±0.952] | -0.054±0.494 (n=1151) [-0.616±0.993] |
| nearest_context | 0.329±0.451 (n=236) [-0.150±1.082] | 0.380±0.445 (n=1150) [0.175±0.832] | 0.373±0.466 (n=1197) [0.000±1.025] | 0.259±0.482 (n=1151) [0.000±0.846] |
| ceiling | 0.809±0.379 (n=2360) [1.000±0.909] | 0.817±0.341 (n=11500) [1.000±0.619] | 0.820±0.350 (n=11970) [1.000±0.782] | 0.815±0.344 (n=11510) [1.000±0.622] |

### direction_acc_de -- raw value ± std (n) [normalized]

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | 0.000±0.000 (n=236) [-2.445±0.000] | 0.000±0.000 (n=1150) [-1.925±0.091] | 0.000±0.000 (n=1197) [-2.368±0.217] | 0.000±0.000 (n=1151) [-1.834±0.330] |
| global_mean_delta | 0.705±0.234 (n=236) [0.000±0.811] | 0.651±0.233 (n=1150) [0.000±0.688] | 0.696±0.225 (n=1197) [0.000±0.753] | 0.636±0.234 (n=1151) [0.000±0.660] |
| per_drug_mean_delta | 0.687±0.190 (n=236) [-0.063±0.658] | 0.706±0.206 (n=1150) [0.160±0.613] | 0.696±0.225 (n=1197) [0.000±0.753] | 0.636±0.234 (n=1151) [0.000±0.660] |
| ridge | 0.882±0.133 (n=236) [0.615±0.463] | 0.573±0.242 (n=1150) [-0.237±0.731] | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.799±0.174 (n=1197) [0.342±0.594] | 0.530±0.232 (n=1151) [-0.329±0.721] |
| nearest_context | 0.672±0.206 (n=236) [-0.115±0.714] | 0.691±0.206 (n=1150) [0.114±0.612] | 0.696±0.225 (n=1197) [0.000±0.753] | 0.636±0.234 (n=1151) [0.000±0.660] |
| ceiling | 0.993±0.024 (n=2360) [1.000±0.082] | 0.990±0.030 (n=11500) [1.000±0.086] | 0.992±0.026 (n=11970) [1.000±0.087] | 0.990±0.030 (n=11510) [1.000±0.079] |

### pert_discrimination -- raw value ± std (n) [normalized]

| baseline | random | held_out_context | held_out_drug | held_out_both |
|---|---|---|---|---|
| no_change | 0.500±0.290 (n=485) [0.000±0.655] | 0.500±0.289 (n=2424) [0.000±0.604] | 0.500±0.290 (n=2424) [0.000±0.659] | 0.500±0.291 (n=2424) [0.000±0.611] |
| global_mean_delta | 0.500±0.290 (n=485) [-0.000±0.655] | 0.500±0.289 (n=2424) [0.000±0.604] | 0.500±0.290 (n=2424) [0.000±0.659] | 0.500±0.291 (n=2424) [0.000±0.611] |
| per_drug_mean_delta | 0.443±0.287 (n=485) [0.128±0.649] | 0.382±0.280 (n=2424) [0.245±0.585] | 0.500±0.290 (n=2424) [0.000±0.659] | 0.500±0.291 (n=2424) [0.000±0.611] |
| ridge | 0.216±0.326 (n=485) [0.642±0.738] | 0.473±0.291 (n=2424) [0.058±0.607] | n/a | n/a |
| ridge_no_drug | n/a | n/a | 0.276±0.332 (n=2424) [0.512±0.756] | 0.497±0.292 (n=2424) [0.006±0.613] |
| nearest_context | 0.392±0.284 (n=485) [0.245±0.643] | 0.347±0.277 (n=2424) [0.318±0.578] | 0.500±0.290 (n=2424) [0.000±0.659] | 0.500±0.291 (n=2424) [0.000±0.611] |
| ceiling | 0.058±0.092 (n=4850) [1.000±0.208] | 0.021±0.063 (n=24240) [1.000±0.132] | 0.060±0.095 (n=24240) [1.000±0.216] | 0.022±0.065 (n=24240) [1.000±0.140] |

---

## held_out_context, broken out per cell line

SPEC §4: held_out_context is iterated once per cell line (one fold = one held-out cell line). There are only **3** cell lines in sciplex3 (A549, K562, MCF7); three contexts is too few to draw strong general conclusions about context-transfer from -- a single unusual cell line can dominate the mean. Per-cell-line numbers are shown below precisely so that risk is visible rather than hidden inside an averaged headline number. Scores are `target=replicate_half` (primary).

### pearson_r (normalized ± std)

| baseline | held_out_context/A549 | held_out_context/K562 | held_out_context/MCF7 | mean (pooled) |
|---|---|---|---|---|
| no_change | NaN | NaN | NaN | NaN ± NaN (n=0) |
| global_mean_delta | 0.000±0.387 | 0.000±0.374 | 0.000±0.349 | 0.000 ± 0.372 (n=24240) |
| per_drug_mean_delta | 0.100±0.431 | 0.110±0.391 | 0.039±0.425 | 0.084 ± 0.418 (n=24240) |
| ridge | -0.184±0.147 | -0.077±0.229 | -0.334±0.498 | -0.197 ± 0.333 (n=24240) |
| nearest_context | 0.103±0.433 | 0.072±0.337 | 0.033±0.398 | 0.072 ± 0.396 (n=24240) |

### mae (normalized ± std)

| baseline | held_out_context/A549 | held_out_context/K562 | held_out_context/MCF7 | mean (pooled) |
|---|---|---|---|---|
| no_change | NaN | NaN | -0.000±5.889 | 0.000 ± 5.889 (n=7440) |
| global_mean_delta | NaN | NaN | -0.551±5.514 | -0.551 ± 5.514 (n=7440) |
| per_drug_mean_delta | NaN | NaN | -3.744±5.631 | -3.744 ± 5.631 (n=7440) |
| ridge | NaN | NaN | -27.508±3.221 | -27.508 ± 3.221 (n=7440) |
| nearest_context | NaN | NaN | -5.485±6.404 | -5.485 ± 6.404 (n=7440) |

### pearson_r_de (normalized ± std)

| baseline | held_out_context/A549 | held_out_context/K562 | held_out_context/MCF7 | mean (pooled) |
|---|---|---|---|---|
| no_change | NaN | NaN | NaN | NaN ± NaN (n=0) |
| global_mean_delta | 0.000±0.854 | 0.000±0.871 | 0.000±0.874 | 0.000 ± 0.865 (n=11500) |
| per_drug_mean_delta | 0.248±0.730 | 0.331±0.808 | 0.067±0.827 | 0.199 ± 0.794 (n=11500) |
| ridge | -0.456±0.807 | -0.114±0.829 | -0.819±1.157 | -0.512 ± 0.996 (n=11500) |
| nearest_context | 0.278±0.723 | 0.209±0.849 | 0.035±0.850 | 0.171 ± 0.810 (n=11500) |

### direction_acc_de (normalized ± std)

| baseline | held_out_context/A549 | held_out_context/K562 | held_out_context/MCF7 | mean (pooled) |
|---|---|---|---|---|
| no_change | -1.817±0.003 | -1.955±0.035 | -2.016±0.001 | -1.924 ± 0.090 (n=11500) |
| global_mean_delta | 0.000±0.664 | 0.000±0.762 | 0.000±0.659 | 0.000 ± 0.687 (n=11500) |
| per_drug_mean_delta | 0.220±0.538 | 0.216±0.627 | 0.056±0.660 | 0.158 ± 0.612 (n=11500) |
| ridge | -0.023±0.599 | -0.162±0.630 | -0.511±0.818 | -0.239 ± 0.729 (n=11500) |
| nearest_context | 0.203±0.560 | 0.079±0.655 | 0.038±0.622 | 0.112 ± 0.612 (n=11500) |

### pert_discrimination (normalized ± std)

| baseline | held_out_context/A549 | held_out_context/K562 | held_out_context/MCF7 | mean (pooled) |
|---|---|---|---|---|
| no_change | -0.000±0.611 | -0.000±0.611 | -0.000±0.587 | 0.000 ± 0.604 (n=24240) |
| global_mean_delta | -0.000±0.611 | 0.000±0.611 | -0.000±0.587 | 0.000 ± 0.604 (n=24240) |
| per_drug_mean_delta | 0.199±0.582 | 0.217±0.593 | 0.235±0.566 | 0.216 ± 0.580 (n=24240) |
| ridge | 0.035±0.611 | 0.095±0.615 | 0.029±0.586 | 0.052 ± 0.606 (n=24240) |
| nearest_context | 0.253±0.570 | 0.253±0.604 | 0.354±0.547 | 0.284 ± 0.576 (n=24240) |

---

## Absolute-vs-delta comparison: the inflation diagnostic

SPEC §7/§11: scoring on absolute post-treatment expression instead of delta is a known trap -- absolute expression is dominated by baseline (housekeeping) expression level, which every baseline reproduces almost perfectly simply by copying the control profile forward, regardless of whether it predicted the drug effect correctly. The absolute numbers below are reported **only** to name and quantify that inflation; they are never used in the main table or treated as performance. Shown for `target=replicate_half` (primary).

**`no_change` is the centerpiece of this diagnostic.** no_change predicts an all-zero delta vector. That vector has exactly zero variance across genes, so `pearson_r` (and `pearson_r_de`) on the **delta** representation is mathematically undefined -- NaN, not 0 -- for every no_change row. On the **absolute** representation, no_change's prediction is just `control + 0 = control`, which is usually extremely close to the true post-treatment profile purely because most genes barely move and baseline expression dominates total variance. That produces a near-perfect pearson_r on absolute for a model that predicted literally nothing about the drug's effect. The contrast between those two numbers *is* the finding.

| split | representation | pearson_r mean | std | n |
|---|---|---|---|---|
| held_out_both | delta | NaN | NaN | 0 |
| held_out_both | absolute | 0.982 | 0.025 | 24240 |
| held_out_context | delta | NaN | NaN | 0 |
| held_out_context | absolute | 0.982 | 0.027 | 24240 |
| held_out_drug | delta | NaN | NaN | 0 |
| held_out_drug | absolute | 0.978 | 0.032 | 24240 |
| random | delta | NaN | NaN | 0 |
| random | absolute | 0.976 | 0.039 | 4850 |

As predicted: no_change's pearson_r is NaN on every delta row above and positive (up to 0.982) on absolute. Any report that scored this harness on absolute expression and called it performance would have handed a zero-information baseline a near-perfect score.

---

## Normalized `mae` is numerically unstable on this dataset -- do not quote it
`normalized = (score - floor) / (ceiling - floor)`. For `mae` on this dataset the floor and the ceiling very nearly coincide: typical mae value is ~0.0276 while the median |ceiling - floor| is only ~0.0019, a ratio of about 14:1. Dividing by a denominator that small amplifies ordinary condition-to-condition noise by roughly 14x, which is why some normalized mae cells in the tables above are in the hundreds or thousands with standard deviations larger still. **Those magnitudes are amplified noise, not effect sizes, and no normalized mae number in this report should be quoted or compared.** The raw mae values (secondary table) are well-behaved and are the ones to use.

Worse, the sign is not even consistent: `mae` is lower-is-better, so a valid ceiling must have LOWER mae than the floor (negative denominator), but 14 of 24 folds come out positive. The direction guard (ADDENDUM 3 Decision 3) caught these and emitted NaN rather than an inverted number -- 432 mae rows across the run. Even in the folds where the guard passes, the denominator is small enough that the resulting scale is unreliable.

**Why this is a finding rather than a defect.** It says that on sciplex3, predicting no change at all is almost exactly as good, in mean-absolute-error terms, as a real replicate experiment. The drug effect is small relative to measurement noise for most conditions (consistent with the ~41% of conditions having zero statistically detectable DE genes), so mae cannot separate a good model from a null one here. That independently reproduces the Arc Virtual Cell Challenge result that nearly every submission scored worse than a naive baseline on mae, and it is the concrete reason this harness reports five metrics instead of leaning on error magnitude alone.

Per-fold mae floor/ceiling and denominator:

| index | fold | ceiling | floor | denominator |
|---|---|---|---|---|
| 0 | held_out_both/A549__fold0 | 0.027933 | 0.027940 | -0.000007 |
| 1 | held_out_both/A549__fold1 | 0.029470 | 0.027722 | 0.001748 |
| 2 | held_out_both/A549__fold2 | 0.027388 | 0.030142 | -0.002754 |
| 3 | held_out_both/A549__fold3 | 0.027792 | 0.024904 | 0.002888 |
| 4 | held_out_both/A549__fold4 | 0.027205 | 0.024150 | 0.003055 |
| 5 | held_out_both/K562__fold0 | 0.030081 | 0.027012 | 0.003069 |
| 6 | held_out_both/K562__fold1 | 0.030951 | 0.026968 | 0.003983 |
| 7 | held_out_both/K562__fold2 | 0.031118 | 0.030853 | 0.000265 |
| 8 | held_out_both/K562__fold3 | 0.029191 | 0.025564 | 0.003627 |
| 9 | held_out_both/K562__fold4 | 0.028313 | 0.024347 | 0.003965 |
| 10 | held_out_both/MCF7__fold0 | 0.018073 | 0.018925 | -0.000852 |
| 11 | held_out_both/MCF7__fold1 | 0.018643 | 0.019698 | -0.001055 |
| 12 | held_out_both/MCF7__fold2 | 0.017558 | 0.024784 | -0.007226 |
| 13 | held_out_both/MCF7__fold3 | 0.017975 | 0.018311 | -0.000336 |
| 14 | held_out_both/MCF7__fold4 | 0.017351 | 0.016943 | 0.000408 |
| 15 | held_out_context/A549 | 0.027838 | 0.027176 | 0.000662 |
| 16 | held_out_context/K562 | 0.029825 | 0.026981 | 0.002844 |
| 17 | held_out_context/MCF7 | 0.017828 | 0.019917 | -0.002089 |
| 18 | held_out_drug/fold0 | 0.029439 | 0.029786 | -0.000347 |
| 19 | held_out_drug/fold1 | 0.030242 | 0.029309 | 0.000933 |
| 20 | held_out_drug/fold2 | 0.029093 | 0.034689 | -0.005596 |
| 21 | held_out_drug/fold3 | 0.028751 | 0.026993 | 0.001758 |
| 22 | held_out_drug/fold4 | 0.027908 | 0.025538 | 0.002370 |
| 23 | random/holdout | 0.028607 | 0.029793 | -0.001186 |

---

## B3 (per_drug_mean_delta) vs B4 (ridge) on held_out_context

This is the comparison SPEC §10 Q3 hinges on. B3 knows the drug and ignores context entirely (same predicted delta for a drug regardless of cell line). B4 (ridge) is the first baseline that can actually use the held-out cell line's own control expression as an input. If ridge does not beat per_drug_mean_delta here, context does not modulate the transcriptional response in a way this baseline suite can learn linearly from control expression alone. A "winner" is only declared when the two baselines' 95% confidence intervals on the raw metric do NOT overlap (M-2) -- a point-estimate gap alone is never treated as evidence here. For a single fold the interval is the exact bootstrap `ci_low`/`ci_high` written by run.py; pooled across folds it is a normal approximation from the pooled mean/std/n (see `_normal_ci` in report.py) since exact per-condition values aren't carried through the pooling step.

| metric | per_drug_mean_delta (mean±std, n) | ridge (mean±std, n) | gap (ridge - B3) | 95% CIs overlap? | winner |
|---|---|---|---|---|---|
| pearson_r | 0.110±0.151, 24240 | 0.008±0.122, 24240 | -0.102 | no | per_drug_mean_delta |
| mae | 0.032±0.013, 24240 | 0.073±0.007, 24240 | 0.042 | no | per_drug_mean_delta |
| pearson_r_de | 0.390±0.426, 11500 | 0.013±0.504, 11500 | -0.377 | no | per_drug_mean_delta |
| direction_acc_de | 0.705±0.206, 11500 | 0.572±0.241, 11500 | -0.133 | no | per_drug_mean_delta |
| pert_discrimination | 0.397±0.278, 24240 | 0.475±0.290, 24240 | 0.079 | no | per_drug_mean_delta |

**per_drug_mean_delta beats ridge, with non-overlapping 95% CIs, on at least one metric above and ridge wins none.** Per SPEC §10: if the answer to Q3 is no, that is a real finding to report plainly, not a result to work around. Knowing only the drug's identity and ignoring the held-out cell line's control expression entirely (B3) does BETTER than a linear model that is given that control expression (B4). That does not prove context-dependence is unlearnable in general -- it is evidence against it being *linearly* learnable from control expression alone, on this dataset, with this feature set. The broader research programme this harness was built to support should treat that as a real negative result, not as motivation to quietly swap in a fancier featurization until the number moves.

---

## Fallback rates, dropped conditions, batch metadata, gene filtering

### Baseline fallback rates (mean fraction of test conditions per split)

| index | held_out_both | held_out_context | held_out_drug | random |
|---|---|---|---|---|
| global_mean_delta | 0.000 | 0.000 | 0.000 | 0.000 |
| nearest_context | 1.000 | 0.000 | 1.000 | 0.000 |
| no_change | 0.000 | 0.000 | 0.000 | 0.000 |
| per_drug_mean_delta | 1.000 | 0.000 | 1.000 | 0.000 |
| ridge | NaN | 0.000 | NaN | 0.000 |
| ridge_no_drug | 0.000 | NaN | 0.000 | NaN |

A nonzero fallback_rate for per_drug_mean_delta/ridge_no_drug/nearest_context on held_out_drug or held_out_both means the test compound was never seen in training and that baseline fell back to a drug-agnostic prediction (global_mean_delta, for per_drug_mean_delta and nearest_context) -- this is expected and correct behavior for those regimes, not a bug, but it means those cells are scoring the fallback baseline, not the baseline's "normal" behavior.

### Dose-collapse fallback (M-3): per_drug_mean_delta falling back from exact (compound, dose) to compound-any-dose

SPEC §11 is explicit that dose is part of condition identity, not a nuisance variable. `per_drug_mean_delta` (B3) silently falls back through THREE tiers: (0) exact compound+dose match, (1) same compound, ANY dose, averaged -- i.e. dose is collapsed, and (2) no compound match at all -> global mean (the `fallback_rate` column above covers only tier 2). The table below reports tier-1 (dose-collapsed) as its own rate, per the fix plan, rather than leaving it invisible.

| index | held_out_both | held_out_context | held_out_drug | random |
|---|---|---|---|---|
| per_drug_mean_delta | 0.000 | 0.002 | 0.000 | 0.043 |

**Blocker (cannot fix in this module):** `nearest_context` (B5) has the identical kind of dose-collapse internally (it falls back from a dose-matched delta on the nearest training cell line to a compound-only match on that same cell line before falling back further to the global mean), but `baselines.py::NearestContext` does not expose a per-tier counter the way `PerDrugMeanDelta.fallback_tier` does -- only the overall `fallback_mask` (tier-2-equivalent) is available. Reporting B5's dose-collapse rate honestly requires adding a `fallback_tier`-style attribute to `NearestContext` in baselines.py, which is out of scope for this module (baselines.py belongs to another agent per the task boundaries). Flagging this rather than reimplementing NearestContext's matching logic here, which would silently drift out of sync with the real implementation.

### Dropped conditions / QC

- 21 condition(s) dropped for having fewer than 30 cells; 2424 conditions remain in the final pseudobulk.

- 0 condition(s) had no matched (cell_line, plate) control group and fell back to a per-cell_line-pooled control instead.

- **990/2424 conditions (40.8%) have ZERO genes passing the single-cell DE test** (Wilcoxon/Mann-Whitney, BH-FDR < 0.05). pearson_r_de and direction_acc_de are NaN (skipped, not coerced to 0) for those conditions. A mean computed over the surviving conditions is therefore a mean over the strongest-responding subset of the data, not the full test set -- treat those two metrics' headline numbers with that selection bias in mind (SPEC §11).

### Cell-level QC

- 5447/799317 cells (0.7%) dropped by cell-level QC; 793870 remain. min_genes=200, max_pct_mito=50.0.

- 779 cell(s) dropped for too few detected genes.

- 4668 cell(s) dropped for %mitochondrial counts above 50.0%. This threshold is reported as configured, not re-justified here against the dataset's actual mito-percent distribution -- if that threshold cuts through the bulk of the distribution rather than an outlier tail, the dropped-cell count above is the place to check that, and a fixed QC cutoff that drops cells at different rates across doses/conditions is a preprocessing-induced confound worth checking before trusting small effect sizes.

### Batch/plate metadata

- **Real batch metadata IS present** for this dataset: the `plate` column. Treated conditions' deltas are computed against controls matched within the same (cell_line, plate) group where possible -- see DATA_RECON.md / the other agent's data.py notes for any per-plate-purity caveats, which this module does not re-derive.

### Dataset-global gene detection filter

- Applied: genes must be detected (count>0) in at least 0.01 of all cells to be retained, shrinking 110983 -> 12871 genes. This is a dataset-global presence/absence filter computed before any train/test split exists and using no condition label or effect size -- the same class of operation as the mitochondrial-content QC filter, not HVG selection and not leakage. Driven by disk/RAM limits on the machine this harness ran on, documented in DATA_RECON.md.

### `halves` gene-union caching (ADDENDUM 2)

- halves cached only over the union of per-fold, train-only HVG-selected genes across every split regime -- not all genes. See module docstring. Not leakage: each fold scores only on its own train-selected genes. (halves gene axis width: 3015). The noise ceiling in this report is computed on exactly the same gene subset a baseline is scored on for that fold, and now on exactly the same TEST ROWS too (REVIEW.md C-2) -- never on a gene axis, or a condition pool, a fold's own held-out rows helped choose or that includes conditions the baseline was not scored against.

---

## SPEC §10 answers

### Q1: Does any baseline beat no_change on delta metrics, and by how much relative to the noise ceiling?

"Beats" below requires the baseline's and no_change's 95% CIs on the raw metric to NOT overlap (M-2) -- a better point estimate alone does not qualify.

At least one baseline beats no_change (non-overlapping 95% CI) on at least one delta metric:

- global_mean_delta on direction_acc_de: 0.664 vs no_change 0.000 (CIs do not overlap); normalized (floor-to-ceiling) = 0.000 ± 0.707
- nearest_context on direction_acc_de: 0.674 vs no_change 0.000 (CIs do not overlap); normalized (floor-to-ceiling) = 0.027 ± 0.681
- per_drug_mean_delta on direction_acc_de: 0.680 vs no_change 0.000 (CIs do not overlap); normalized (floor-to-ceiling) = 0.044 ± 0.680
- ridge_no_drug on direction_acc_de: 0.666 vs no_change 0.000 (CIs do not overlap); normalized (floor-to-ceiling) = 0.010 ± 0.737
- ridge on direction_acc_de: 0.625 vs no_change 0.000 (CIs do not overlap); normalized (floor-to-ceiling) = -0.095 ± 0.761
- nearest_context on pert_discrimination: 0.452 vs no_change 0.500 (CIs do not overlap); normalized (floor-to-ceiling) = 0.102 ± 0.631
- per_drug_mean_delta on pert_discrimination: 0.465 vs no_change 0.500 (CIs do not overlap); normalized (floor-to-ceiling) = 0.074 ± 0.627
- ridge_no_drug on pert_discrimination: 0.416 vs no_change 0.500 (CIs do not overlap); normalized (floor-to-ceiling) = 0.193 ± 0.678
- ridge on pert_discrimination: 0.443 vs no_change 0.500 (CIs do not overlap); normalized (floor-to-ceiling) = 0.125 ± 0.641

### Q2: How much worse is held_out_context than random? (size of the generalization problem)

- pearson_r: best normalized score on random = 0.948, on held_out_context = 0.084 (drop = 0.864).

- mae: best normalized score on random = 1.874, on held_out_context = 0.000 (drop = 1.874).

- pearson_r_de: best normalized score on random = 0.562, on held_out_context = 0.199 (drop = 0.363).

- direction_acc_de: best normalized score on random = 0.611, on held_out_context = 0.158 (drop = 0.453).

- pert_discrimination: best normalized score on random = 0.492, on held_out_context = 0.284 (drop = 0.208).

### Q3: Does ridge beat per_drug_mean_delta on held_out_context? (is context-dependence linearly learnable?)

See the dedicated "B3 vs B4" section above for the full per-metric breakdown, CIs, and numbers.

### Q4: How much does scoring on absolute expression inflate the numbers?

- no_change's pearson_r is NaN on delta (undefined, zero-variance prediction) and 0.980 (std 0.029, n=77570) on absolute, for a baseline that predicts zero drug effect. See the absolute-vs-delta section for the full per-split breakdown; that gap is the inflation this harness exists to name.
