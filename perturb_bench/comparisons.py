"""Paired, drug-cluster bootstrap on fixed out-of-sample predictions.

Average paired partition scores within conditions, then weight conditions equally
within each context and contexts equally in the pooled statistic. Resample drugs
jointly across contexts. Intervals are conditional on fitted models, the observed
contexts, and (for discrimination) the fixed candidate pool. They do not quantify
training-set, unseen-context, or pathway-level uncertainty.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .metrics import METRIC_DIRECTION

IDENTITY = ["split", "fold", "cond_id", "compound", "dose", "time", "cell_line"]
SCORE_COLUMNS = IDENTITY + ["baseline", "partition"] + list(METRIC_DIRECTION)


def condition_score_rows(raw, obs, split, fold, baseline):
    """Attach identities to partition-major primary delta scores without parsing IDs."""
    if len(obs) == 0 or len(raw) % len(obs):
        raise ValueError("Scores must contain a whole number of condition partitions")
    n_rep = len(raw) // len(obs)
    rows = pd.concat([obs.reset_index(drop=True)] * n_rep, ignore_index=True)
    rows["cond_id"] = np.tile(obs.index.astype(str), n_rep)
    if "time" not in rows:
        rows["time"] = 0.0
    rows["partition"] = np.repeat(np.arange(n_rep), len(obs))
    rows["split"], rows["fold"], rows["baseline"] = split, fold, baseline
    for metric in METRIC_DIRECTION:
        rows[metric] = raw[metric].to_numpy()
    return rows[SCORE_COLUMNS]


def paired_conditions(scores, metric, baseline_a, baseline_b):
    """Pair before dropping nonfinite scores; average only shared finite partitions."""
    keys = IDENTITY + ["partition"]
    sides = []
    for name, column in [(baseline_a, "a"), (baseline_b, "b")]:
        side = scores.loc[scores.baseline == name, keys + [metric]].copy()
        if side[keys].isna().any().any() or side.duplicated(keys).any():
            raise ValueError("Missing identities or duplicate condition/partition scores")
        sides.append(side.rename(columns={metric: column}))
    pairs = sides[0].merge(sides[1], on=keys, how="outer", validate="one_to_one")
    valid = np.isfinite(pairs.a) & np.isfinite(pairs.b)
    total = len(pairs[IDENTITY].drop_duplicates())
    matched = pairs.loc[valid].groupby(IDENTITY, sort=True)[["a", "b"]].mean().reset_index()
    matched["difference"] = matched.a - matched.b
    counts = dict(n_conditions_total=total, n_conditions=len(matched),
                  n_excluded_conditions=total - len(matched),
                  n_partition_pairs=len(pairs), n_usable_partition_pairs=int(valid.sum()),
                  n_excluded_partition_pairs=int((~valid).sum()))
    return matched, counts


def paired_comparison(scores, metric, baseline_a="ridge", baseline_b="per_drug_mean_delta",
                      seed=0, n_boot=2000):
    """Raw A-minus-B difference and percentile 95% drug-cluster interval.

    The point statistic remains condition weighted within context. A sampled drug
    carries all its conditions, so unequal cluster sizes are retained. Missing
    contexts in >5% of bootstrap draws suppress inference rather than silently
    changing the estimand. Fewer than two drugs in any context also suppress it.
    """
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    paired, counts = paired_conditions(scores, metric, baseline_a, baseline_b)
    contexts = sorted(scores.cell_line.unique())
    drugs = sorted(paired.compound.unique())
    result = dict(metric=metric, baseline_a=baseline_a, baseline_b=baseline_b,
                  difference=np.nan, mean_a=np.nan, mean_b=np.nan,
                  ci_low=np.nan, ci_high=np.nan, n_drugs=len(drugs),
                  n_contexts=len(contexts), n_boot=n_boot, seed=seed,
                  valid_bootstrap_fraction=0.0, status="insufficient_data",
                  evidence="inconclusive", **counts)
    if paired.empty or set(paired.cell_line) != set(contexts):
        return result
    means = paired.groupby("cell_line")[["a", "b", "difference"]].mean().mean()
    result.update(mean_a=float(means.a), mean_b=float(means.b), difference=float(means.difference))
    if paired.groupby("cell_line").compound.nunique().min() < 2:
        return result
    # Each row of weights is a bootstrap draw; the same drug multiplicities
    # apply to EVERY context. Aggregated sums avoid copying large dataframes.
    totals = paired.groupby(["compound", "cell_line"]).difference.agg(["sum", "count"])
    sums = totals["sum"].unstack().reindex(index=drugs, columns=contexts).fillna(0).to_numpy()
    sizes = totals["count"].unstack().reindex(index=drugs, columns=contexts).fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(drugs), np.full(len(drugs), 1 / len(drugs)), size=n_boot)
    denominators = np.einsum("bd,dc->bc", weights, sizes, optimize=False)
    valid = (denominators > 0).all(axis=1)
    result["valid_bootstrap_fraction"] = float(valid.mean())
    if valid.mean() < 0.95:
        result["status"] = "insufficient_cluster_coverage"
        return result
    draws = ((np.einsum("bd,dc->bc", weights[valid], sums, optimize=False)) / denominators[valid]).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    higher = METRIC_DIRECTION[metric] == "higher_better"
    # Metric directions are checked here rather than assuming all scores maximize.
    if lo > 0:
        evidence = baseline_a if higher else baseline_b
    elif hi < 0:
        evidence = baseline_b if higher else baseline_a
    else:
        evidence = "inconclusive"
    result.update(ci_low=float(lo), ci_high=float(hi), status="ok", evidence=evidence)
    return result


def build_comparisons(scores, seed=0, n_boot=2000):
    """Only held-out-context B4/B3 inference; other regimes remain descriptive."""
    selected = scores[scores.split == "held_out_context"]
    scopes = [("equal_context_mean", selected)] + [
        (str(context), group) for context, group in selected.groupby("cell_line", sort=True)]
    rows = []
    for scope, group in scopes:
        for metric in METRIC_DIRECTION:
            digest = hashlib.sha256(f"{seed}|{scope}|{metric}".encode()).digest()
            row_seed = int.from_bytes(digest[:4], "little")
            row = paired_comparison(group, metric, seed=row_seed, n_boot=n_boot)
            row.update(split="held_out_context", scope=scope, target="replicate_half",
                       representation="delta", endpoint="primary" if
                       scope == "equal_context_mean" and metric == "pearson_r" else "exploratory")
            rows.append(row)
    return pd.DataFrame(rows)
