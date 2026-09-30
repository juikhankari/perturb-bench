"""Atomic per-fold records keep summary and condition scores together on resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 1


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=True))
    temporary.replace(path)


def prepare(directory, config, quick, resume):
    directory = Path(directory)
    folder = directory / "fold_checkpoints"
    manifest = folder / "manifest.json"
    expected = dict(schema=SCHEMA, config=config, quick=quick)
    records = []
    if resume and manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError("Resume configuration/schema mismatch; use a new output directory")
        for path in sorted(folder.glob("fold-*.json")):
            record = json.loads(path.read_text())
            if not record.get("results") or not record.get("scores"):
                raise ValueError(f"Incomplete checkpoint: {path}")
            records.append(record)
    elif resume and (directory / "results.csv").exists():
        raise ValueError("Legacy results lack paired-score checkpoints; rerun in a new output directory")
    folder.mkdir(exist_ok=True)
    if not resume:
        for path in folder.glob("fold-*.json"):
            path.unlink()
    atomic_json(manifest, expected)
    return folder, records


def save(folder, split, fold, results, scores):
    digest = hashlib.sha256(f"{split}|{fold}".encode()).hexdigest()
    atomic_json(folder / f"fold-{digest}.json",
                dict(split=split, fold=fold, results=results, scores=scores))
