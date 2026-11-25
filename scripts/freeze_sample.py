"""Freeze evaluation IDs before model outputs. Preparation helper, to inline in final notebook."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import requests


ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    settings_path = ROOT / "config/experiment.json"
    settings = json.loads(settings_path.read_text())
    destination = ROOT / "data/frozen/frozen_sample.csv"
    manifest_path = ROOT / "data/frozen/sample_manifest.json"
    if destination.exists() or manifest_path.exists():
        if not destination.exists() or not manifest_path.exists():
            raise RuntimeError("Partial freeze exists; inspect before proceeding")
        manifest = json.loads(manifest_path.read_text())
        if sha256(destination.read_bytes()) != manifest["sample_sha256"]:
            raise RuntimeError("Frozen sample checksum changed")
        if manifest["seed"] != settings["sample_seed"] or manifest["n_sampled"] != settings["sample_size"]:
            raise RuntimeError("Configuration differs from frozen sample")
        print(json.dumps({"status": "existing_frozen_sample_verified", **manifest}, indent=2))
        return

    response = requests.get("https://api.github.com/repos/OanaMariaCamburu/e-SNLI/commits/master", timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    source_url = f"https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/{revision}/dataset/esnli_test.csv"
    response = requests.get(source_url, timeout=60)
    response.raise_for_status()
    source_bytes = response.content
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"))))
    required = {"pairID", "gold_label", "Sentence1", "Sentence2"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Unexpected dataset schema: {list(rows[0]) if rows else 'empty'}")
    source_ids = [row["pairID"] for row in rows]
    if len(set(source_ids)) != len(rows):
        raise RuntimeError("Source IDs are not unique; explicit resolution is required")
    labels = {"entailment", "neutral", "contradiction"}
    if any(row["gold_label"] not in labels or not row["Sentence1"].strip() or not row["Sentence2"].strip() for row in rows):
        raise RuntimeError("Invalid source rows; do not silently filter the sampling frame")

    ordered_rows = sorted(enumerate(rows), key=lambda item: item[1]["pairID"])
    selected = random.Random(settings["sample_seed"]).sample(ordered_rows, settings["sample_size"])
    output = io.StringIO(newline="")
    fields = ["source_id", "dataset_revision", "split", "source_row", "premise", "hypothesis", "gold_semantic_label", "sample_seed", "sample_rank"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for rank, (original_row, row) in enumerate(selected):
        writer.writerow(dict(source_id=row["pairID"], dataset_revision=revision, split="test", source_row=original_row,
                             premise=row["Sentence1"], hypothesis=row["Sentence2"], gold_semantic_label=row["gold_label"],
                             sample_seed=settings["sample_seed"], sample_rank=rank))
    sample_bytes = output.getvalue().encode("utf-8")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url, "dataset_revision": revision, "split": "test",
        "source_file_sha256": sha256(source_bytes), "sample_sha256": sha256(sample_bytes),
        "seed": settings["sample_seed"], "sampling_algorithm": "random.Random(seed).sample over source IDs sorted lexicographically",
        "python_version": platform.python_version(), "n_source_rows": len(rows), "n_sampled": len(selected),
        "n_distinct_source_ids": len({row["pairID"] for _, row in selected}),
        "n_distinct_text_pairs": len({(row["Sentence1"], row["Sentence2"]) for _, row in selected}),
        "label_counts": dict(sorted(Counter(row["gold_label"] for _, row in selected).items())),
        "source_row_index_convention": "zero-based CSV data-row index, excluding header",
        "frozen_before_model_outputs": True, "model_responses_generated_at_freeze": 0,
    }
    cached = ROOT / "cache/esnli" / revision / "esnli_test.csv"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(source_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(sample_bytes)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    settings["source_revision"] = revision
    settings["sample_manifest_sha256"] = sha256(manifest_path.read_bytes())
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
