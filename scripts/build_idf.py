"""Training-only document frequencies, pinned to the frozen data revision."""
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import requests
import numpy as np
import nltk

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z]+")


def word_tokens(text):
    return TOKEN_RE.findall(text.lower())


def prepare_idf(root, config):
    out = root / "data/frozen/idf_reference.json"
    if out.exists():
        result = json.loads(out.read_text())
        assert result["dataset_revision"] == config["source_revision"]
        return result
    counts, n, sources = Counter(), 0, []
    for name in ("esnli_train_1.csv", "esnli_train_2.csv"):
        path = root / "cache/esnli" / config["source_revision"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/{config["source_revision"]}/dataset/{name}'
        if not path.exists():
            print("Downloading training-only reference", name, flush=True)
            r = requests.get(url, stream=True, timeout=(30, 60))
            r.raise_for_status()
            temp = path.with_suffix(".partial")
            with temp.open("wb") as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk)
            temp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                counts.update(set(word_tokens(row["Sentence1"] + " " + row["Sentence2"])))
                n += 1
        sources.append({"url": url, "sha256": digest})
        print("Reference rows processed", n, flush=True)
    values = [math.log((n+1)/(df+1))+1 for df in counts.values()]
    thresholds = np.quantile(values, [1/3, 2/3]).tolist()
    result = {"dataset_revision": config["source_revision"], "reference_split": "train", "n_documents": n,
              "tokenization": "lowercase ASCII alphabetic runs [a-z]+", "document": "premise + hypothesis",
              "formula": "log((N+1)/(df+1))+1", "unseen_df": 0,
              "vocabulary_tertile_thresholds": thresholds, "document_frequencies": dict(counts), "sources": sources}
    out.write_text(json.dumps(result, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    cfg = json.loads((ROOT / "config/experiment.json").read_text())
    reference = prepare_idf(ROOT, cfg)
    nltk_dir = ROOT / "cache/nltk"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    lexicon_path = nltk_dir / "sentiment/vader_lexicon.zip"
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    if not lexicon_path.exists():
        # requests uses the installed certificate bundle; keep TLS verification enabled.
        r = requests.get("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip", timeout=60)
        r.raise_for_status()
        lexicon_path.write_bytes(r.content)
    (ROOT / "data/frozen/sentiment_reference.json").write_text(json.dumps({
        "source":"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip",
        "sha256":hashlib.sha256(lexicon_path.read_bytes()).hexdigest()},indent=2)+"\n")
    print(json.dumps({k:v for k,v in reference.items() if k != "document_frequencies"}), flush=True)
