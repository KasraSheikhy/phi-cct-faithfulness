"""Download an immutable, unquantized Qwen snapshot; does not generate outputs."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download

if __name__ == "__main__":
    cfg_path = ROOT / "config/experiment.json"
    cfg = json.loads(cfg_path.read_text())
    info = HfApi().model_info(cfg["model_id"], revision=cfg.get("model_revision") or "main", files_metadata=True)
    cfg["model_revision"] = info.sha
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    weight_bytes = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
    state = {"model_id": cfg["model_id"], "revision": info.sha, "weight_bytes": weight_bytes,
             "weight_gib": weight_bytes / 2**30, "status": "downloading", "quantization": False}
    status_path = ROOT / "outputs/model_download.json"
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
    start = time.perf_counter()
    path = snapshot_download(cfg["model_id"], revision=info.sha,
                             allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"], max_workers=2)
    state.update(status="downloaded", snapshot_path=path, elapsed_seconds=time.perf_counter()-start)
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
