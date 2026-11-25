"""Read-only live progress monitor; listens exclusively on the Mac's loopback interface."""
import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


class Progress:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.checkpoint = None
        self.offset = 0
        self.pending = b""
        self.reset()
        self.last_manifest = {}
        with (root / "data/frozen/frozen_sample.csv").open() as file:
            self.source_ids = {row["source_id"] for row in csv.DictReader(file)}
        excluded = set()
        for name in ["quality_exclusions.csv", "edit_audit.csv"]:
            path = root / "data/audit" / name
            if path.exists():
                with path.open() as file:
                    for row in csv.DictReader(file):
                        if name == "quality_exclusions.csv" or row.get("decision") == "reject":
                            excluded.add(row["edit_id"])
        counts = Counter()
        with (root / "data/edits/selected_edits.csv").open() as file:
            for row in csv.DictReader(file):
                if row["edit_id"] not in excluded:
                    counts[row["source_id"]] += 1
        self.expected_by_source = {sid: 3 * (1 + counts[sid]) for sid in self.source_ids}

    def reset(self):
        self.count = self.valid = self.generation_errors = self.parse_errors = 0
        self.seconds = self.tokens = 0
        self.recent = deque(maxlen=100)
        self.samples = Counter()
        self.families = Counter()
        self.last_at = None
        self.seen = set()

    def read_json(self, path, fallback=None):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return fallback

    def update_checkpoint(self, path):
        if not path.exists():
            return
        size = path.stat().st_size
        if path != self.checkpoint or size < self.offset:
            self.reset()
            self.checkpoint, self.offset, self.pending = path, 0, b""
        with path.open("rb") as file:
            file.seek(self.offset)
            chunk = file.read()
            self.offset = file.tell()
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if row["job_id"] in self.seen:
                continue
            self.seen.add(row["job_id"])
            self.count += 1
            self.valid += row.get("parse_status") == "ok"
            self.generation_errors += row.get("generation_status") != "ok"
            self.parse_errors += row.get("generation_status") == "ok" and row.get("parse_status") != "ok"
            self.seconds += row.get("elapsed_seconds", 0)
            self.recent.append(row.get("elapsed_seconds", 0))
            self.tokens += row.get("generated_tokens", 0)
            self.samples[row["source_id"]] += 1
            self.families[row.get("family", "unknown")] += 1
            self.last_at = row.get("generated_at_utc", self.last_at)

    def snapshot(self):
        with self.lock:
            now = datetime.now(timezone.utc)
            manifest = self.read_json(self.root / "outputs/full_run_manifest.json", self.last_manifest)
            self.last_manifest = manifest
            if not manifest:
                return {"status": "not_started", "completed": 0, "expected": 9000, "now": now.isoformat()}
            checkpoint = (self.root / manifest["checkpoint"]).resolve()
            if self.root.resolve() not in checkpoint.parents:
                raise ValueError("Unexpected checkpoint path")
            self.update_checkpoint(checkpoint)
            expected = manifest["expected_jobs"]
            remaining = max(0, expected - self.count)
            mean = self.seconds / self.count if self.count else None
            recent_mean = sum(self.recent) / len(self.recent) if self.recent else None
            smoke = self.read_json(self.root / "outputs/smoke_run_manifest.json", {})
            smoke_mean = smoke.get("mean_response_seconds") if smoke.get("config_hash") == manifest.get("config_hash") else None
            started = parse_time(manifest.get("started_at_utc"))
            elapsed = max(0, (now - started).total_seconds()) if started else None
            last = parse_time(self.last_at)
            silence = max(0, (now - last).total_seconds()) if last else None
            stale = manifest["status"] == "running" and silence is not None and silence > max(120, (recent_mean or 0) * 8)
            active = manifest["status"] in {"loading", "running"}
            eta_seconds = remaining * (recent_mean or mean or smoke_mean or 0) if active else None
            cautious_seconds = remaining * max(x for x in [mean, recent_mean, smoke_mean, 0] if x is not None) * 1.25 if active else None
            post = self.read_json(self.root / "outputs/postprocess_status.json", {})
            analysis = self.read_json(self.root / "outputs/results/analysis_manifest.json", {})
            analysis_started = (self.root / "outputs/full_analysis.log").exists()
            pdf_ready = (self.root / "report/phi_cct_report.pdf").exists()
            deadline = parse_time(self.read_json(self.root / "config/experiment.json", {}).get("deadline"))
            return {
                "status": manifest["status"], "stale": stale, "completed": self.count, "expected": expected,
                "remaining": remaining, "fraction": self.count / expected if expected else 0,
                "valid": self.valid, "generation_errors": self.generation_errors, "parse_errors": self.parse_errors,
                "sources_complete": sum(self.samples[sid] >= n for sid, n in self.expected_by_source.items()),
                "sources_total": len(self.source_ids), "source_coverage_seen": len(self.samples),
                "families": dict(self.families), "mean_seconds": mean, "recent_mean_seconds": recent_mean,
                "elapsed_seconds": elapsed, "last_response_at": self.last_at, "seconds_since_response": silence,
                "eta_at": (now + timedelta(seconds=eta_seconds)).isoformat() if eta_seconds is not None else None,
                "cautious_eta_at": (now + timedelta(seconds=cautious_seconds)).isoformat() if cautious_seconds is not None else None,
                "remaining_seconds": eta_seconds, "cautious_remaining_seconds": cautious_seconds,
                "deadline": deadline.isoformat() if deadline else None,
                "model": manifest["model_id"], "device": "Apple M4 Pro / MPS", "precision": manifest["precision"],
                "gpu_memory_gib": manifest.get("mps_driver_gib"),
                "analysis_started": analysis_started, "analysis_complete": bool(post) and bool(analysis),
                "analysis_status": analysis.get("status"), "pdf_exists": pdf_ready,
                "final_review_pending": post.get("submission") != "final",
                "now": now.isoformat(), "error": manifest.get("error"),
            }


def serve(root, port):
    progress = Progress(root)
    document = root / "monitor/index.html"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                body, content_type = document.read_bytes(), "text/html; charset=utf-8"
            elif path == "/api/status":
                body = json.dumps(progress.snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Progress monitor: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(ROOT, args.port)
