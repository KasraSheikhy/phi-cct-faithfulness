"""Read-only hardware/package probe; no model download or GPU allocation."""
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


def hardware_report():
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "free_disk_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 2),
        "packages": {},
        "model_loaded": False,
    }
    for package in ("torch", "transformers", "accelerate", "numpy", "pandas", "scipy", "statsmodels", "spacy", "nltk"):
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
    if platform.system() == "Darwin":
        proc = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            hw = json.loads(proc.stdout)["SPHardwareDataType"][0]
            # Only include relevant specifications, never serials or hardware UUIDs.
            report["hardware"] = {key: hw.get(key) for key in ("machine_name", "chip_type", "physical_memory")}
    try:
        import torch
        report["mps_built"] = torch.backends.mps.is_built()
        report["mps_available"] = torch.backends.mps.is_available()
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            report["cuda_devices"] = [
                {"name": torch.cuda.get_device_name(i),
                 "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                for i in range(torch.cuda.device_count())
            ]
        if report["mps_available"] and hasattr(torch.mps, "recommended_max_memory"):
            report["mps_recommended_max_memory_gib"] = round(torch.mps.recommended_max_memory() / 2**30, 2)
        report["status"] = "hardware_probe_only_model_smoke_pending"
    except ImportError as exc:
        report["status"] = "dependencies_missing"
        report["dependency_error"] = str(exc)
    return report


if __name__ == "__main__":
    result = hardware_report()
    destination = Path(__file__).resolve().parents[1] / "outputs" / "preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
