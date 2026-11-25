"""Follow model download through hardware test, smoke and the authorized full evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def stage(mode):
    log=ROOT/"outputs"/f'{mode}_execution.log'
    command=[sys.executable,str(ROOT/"scripts/run_inference.py"),"--mode",mode,"--backend","mps"]
    if shutil.which("caffeinate"):
        command=["caffeinate","-i",*command]
    print("Starting",mode,"; log",log.name,flush=True)
    with log.open("a") as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{mode} failed; inspect {log}')
    manifest=json.loads((ROOT/f'outputs/{mode}_run_manifest.json').read_text())
    print(json.dumps(manifest),flush=True)
    if manifest["status"]!="complete":
        raise RuntimeError(f'{mode} did not complete: {manifest["status"]}')
    return manifest


if __name__=="__main__":
    start=time.monotonic()
    while True:
        state_path=ROOT/"outputs/model_download.json"
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("status")=="downloaded":break
        if time.monotonic()-start>5400:raise RuntimeError("Download not ready after 90 minutes; inspect connection/status")
        time.sleep(5)
    stage("hardware")
    smoke=stage("smoke")
    if smoke["valid_responses"]<.95*smoke["expected_jobs"]:
        raise RuntimeError("Smoke parsing success below 95%; diagnose before full inference")
    cfg=json.loads((ROOT/"config/experiment.json").read_text())
    deadline=datetime.fromisoformat(cfg["deadline"])
    remaining_hours=(deadline-datetime.now(timezone.utc)).total_seconds()/3600
    projection=smoke["projected_9000_response_hours"]
    budget=dict(remaining_hours=remaining_hours,smoke_projected_hours=projection,
                projected_with_25_percent_margin=projection*1.25,decision="local_full_run" if projection*1.25<remaining_hours-.5 else "seek_faster_gpu")
    (ROOT/"outputs/compute_decision.json").write_text(json.dumps(budget,indent=2)+"\n")
    print(json.dumps(budget),flush=True)
    if budget["decision"]!="local_full_run":
        raise RuntimeError("Mac throughput leaves insufficient time with margin; use a faster Colab GPU")
    stage("full")
    print("Full inference finished; real analysis can now run.",flush=True)
