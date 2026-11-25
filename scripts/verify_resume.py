"""Verify completed checkpoints resume without loading a model, using discarded synthetic jobs."""
import json
from pathlib import Path
import shutil
import tempfile
import pandas as pd
import run_inference as runner
from experiment_core import stable_hash

ROOT=Path(__file__).resolve().parents[1]


if __name__=="__main__":
    config=json.loads((ROOT/"config/experiment.json").read_text())
    with tempfile.TemporaryDirectory(prefix="phi_cct_resume_SYNTHETIC_") as d:
        root=Path(d)
        for folder in ("scripts","data/frozen","data/edits","data/audit"):
            shutil.copytree(ROOT/folder,root/folder)
        for folder in ("outputs/checkpoints","outputs/raw"):(root/folder).mkdir(parents=True,exist_ok=True)
        def prevent_model_load(*args,**kwargs):raise RuntimeError("SYNTHETIC_TEST_MODEL_LOADING_DISABLED")
        runner.load_qwen=prevent_model_load
        try:runner.run(root,config,"hardware","mps")
        except RuntimeError as exc:assert "SYNTHETIC_TEST_MODEL_LOADING_DISABLED" in str(exc)
        manifest=json.loads((root/"outputs/hardware_run_manifest.json").read_text())
        sample=pd.DataFrame([dict(source_id="SYNTHETIC_HARDWARE_ONLY",premise="A person walks beside a dog.",hypothesis="A person is outdoors.",gold_semantic_label="neutral")])
        edits=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False).iloc[:0]
        jobs=runner.make_jobs(sample,edits,config)
        raw=[]
        for job in jobs:
            raw.append({**job,"job_id":stable_hash([manifest["config_hash"],job["source_id"],job["edit_id"],job["layout_id"],job["prompt_hash"]]),
                        "elapsed_seconds":.1,"parse_status":"ok","generation_status":"ok","raw_response":"SYNTHETIC_TEST_ONLY"})
        path=root/manifest["checkpoint"];path.write_text("\n".join(json.dumps(r) for r in raw)+"\n")
        resumed=runner.run(root,config,"hardware","mps")
        final=json.loads((root/"outputs/hardware_run_manifest.json").read_text())
        assert len(resumed)==3 and final["status"]=="complete" and final["valid_responses"]==3
        assert final["projected_9000_response_hours"]>0
    record={"passed":True,"data_type":"discarded_synthetic_jobs_only","model_loaded":False,
            "completed_checkpoint_skips_model_loading":True,"resume_preserves_smoke_validity_and_timing_fields":True}
    (ROOT/"outputs/resume_verification.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
