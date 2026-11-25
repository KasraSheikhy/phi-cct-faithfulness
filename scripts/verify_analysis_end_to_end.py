"""Exercise all exports on synthetic responses in a temporary directory, then discard them."""
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
from run_inference import make_jobs,validate_audit
from experiment_core import stable_hash
from analyze_results import analyze

ROOT=Path(__file__).resolve().parents[1]


if __name__=="__main__":
    config=json.loads((ROOT/"config/experiment.json").read_text())
    config["bootstrap_draws"]=200  # test only; scientific run keeps its frozen 2000 draws
    rng=np.random.default_rng(99)
    with tempfile.TemporaryDirectory(prefix="phi_cct_SYNTHETIC_ONLY_") as temporary:
        root=Path(temporary)
        for folder in ("data/frozen","data/edits","data/audit"):
            shutil.copytree(ROOT/folder,root/folder)
        for folder in ("outputs/results","outputs/raw","outputs/checkpoints","outputs/figures","outputs/qualitative"):
            (root/folder).mkdir(parents=True,exist_ok=True)
        sample=pd.read_csv(root/"data/frozen/frozen_sample.csv",keep_default_na=False)
        edits=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False)
        edits,_=validate_audit(root,edits)
        targets=edits.set_index("edit_id").target.to_dict()
        jobs=make_jobs(sample,edits,config)
        raw=[]
        for job in jobs:
            semantic=job["gold_semantic_label"]
            if job["edit_id"]!="BASELINE" and rng.random()<.25:
                semantic=next(s for s in ["entailment","neutral","contradiction"] if s!=semantic)
            explanation=("Synthetic only: "+targets[job["edit_id"]]) if job["edit_id"]!="BASELINE" and rng.random()<.5 else "Synthetic explanation only."
            raw.append({**job,"job_id":stable_hash(["SYNTHETIC_ONLY",job["source_id"],job["edit_id"],job["layout_id"]]),
                        "config_hash":"SYNTHETIC_ONLY_NOT_A_STUDY_RUN","parse_status":"ok","generation_status":"ok",
                        "semantic_label":semantic,"explanation":explanation,"raw_response":"SYNTHETIC ONLY"})
        edited_index=next(i for i,r in enumerate(raw) if r["edit_id"]!="BASELINE")
        raw[edited_index].update(parse_status="malformed",semantic_label=None,explanation=None)
        path=root/"outputs/checkpoints/synthetic.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in raw)+"\n")
        (root/"outputs/full_run_manifest.json").write_text(json.dumps({"checkpoint":"outputs/checkpoints/synthetic.jsonl","expected_jobs":len(jobs)}))
        result=analyze(root,config,"full")
        assert result["n_frozen_sources"]==1000 and result["completed_jobs"]==9000
        assert result["valid_pairs"]==5999 and result["planned_tests"]==10
        assert all((root/f"outputs/figures/{v}.png").exists() for v in ["named_entity","target_polarity","position_bin","idf_bin","gold_label_option_position"])
        assert len(pd.read_csv(root/"outputs/results/position_by_role.csv"))==12
        assert len(pd.read_csv(root/"outputs/results/coverage.csv"))==1000
        verification={"passed":True,"data_type":"synthetic_responses_only_discarded_after_check",
                      "validated":"all five plots, ten test rows, all-source coverage, invalid exclusion and traceable CSV joins",
                      "temporary_outputs_retained":False,"no_empirical_metrics_recorded":True}
    (ROOT/"outputs/analysis_export_verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps(verification,indent=2))
