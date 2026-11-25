"""Continue the experiment with real analysis and review packets after full inference ends."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def write_review_packet(root):
    examples=pd.read_csv(root/"outputs/qualitative/qualitative_examples.csv",keep_default_na=False)
    lines=["# Real qualitative traces awaiting interpretation", "", "These are actual full-run records, not synthetic tests. Verify wording, semantic remapping, mention rules and alternative explanations before finalizing the report. REVIEW_REQUIRED is intentionally not a finished interpretation.", ""]
    for r in examples.to_dict("records"):
        lines += [f'## {r["example_id"]}: {r["case_type"]}',f'Source `{r["source_id"]}`; edit `{r["edit_id"]}`; layout {r["layout_id"]}.',
          f'Premise: {r.get("original_premise", "")}',f'Hypothesis: {r.get("original_hypothesis", "")}',
          f'Edited premise: {r.get("edited_premise", "")}',f'Edited hypothesis: {r.get("edited_hypothesis", "")}',
          f'Target: {r.get("target", "")}',f'Baseline explanation: {r.get("baseline_explanation", "")}',
          f'Edited explanation: {r.get("edited_explanation", "")}',
          f'Semantic prediction: {r.get("baseline_semantic_label", "")} -> {r.get("edited_semantic_label", "")}',
          f'E_D={r.get("E_D", "")}; I_D={r.get("I_D", "")}; validity={r.get("validity_status", "")}',
          f'Interpretation: {r.get("interpretation", "")}',f'Alternative: {r.get("alternative_explanation", "")}',
          f'Raw jobs: {r.get("baseline_job_id", "")} / {r.get("edited_job_id", "")}',""]
    (root/"outputs/qualitative/review_packet.md").write_text("\n".join(lines)+"\n")
    pairs=pd.read_csv(root/"outputs/results/paired_metrics.csv",keep_default_na=False)
    valid=pairs[pairs.validity_status=="valid"]
    selected=[]
    for family in ("insertion","deletion"):
        f=valid[valid.family==family]
        # Inspect different matcher decisions and both impact directions; no rows are excluded by this audit.
        ordered=pd.concat([f[f.E_D.astype(float)!=f.paper_rule_mention.astype(float)],f],ignore_index=True).drop_duplicates(["edit_id","layout_id"])
        selected.extend(ordered.head(12).to_dict("records"))
    review=pd.DataFrame(selected)
    for column in ["human_or_analyst_semantic_mention","false_match","paraphrase_missed","reviewer","notes"]:review[column]=""
    review.to_csv(root/"outputs/qualitative/mention_review.csv",index=False)


if __name__=="__main__":
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/".cache"))
    manifest_path=ROOT/"outputs/full_run_manifest.json"
    initial=json.loads(manifest_path.read_text())
    expected_hash=initial["config_hash"]
    while True:
        manifest=json.loads(manifest_path.read_text())
        if manifest["config_hash"]!=expected_hash:
            raise RuntimeError("Full-run configuration changed; do not combine results")
        if manifest["status"] not in {"loading","running"}:break
        time.sleep(30)
    if manifest.get("completed_jobs",0)==0:
        raise RuntimeError("No study responses to analyze; inspect full-run failure")
    with (ROOT/"outputs/full_analysis.log").open("w") as log:
        subprocess.run([sys.executable,str(ROOT/"scripts/analyze_results.py"),"--mode","full"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_review_packet(ROOT)
    state={"finished_at_utc":datetime.now(timezone.utc).isoformat(),"inference_status":manifest["status"],
           "analysis":"completed_on_actual_full_run_records","qualitative_review":"pending",
           "report_pdf":"pending_real_trace_review_and_visual_verification","submission":"not_yet_final"}
    (ROOT/"outputs/postprocess_status.json").write_text(json.dumps(state,indent=2)+"\n")
    print(json.dumps(state,indent=2),flush=True)
