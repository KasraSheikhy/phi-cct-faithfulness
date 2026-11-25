"""Build a verified delivery archive from an explicit allowlist; exclude model/env caches."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import zipfile
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def package(root):
    root=Path(root)
    coverage=pd.read_csv(root/"outputs/results/coverage.csv")
    assert len(coverage)==1000 and coverage.completion_status.eq("complete").all()
    assert coverage.valid_pairs.sum()==6000
    assert len(PdfReader(root/"report/phi_cct_report.pdf").pages)==2
    q=pd.read_csv(root/"outputs/qualitative/qualitative_examples.csv",keep_default_na=False)
    assert len(q)>=5 and not q.interpretation.str.contains("REVIEW_REQUIRED").any()
    validation=json.loads((root/"outputs/notebook_validation.json").read_text())
    assert validation["canonical_restart_run_all"]=="passed_in_clean_directory"
    paths=set()
    for name in ["README.md","GUIDE_FA.md","LICENSE","CITATION.cff","THIRD_PARTY_NOTICES.md",
                 ".gitignore","requirements-bootstrap.txt","requirements-local.lock.txt"]:paths.add(root/name)
    for pattern in ["config/*.json","scripts/*.py","notebooks/*.ipynb","data/**/*","references/*.md",
                    "outputs/results/*","outputs/qualitative/*","outputs/figures/*","report/*.pdf","monitor/*.html"]:
        paths.update(p for p in root.glob(pattern) if p.is_file())
    paths.update(p for p in (root/"docs").glob("*.md") if p.name!="work_log.md")
    for name in ["full_run_manifest.json","smoke_run_manifest.json","pilot_v1_manifest.json","pilot_v1_compute_decision.json",
                 "hardware_initial_load_failure.json","preflight.json","compute_decision.json","pipeline_verification.json",
                 "real_output_verification.json","analysis_export_verification.json","resume_verification.json",
                 "notebook_validation.json","pdf_verification.json","postprocess_status.json"]:
        if (root/"outputs"/name).exists():paths.add(root/"outputs"/name)
    for name in ["full_raw_outputs.csv","smoke_raw_outputs.csv","pilot_v1_raw_outputs.csv"]:
        paths.add(root/"outputs/raw"/name)
    for name in ["full_run_manifest.json","smoke_run_manifest.json","pilot_v1_manifest.json"]:
        m=json.loads((root/"outputs"/name).read_text());paths.add(root/m["checkpoint"])
    paths.add(root/"cache/nltk/sentiment/vader_lexicon.zip")
    assert all(p.exists() for p in paths)
    assert not any(".venv" in p.parts or "huggingface" in p.parts or "__pycache__" in p.parts for p in paths)
    inventory={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"completed_evaluation_with_documented_methodological_limitations",
              "n_source_examples":1000,"n_raw_main_responses":9000,"n_valid_pairs":6000,
              "primary_tests":10,"non_estimable_tests":1,"report_pages":2,"sha256_files":inventory}
    target=root/"deliverables";target.mkdir(exist_ok=True)
    archive=target/"phi-cct-esnli-qwen3-8b-submission.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,Path("phi-cct-faithfulness")/p.relative_to(root))
        z.writestr("phi-cct-faithfulness/SUBMISSION_MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in inventory.items():
            assert hashlib.sha256(z.read("phi-cct-faithfulness/"+name)).hexdigest()==digest
    receipt={"archive":archive.name,"bytes":archive.stat().st_size,"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
             "file_count":len(paths)+1,"verified_zip_crc_and_all_file_hashes":True}
    (target/"archive_verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    return archive


if __name__=="__main__":package(ROOT)
