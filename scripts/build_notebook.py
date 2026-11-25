"""Package readable pipeline code and explicit frozen inputs in one portable notebook."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def build_notebook(root):
    root=Path(root)
    source_names=["experiment_core.py","freeze_sample.py","build_idf.py","prepare_edits.py","complete_edits.py","run_inference.py","analyze_results.py","review_actual_examples.py","verify_real_outputs.py"]
    sources={name:(root/"scripts"/name).read_text() for name in source_names}
    hashes={name:hashlib.sha256(src.encode()).hexdigest() for name,src in sources.items()}
    assets=["config/experiment.json","config/protocol_manifest.json","data/frozen/frozen_sample.csv",
         "data/frozen/sample_manifest.json","data/frozen/idf_reference.json","data/frozen/sentiment_reference.json",
         "data/edits/selected_edits.csv","data/edits/edit_proposals.csv","data/edits/feature_manifest.json",
         "data/audit/edit_audit.csv","data/audit/quality_exclusions.csv","data/audit/user_review_attestation.json",
         "cache/nltk/sentiment/vader_lexicon.zip","docs/experiment_plan.md"]
    recorded=["outputs/full_run_manifest.json","outputs/smoke_run_manifest.json","outputs/checkpoints/full_mps_f7e3f5770041.jsonl","outputs/checkpoints/smoke_mps_f7e3f5770041.jsonl"]
    for optional in ["outputs/qualitative/reviewed_qualitative.csv","outputs/qualitative/mention_review.csv","outputs/qualitative/nationality_proxy_audit.csv",*recorded]:
        if (root/optional).exists():assets.append(optional)
    payload={name:base64.b64encode(gzip.compress((root/name).read_bytes(),mtime=0)).decode() for name in assets}
    checksums={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in assets}
    cfg=json.loads((root/"config/experiment.json").read_text())
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s):cells.append(nbformat.v4.new_code_cell(s))
    md("""# Input and intervention moderators of phi-CCT

**Required study:** 1,000 frozen e-SNLI test sources, Qwen/Qwen3-8B, FP16 without quantization, greedy decoding, thinking disabled, explanation then prediction. Insertion and deletion are reported separately. All five requested variables are included, even when a test is not estimable.

This notebook contains the independently written Python/PyTorch pipeline and explicit frozen inputs. No hidden local scripts are required. An embedded compressed payload preserves the exact sampled rows, training-only IDF statistics, candidate edits, audit, and protocol; the generation algorithms are readable in subsequent cells. Model weights are downloaded from the pinned public model revision.

**Completed study:** all 9,000 scheduled responses and 6,000 valid pairs cover all 1,000 sources. Default `replay` mode recomputes analysis from the included, checksum-verified actual raw responses; it does not claim a new model run. `full` retains the complete PyTorch generation path. To regenerate rather than reuse checkpoints, choose a clean working directory, set RESTORE_RECORDED_OUTPUTS=False, and use a suitable GPU. No model weights are embedded.
""")
    md("## Configuration\nChoose a persistent working directory for checkpoints. On Colab, save/download this directory before the runtime expires; its local disk is temporary. Use the same backend and precision for all final comparisons.")
    code('''import os, sys, json, hashlib, platform
from pathlib import Path

RUN_MODE = "replay"  # replay | prepare | smoke | full
RESTORE_RECORDED_OUTPUTS = True
BACKEND = "auto"      # auto | mps | cuda
INSTALL_MISSING = True
if Path.cwd().name == "notebooks":
    ROOT = Path.cwd().parent
elif (Path.cwd()/"config/experiment.json").exists():
    ROOT = Path.cwd()
else:
    ROOT = Path.cwd()/"phi-cct-faithfulness"
ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(ROOT/"cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["NLTK_DATA"] = str(ROOT/"cache/nltk")
os.environ["MPLCONFIGDIR"] = str(ROOT/".cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT/".cache")
CONFIG = json.loads('''+repr(json.dumps(cfg))+''')
SCRIPT_HASHES = '''+repr(hashes)+'''
print("Working directory:",ROOT)
print("Run mode:",RUN_MODE,"; status:",CONFIG["status"])
''')
    md("## Dependencies and hardware\nMissing packages can be installed in this kernel. Exact versions from the evaluated runtime are exported; the Mac environment lock is platform-specific. Do not quantize the model or substitute another checkpoint.")
    code('''import importlib.util, importlib.metadata, subprocess
from packaging.requirements import Requirement
requirements = {
 "torch": "torch>=2.6,<3", "transformers": "transformers==4.57.6", "accelerate": "accelerate==1.15.0",
 "huggingface_hub": "huggingface-hub==0.36.2", "numpy":"numpy>=1.26,<3", "pandas":"pandas==2.3.3",
 "scipy":"scipy>=1.13,<2", "statsmodels":"statsmodels==0.14.6", "matplotlib":"matplotlib>=3.9,<4",
 "nltk":"nltk>=3.9,<4", "spacy":"spacy>=3.8,<4", "requests":"requests>=2.32,<3"
}
missing=[]
for module,req in requirements.items():
    parsed=Requirement(req)
    try: compatible=parsed.specifier.contains(importlib.metadata.version(parsed.name))
    except importlib.metadata.PackageNotFoundError: compatible=False
    if importlib.util.find_spec(module) is None or not compatible: missing.append(req)
if missing:
    if not INSTALL_MISSING: raise RuntimeError("Install missing dependencies: "+str(missing))
    subprocess.check_call([sys.executable,"-m","pip","install",*missing])
if importlib.util.find_spec("en_core_web_sm") is None:
    if not INSTALL_MISSING: raise RuntimeError("Install en_core_web_sm 3.8.0")
    subprocess.check_call([sys.executable,"-m","pip","install","https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"])
import torch
if BACKEND=="auto": BACKEND="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "unavailable"
ENVIRONMENT={"python":platform.python_version(),"platform":platform.platform(),"backend":BACKEND,
             "packages":{name:importlib.metadata.version(name) for name in ["torch","transformers","accelerate","numpy","pandas","scipy","statsmodels","spacy","nltk"]}}
if BACKEND=="cuda": ENVIRONMENT["gpu"]=torch.cuda.get_device_name(0)
if BACKEND=="mps": ENVIRONMENT["mps_recommended_gib"]=torch.mps.recommended_max_memory()/2**30
(ROOT/"outputs").mkdir(exist_ok=True)
(ROOT/"outputs/notebook_environment.json").write_text(json.dumps(ENVIRONMENT,indent=2)+chr(10))
print(json.dumps(ENVIRONMENT,indent=2))
''')
    md("## Explicit frozen inputs\nThe payload is data, not executable code. SHA-256 checks refuse to overwrite changed inputs. This preserves manual audit records and prevents changing the sample after outcomes. The original source URLs, training split, source revisions and hashes remain in the manifests.")
    code('import base64, gzip\nFROZEN_ASSETS = '+repr(payload)+'\nASSET_SHA256 = '+repr(checksums)+'''
for name,encoded in FROZEN_ASSETS.items():
    if not RESTORE_RECORDED_OUTPUTS and (name.startswith("outputs/checkpoints/") or name in {"outputs/full_run_manifest.json","outputs/smoke_run_manifest.json"}):
        continue
    data=gzip.decompress(base64.b64decode(encoded))
    assert hashlib.sha256(data).hexdigest()==ASSET_SHA256[name]
    path=ROOT/name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=ASSET_SHA256[name]:
        raise RuntimeError("Existing frozen input differs: "+name+". Use a clean directory or inspect the difference.")
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_bytes(data)
for folder in ["outputs/raw","outputs/results","outputs/checkpoints","outputs/figures","outputs/qualitative","report"]:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
print("Verified embedded frozen inputs:",len(FROZEN_ASSETS))
''')
    cells[-1].metadata.update({"tags":["hide-input"],"jupyter":{"source_hidden":True},"collapsed":True})
    sections={"experiment_core.py":"Core definitions: semantic mapping, explanation-first parsing, lexical mention rules and phi",
        "freeze_sample.py":"Sampling algorithm: fixed seed, sorted IDs, no outcome-dependent replacement",
        "build_idf.py":"Training-only IDF algorithm",
        "prepare_edits.py":"Outcome-blind counterfactual construction and annotations",
        "complete_edits.py":"Fallback for sources without a screened deletion",
        "run_inference.py":"Fresh-context, greedy FP16 inference and resumable checkpoints",
        "analyze_results.py":"Coverage, source-cluster bootstrap, all five tests and plots",
        "review_actual_examples.py":"Recorded qualitative annotations and post-hoc nationality-mention sensitivity",
        "verify_real_outputs.py":"Independent verification of actual raw outputs"}
    for name in source_names:
        md("## "+sections[name]+("\nThese functions document preparation. The frozen artifacts above are reused for the reported study; do not regenerate interventions after seeing outcomes." if name in ["freeze_sample.py","build_idf.py","prepare_edits.py","complete_edits.py"] else ""))
        src=sources[name].split('\nif __name__')[0]
        src=re.sub(r'^ROOT\s*=.*\n','',src,flags=re.M)
        src=re.sub(r'^from (experiment_core|prepare_edits|analyze_results) import .*\n','',src,flags=re.M)
        for filename in ("experiment_core.py","run_inference.py"):
            src=src.replace(f'hashlib.sha256((root/"scripts/{filename}").read_bytes()).hexdigest()',f'SCRIPT_HASHES["{filename}"]')
        code(src)
    md("## Verify the frozen design and metric\nAll checks below use the frozen input metadata or explicitly synthetic outputs; they do not create study findings. Numeric/date NER tags alone are not counted as named entities. Audit meaning/gold-label judgments remain explicitly uncertain after the pre-run global acceptability review.")
    code('''validate_core()
sample=pd.read_csv(ROOT/"data/frozen/frozen_sample.csv",keep_default_na=False)
manifest=json.loads((ROOT/"data/frozen/sample_manifest.json").read_text())
assert len(sample)==sample.source_id.nunique()==1000
assert hashlib.sha256((ROOT/"data/frozen/frozen_sample.csv").read_bytes()).hexdigest()==manifest["sample_sha256"]
edits=pd.read_csv(ROOT/"data/edits/selected_edits.csv",keep_default_na=False)
active_edits,audit=validate_audit(ROOT,edits)
print("Frozen sources:",len(sample),"; active proposals:",len(active_edits))
display(sample.gold_semantic_label.value_counts())
display(active_edits.groupby(["family","named_entity"]).size().rename("n_edits").to_frame())
display(active_edits.groupby(["family","target_polarity"]).size().rename("n_edits").to_frame())
print("Manual review:",len(audit),"global acceptability judgments; detailed semantic/gold changes uncertain.")
''')
    md("## Smoke and full evaluation\nThe primary settings remain fixed across every baseline/edit and layout. An insertion checks the new explanation; deletion checks the matched baseline explanation as a documented extension. Model/loading failures are recorded, not negative observations. The full run resumes only matching immutable configurations. Mode `prepare` performs no model inference.")
    code('''if RUN_MODE in {"smoke","full"}:
    if BACKEND not in {"mps","cuda"}: raise RuntimeError("A suitable GPU is required; CPU execution is not silently substituted")
    from huggingface_hub import snapshot_download
    snapshot_download(CONFIG["model_id"],revision=CONFIG["model_revision"],cache_dir=str(ROOT/"cache/huggingface/hub"),
                      allow_patterns=["*.json","*.safetensors","*.jinja","*.txt","*.model"],max_workers=2)
    smoke_history=run(ROOT,CONFIG,"smoke",BACKEND)
    smoke=json.loads((ROOT/"outputs/smoke_run_manifest.json").read_text())
    print("Smoke projection (9000 responses, before margin):",smoke.get("projected_9000_response_hours"),"hours")
    if smoke["status"]!="complete" or smoke["valid_responses"]<.95*smoke["expected_jobs"]:
        raise RuntimeError("Smoke failed or parsing success below 95%; diagnose and document changes before a full run")
    if RUN_MODE=="full":
        full_history=run(ROOT,CONFIG,"full",BACKEND)
elif RUN_MODE=="replay":
    recorded_manifest=json.loads((ROOT/"outputs/full_run_manifest.json").read_text())
    recorded=read_jsonl(ROOT/recorded_manifest["checkpoint"])
    assert recorded_manifest["status"]=="complete" and len(recorded)==9000
    assert len({r["job_id"] for r in recorded})==9000
    print("Replaying all 9000 actual recorded responses; no model generation is performed.")
elif RUN_MODE!="prepare":
    raise ValueError("Unknown RUN_MODE")
else:
    print("Preparation validated. Full evaluation remains incomplete until RUN_MODE=full finishes.")
''')
    md("## Analysis and traceable outputs\nFor each variable and intervention family, the null is equality of subgroup phi values. A centered source-cluster bootstrap gives an approximate omnibus p-value; BH uses all 10 planned tests. Sparse or degenerate groups are explicitly non-estimable. Percentile intervals omit undefined draws and report their count. Significant moderation alone does not establish a causal confounder.")
    code('''if RUN_MODE in {"smoke","full","replay"}:
    analysis=analyze(ROOT,CONFIG,"full" if RUN_MODE=="replay" else RUN_MODE)
    if RUN_MODE in {"full","replay"}:
        verify_real_outputs(ROOT)
        run_mention_sensitivity(ROOT,CONFIG)
    display(pd.read_csv(ROOT/"outputs/results/summary.csv"))
    display(pd.read_csv(ROOT/"outputs/results/tests.csv"))
    from IPython.display import Image,display
    for variable in VARIABLES:
        display(Image(filename=str(ROOT/f"outputs/figures/{variable}.png")))
    display(Image(filename=str(ROOT/"outputs/figures/position_by_role.png")))
    display(pd.read_csv(ROOT/"outputs/qualitative/qualitative_examples.csv").head(8))
    if RUN_MODE in {"full","replay"}:
        display(pd.read_csv(ROOT/"outputs/results/mention_sensitivity.csv"))
        display(pd.read_csv(ROOT/"outputs/results/verbosity.csv"))
    print("Primary planned tests and explicitly post-hoc sensitivity are kept separate. Full traces are in the qualitative CSV.")
else:
    print("No empirical metrics or p-values are produced by prepare mode.")
''')
    md("""## Interpretation and AI assistance

These are observational subgroup comparisons, with lexical and template differences between groups; they do not identify a causal effect of named entities, sentiment, rarity or location. Controlled option layouts are matched within source. A semantic flip is sensitivity, not itself evidence of unfaithfulness. Null results and non-estimable tests are included in the exported tables.

The completed primary result is an insertion named-entity contrast: non-entity phi=0.265 versus entity phi=-0.158, p=0.0005 and BH q=0.005. In this design all entity insertions are French/Canadian NORP markers. The other primary comparisons do not survive BH; deletion polarity is non-estimable under the pre-specified rule because only 13 sources have negative targets. The exploratory nationality-reference proxy changes entity phi to +0.257 and the group gap to 0.009, consistent with a lexical mention-matching mechanism. This post-hoc proxy is not a universal semantic oracle or a new confirmatory test.

AI assisted with independently writing this pipeline, literature/documentation review, debugging and artifact preparation. Synthetic checks validate metric arithmetic, semantic remapping and failure handling; model outputs and empirical interpretations require separate inspection. Official code was consulted as a cited reference, not reused as a pipeline. Primary mention matching adds word boundaries to avoid substring false matches; the paper's substring/stem rule is saved as a sensitivity flag. The audit records and verification artifacts document the human review and trace checks.

## References

- Siegel et al., [Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations, v3](https://arxiv.org/html/2503.13445v3), sections 3.2, 3.4, 4.2 and appendix G.
- [Official corr_faith code](https://github.com/google-deepmind/corr_faith).
- [Original e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI), revision recorded above.
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B), including the explicit thinking switch.
- [spaCy English models](https://spacy.io/models/en) and [NLTK VADER](https://www.nltk.org/api/nltk.sentiment.vader.html).
""")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    nbformat.validate(nb)
    nbformat.write(nb,root/"notebooks/phi_cct_esnli.ipynb")
    print("Built self-contained notebook with",len(cells),"cells and",len(assets),"explicit input assets")


if __name__=="__main__":
    build_notebook(ROOT)
