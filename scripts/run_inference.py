"""Resumable FP16 PyTorch inference. Main evaluation requires a completed blinded audit."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from experiment_core import stable_hash, build_prompt, parse_response

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME",str(ROOT/"cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET","1")


def read_jsonl(path):
    if not path.exists():
        return []
    result=[]
    lines=path.read_text().splitlines()
    for index,line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
            # Interrupted last write: remove only the unterminated tail before resuming.
            path.with_suffix(".interrupted-tail.txt").write_text(line)
            path.write_text("\n".join(lines[:index])+"\n")
    return result


def append_record(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())


def validate_audit(root,edits):
    audit=pd.read_csv(root/"data/audit/edit_audit.csv",keep_default_na=False)
    reviewed=audit[audit.decision.isin(["accept","reject"])]
    if len(reviewed)<20 or set(reviewed.family)!={"insertion","deletion"}:
        raise RuntimeError("At least 20 completed blinded human audit records covering both families are required before study inference")
    if (reviewed.reviewer.str.strip()=="").any() or (reviewed.reviewed_at.str.strip()=="").any():
        raise RuntimeError("Audit reviewer and timestamp are required")
    if not reviewed.outcomes_blinded.astype(str).str.lower().isin(["true","1"]).all():
        raise RuntimeError("Audit must be outcome-blinded")
    for col in ("grammaticality","coherence","meaning_changed","gold_label_may_change"):
        if (reviewed[col].astype(str).str.strip()=="").any():
            raise RuntimeError("Audit must record all requested quality/meaning judgments")
    rejected=set(reviewed.loc[reviewed.decision=="reject","edit_id"])
    exclusions=root/"data/audit/quality_exclusions.csv"
    if exclusions.exists():
        rejected.update(pd.read_csv(exclusions,keep_default_na=False).edit_id)
    return edits[~edits.edit_id.isin(rejected)].copy(),audit


def load_qwen(root,config,backend):
    if not config.get("model_revision"):
        raise RuntimeError("Resolve immutable model revision before inference")
    if config["quantization"] or config["do_sample"] or config["enable_thinking"]:
        raise RuntimeError("Required controlled inference settings violated")
    if backend=="mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Mac GPU unavailable to this process")
    if backend=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    # Explicit cache path works even when HF_HOME was set after library import.
    cache_dir=str(root/"cache/huggingface/hub")
    tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["model_revision"],cache_dir=cache_dir,local_files_only=True)
    start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(config["model_id"],revision=config["model_revision"],
              cache_dir=cache_dir,local_files_only=True,dtype=torch.float16,device_map=None if backend=="mps" else {"":backend},
              low_cpu_mem_usage=True,attn_implementation="sdpa").eval()
    if backend=="mps":
        # Avoid Transformers' single 14+ GiB allocator warm-up buffer, which exceeds
        # Metal's single-buffer limit. Normal per-parameter transfers retain all
        # FP16 weights and MPS memory safety limits; inference remains entirely on GPU.
        model=model.to("mps")
    if getattr(model.config,"quantization_config",None):
        raise RuntimeError("Unexpected quantized model")
    if any(p.dtype!=torch.float16 for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError("Unexpected model parameter precision")
    if any(p.device.type!=backend for p in model.parameters()):
        raise RuntimeError("Unexpected CPU/offloaded parameter in controlled GPU run")
    generation=GenerationConfig(do_sample=False,max_new_tokens=config["max_new_tokens"],
            eos_token_id=model.config.eos_token_id,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,use_cache=True)
    return model,tokenizer,generation,time.perf_counter()-start


def generate_one(model,tokenizer,generation,prompt,backend):
    start=time.perf_counter()
    record=dict(raw_response="",generation_status="not_started",stop_reason="",error="",input_tokens=0,generated_tokens=0)
    try:
        inputs=tokenizer.apply_chat_template([{"role":"user","content":prompt}],
                add_generation_prompt=True,enable_thinking=False,tokenize=True,return_dict=True,return_tensors="pt")
        record["input_tokens"]=inputs["input_ids"].shape[-1]
        if record["input_tokens"]+generation.max_new_tokens>model.config.max_position_embeddings:
            record.update(generation_status="input_too_long",stop_reason="context_limit")
            return record
        inputs={k:v.to(backend) for k,v in inputs.items()}
        with torch.inference_mode():
            output=model.generate(**inputs,generation_config=generation)
        if backend=="mps": torch.mps.synchronize()
        if backend=="cuda": torch.cuda.synchronize()
        ids=output[0,record["input_tokens"]:].tolist()
        eos=generation.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos)
        ended=bool(ids and ids[-1] in eos)
        record.update(raw_response=tokenizer.decode(ids,skip_special_tokens=True),generated_tokens=len(ids),
                      generation_status="ok" if ended else "token_limit",stop_reason="eos" if ended else "max_new_tokens")
    except Exception as exc:
        record.update(generation_status="generation_error",error=f'{type(exc).__name__}: {exc}',stop_reason="exception")
        gc.collect()
        if backend=="mps": torch.mps.empty_cache()
        if backend=="cuda": torch.cuda.empty_cache()
    finally:
        record["elapsed_seconds"]=time.perf_counter()-start
    return record


def make_jobs(sample,edits,config):
    jobs=[]
    for s in sample.to_dict("records"):
        for layout_id,layout in enumerate(config["option_layouts"]):
            for edit in [None]+edits[edits.source_id==s["source_id"]].to_dict("records"):
                edit_id="BASELINE" if edit is None else edit["edit_id"]
                premise=s["premise"] if edit is None else edit["edited_premise"]
                hypothesis=s["hypothesis"] if edit is None else edit["edited_hypothesis"]
                prompt=build_prompt(premise,hypothesis,layout)
                jobs.append(dict(source_id=s["source_id"],edit_id=edit_id,layout_id=layout_id,
                      family="baseline" if edit is None else edit["family"],prompt=prompt,
                      prompt_hash=stable_hash(prompt),input_hash=stable_hash([premise,hypothesis]),
                      gold_semantic_label=s["gold_semantic_label"]))
    return jobs


def run(root,config,mode="smoke",backend="mps"):
    root=Path(root)
    torch.manual_seed(config["sample_seed"])
    sample=pd.read_csv(root/"data/frozen/frozen_sample.csv",keep_default_na=False)
    edits=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False)
    if mode!="hardware":
        protocol=json.loads((root/"config/protocol_manifest.json").read_text())
        for filename,expected in protocol["files"].items():
            if hashlib.sha256((root/filename).read_bytes()).hexdigest()!=expected:
                raise RuntimeError("Frozen protocol input changed: "+filename)
        edits,audit=validate_audit(root,edits)
    if mode=="hardware":
        # Synthetic infrastructure check, not study evidence or intervention audit.
        sample=pd.DataFrame([dict(source_id="SYNTHETIC_HARDWARE_ONLY",premise="A person walks beside a dog.",hypothesis="A person is outdoors.",gold_semantic_label="neutral")])
        edits=edits.iloc[:0]
    elif mode=="smoke":
        sample=sample.iloc[:config["smoke_source_count"]]
    jobs=make_jobs(sample,edits,config)
    config_hash=stable_hash({"configuration":config,"backend":backend,
        "core_sha256":hashlib.sha256((root/"scripts/experiment_core.py").read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256((root/"scripts/run_inference.py").read_bytes()).hexdigest(),
        "edits_sha256":hashlib.sha256((root/"data/edits/selected_edits.csv").read_bytes()).hexdigest(),
        "audit_sha256":hashlib.sha256((root/"data/audit/edit_audit.csv").read_bytes()).hexdigest() if mode!="hardware" else "not_applicable",
        "quality_exclusions_sha256":hashlib.sha256((root/"data/audit/quality_exclusions.csv").read_bytes()).hexdigest() if (root/"data/audit/quality_exclusions.csv").exists() else "none",
        "torch":torch.__version__})
    out=root/"outputs/checkpoints"/f'{mode}_{backend}_{config_hash[:12]}.jsonl'
    existing=read_jsonl(out)
    completed={r["job_id"] for r in existing}
    for job in jobs:
        job["job_id"]=stable_hash([config_hash,job["source_id"],job["edit_id"],job["layout_id"],job["prompt_hash"]])
    manifest_path=root/"outputs"/f'{mode}_run_manifest.json'
    previous_manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous_manifest.get("config_hash")!=config_hash: previous_manifest={}
    manifest=dict(mode=mode,backend=backend,config_hash=config_hash,checkpoint=str(out.relative_to(root)),
            model_id=config["model_id"],model_revision=config["model_revision"],precision="float16",quantization=False,
            do_sample=False,enable_thinking=False,response_order="explanation_then_label",torch_version=torch.__version__,
            platform=platform.platform(),expected_jobs=len(jobs),completed_jobs=len(completed),status="loading",
            loading_strategy="cpu_fp16_then_per_parameter_mps_transfer" if backend=="mps" else "direct_gpu_device_map",
            started_at_utc=datetime.now(timezone.utc).isoformat())
    manifest={**previous_manifest,**manifest}
    if previous_manifest.get("started_at_utc"):
        manifest["started_at_utc"]=previous_manifest["started_at_utc"]
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    if all(j["job_id"] in completed for j in jobs):
        mean=sum(r["elapsed_seconds"] for r in existing)/len(existing) if existing else None
        manifest.update(status="complete",completed_jobs=len(jobs),valid_responses=sum(r["parse_status"]=="ok" for r in existing),
                        mean_response_seconds=mean,projected_9000_response_hours=mean*9000/3600 if mean is not None else None)
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        return existing
    try:
        model,tokenizer,generation,load_time=load_qwen(root,config,backend)
    except Exception as exc:
        manifest.update(status="load_failed",error=f'{type(exc).__name__}: {exc}')
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        raise
    manifest.update(status="running",load_seconds=load_time)
    history=list(existing)
    for job in jobs:
        if job["job_id"] in completed: continue
        result=generate_one(model,tokenizer,generation,job["prompt"],backend)
        parsed=parse_response(result["raw_response"],config["option_layouts"][job["layout_id"]],result["generation_status"])
        record={**job,**result,**parsed,"model_id":config["model_id"],"model_revision":config["model_revision"],
                "config_hash":config_hash,"backend":backend,"generated_at_utc":datetime.now(timezone.utc).isoformat()}
        append_record(out,record)
        history.append(record)
        manifest.update(completed_jobs=len(history),valid_responses=sum(r["parse_status"]=="ok" for r in history))
        if backend=="mps":
            manifest["mps_allocated_gib"]=torch.mps.current_allocated_memory()/2**30
            manifest["mps_driver_gib"]=torch.mps.driver_allocated_memory()/2**30
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        print(f'{mode}: {len(history)}/{len(jobs)}; {result["elapsed_seconds"]:.2f}s; {parsed["parse_status"]}',flush=True)
        if result["generation_status"]=="generation_error":
            # Stop systemic GPU/OOM failures instead of recording thousands of identical errors.
            manifest.update(status="incomplete_stopped_after_generation_error")
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
            break
    else:
        manifest["status"]="complete"
    durations=[r["elapsed_seconds"] for r in history]
    manifest["mean_response_seconds"]=sum(durations)/len(durations) if durations else None
    manifest["projected_9000_response_hours"]=manifest["mean_response_seconds"]*9000/3600 if durations else None
    manifest["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    pd.DataFrame(history).to_csv(root/"outputs/raw"/f'{mode}_raw_outputs.csv',index=False)
    print(json.dumps(manifest,indent=2),flush=True)
    del model
    gc.collect()
    if backend=="mps": torch.mps.empty_cache()
    return history


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["hardware","smoke","full"],default="smoke")
    p.add_argument("--backend",choices=["mps","cuda"],default="mps")
    args=p.parse_args()
    run(ROOT,json.loads((ROOT/"config/experiment.json").read_text()),args.mode,args.backend)
