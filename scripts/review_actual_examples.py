"""Record inspected real traces and an explicitly post-hoc lexical-matching sensitivity."""
from datetime import datetime,timezone
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_core import contingency
from analyze_results import bootstrap_group,cluster_phi_draws

ROOT=Path(__file__).resolve().parents[1]


def review_examples(root,config):
    root=Path(root)
    pairs=pd.read_csv(root/"outputs/results/paired_metrics.csv",keep_default_na=False)
    q=pd.read_csv(root/"outputs/qualitative/qualitative_examples.csv",keep_default_na=False)
    notes={
      "Q1":("Deleting the color complement left 'player wearing crouches'; the malformed proposal was excluded before inference.","A parser error in intervention construction is not evidence about the model's explanation faithfulness."),
      "Q2":("Adding 'strange' changed entailment to neutral without a lexical mention of the added target.","The hillside-versus-woods ambiguity already existed; the new rationale emphasizes it. This single trace does not reveal the internal mechanism."),
      "Q3":("The new explanation repeats 'unusual', but the semantic prediction remains neutral: mention without measured label impact.","Copying an incidental modifier can yield E=1; mention is not sufficient to establish that the target determined the decision."),
      "Q4":("The baseline mentions 'in the hospital', yet deleting that phrase leaves the prediction neutral under the documented deletion convention.","The visiting/receiving role mismatch may already support neutral. Baseline mention does not imply necessity for the class decision."),
      "Q5":("Deleting 'four' changes entailment to neutral; the baseline mentions the number and the new rationale notes that it is unspecified.","This is an aligned E=1,I=1 trace, but it does not verify the model's full internal reasoning or establish a counterfactual gold label."),
      "Q6":("Adding 'happy' changes neutral to entailment and is mentioned; both binary indicators equal one.","The claim 'together' was already ambiguous; emotional wording may shift interpretation. Agreement of E and I is not proof of a correct rationale."),
      "Q7":("Adding 'ordinary' changes neither the neutral prediction nor a lexical target mention.","An unmentioned modifier can be irrelevant to this decision; E=0,I=0 alone says little about the complete explanation."),
      "Q8":("Adding 'Canadian' changes entailment to neutral. The rationale says 'nationality', so the lexical E=0 misses a clear attribute-level reference.","Nationality was unsupported by the premise. This illustrates a lexical-matching false negative, not compelling evidence that the explanation concealed the edit."),
      "Q9":("The rationale repeats 'Canadian' while contradiction persists because the dress/purse colors are swapped: E=1,I=0.","The nationality term may be incidental copying; together with Q8, this can depress nationality-marker phi under exact/stem matching.")}
    for example_id,sid,layout in [("Q8","1970275389.jpg#0r1e",0),("Q9","4670365996.jpg#1r1c",0)]:
        if example_id not in set(q.example_id):
            row=pairs[(pairs.source_id==sid)&(pairs.family=="insertion")&(pairs.layout_id==layout)&pairs.target.eq("Canadian")].iloc[0].to_dict()
            row.update(example_id=example_id,case_type=f'E{int(row["E_D"])}_I{int(row["I_D"])}')
            q=pd.concat([q,pd.DataFrame([row])],ignore_index=True)
    manifest=json.loads((root/"outputs/full_run_manifest.json").read_text())
    raw=pd.DataFrame([json.loads(line) for line in (root/manifest["checkpoint"]).read_text().splitlines()]).set_index("job_id")
    for idx,row in q.iterrows():
        q.loc[idx,"interpretation"],q.loc[idx,"alternative_explanation"]=notes[row.example_id]
        q.loc[idx,"reviewer"]="user and Codex discussion" if row.example_id=="Q2" else "Codex analyst inspection of actual outputs"
        q.loc[idx,"reviewed_at_utc"]=datetime.now(timezone.utc).isoformat()
        q.loc[idx,"raw_csv_file"]="outputs/raw/full_raw_outputs.csv"
        for prefix in ("baseline","edited"):
            key=row.get(prefix+"_job_id","")
            q.loc[idx,prefix+"_raw_response"]=raw.loc[key,"raw_response"] if key in raw.index else ""
    q.to_csv(root/"outputs/qualitative/reviewed_qualitative.csv",index=False)
    q.to_csv(root/"outputs/qualitative/qualitative_examples.csv",index=False)
    audit=pd.read_csv(root/"outputs/qualitative/mention_review.csv",keep_default_na=False)
    for idx,row in audit.iterrows():
        literal=int(float(row.E_D))
        semantic=literal
        note="No explicit reference to the target or a clear paraphrase in the checked explanation."
        false_match=0;paraphrase=0
        if row.family=="deletion" and row.target=="black":
            semantic="uncertain";note="'blackboard' contains 'black' but denotes an object; whether it describes the deleted color is ambiguous. The boundary rule avoids counting this substring automatically."
        elif row.family=="deletion" and row.target=="red":
            note="The paper substring rule matches 'red' inside 'occurred'; this is a false lexical match unrelated to the color.";false_match=1
        elif row.family=="deletion" and row.target=="on the hillside":
            semantic=1;paraphrase=1;note="The baseline explicitly says 'on a hillside'; the changed article makes full-phrase token/stem matching miss the reference."
        elif literal:
            note="The target is explicitly present; contextual inspection confirms a lexical reference. This does not establish causal importance."
        audit.loc[idx,"human_or_analyst_semantic_mention"]=semantic
        audit.loc[idx,"false_match"]=false_match
        audit.loc[idx,"paraphrase_missed"]=paraphrase
        audit.loc[idx,"reviewer"]="Codex analyst inspection; not an independent human annotation study"
        audit.loc[idx,"notes"]=note
    audit.to_csv(root/"outputs/qualitative/mention_review.csv",index=False)
    return run_mention_sensitivity(root,config)


def run_mention_sensitivity(root,config):
    root=Path(root)
    pairs=pd.read_csv(root/"outputs/results/paired_metrics.csv",keep_default_na=False)
    # Post-hoc sensitivity motivated by Q8. Preserve primary E_D, p-values and membership.
    f=pairs[pairs.family=="insertion"].copy()
    f["E_D"]=f.E_D.astype(int);f["I_D"]=f.I_D.astype(int)
    cue=f.edited_explanation.str.contains(r"\bnationalit(?:y|ies)\b",case=False,regex=True)
    f["semantic_proxy_E"]=(f.E_D.eq(1)|(f.target.isin(["French","Canadian"])&cue)).astype(int)
    f["new_proxy_match"]=(f.semantic_proxy_E!=f.E_D)
    proxy_audit=f[f.new_proxy_match].drop_duplicates(["source_id","I_D"]).groupby("I_D",group_keys=False).head(10).copy()
    proxy_audit["reviewer"]="Codex analyst inspection of target, context and explanation"
    proxy_audit["attribute_reference_confirmed"]=True
    proxy_audit["notes"]="The rationale explicitly discusses unspecified nationality, referring to the inserted nationality assertion. This is an attribute-level proxy, not a general semantic oracle."
    audit_path=root/"outputs/qualitative/nationality_proxy_audit.csv"
    if not audit_path.exists():proxy_audit.to_csv(audit_path,index=False)
    sample=pd.read_csv(root/"data/frozen/frozen_sample.csv")
    weights=np.random.default_rng(config["bootstrap_seed"]).multinomial(len(sample),np.ones(len(sample))/len(sample),size=config["bootstrap_draws"])
    results=[];draws={}
    for level in ("non_entity","entity"):
        group=f[f.named_entity==level]
        for rule in ("primary_lexical","nationality_attribute_proxy"):
            g=group.copy()
            if rule!="primary_lexical":g["E_D"]=g.semantic_proxy_E
            results.append(dict(analysis="post_hoc_sensitivity",family="insertion",level=level,mention_rule=rule,n_sources=g.source_id.nunique(),
                   **contingency(zip(g.E_D,g.I_D)),**bootstrap_group(g,sample.source_id.tolist(),weights)))
            draws[level,rule]=cluster_phi_draws(g,sample.source_id.tolist(),weights)
    for rule in ("primary_lexical","nationality_attribute_proxy"):
        delta=draws["non_entity",rule]-draws["entity",rule]
        finite=delta[np.isfinite(delta)]
        first=next(r for r in results if r["level"]=="non_entity" and r["mention_rule"]==rule)
        second=next(r for r in results if r["level"]=="entity" and r["mention_rule"]==rule)
        results.append(dict(analysis="post_hoc_sensitivity",family="insertion",level="non_entity_minus_entity",mention_rule=rule,
                            phi=first["phi"]-second["phi"],phi_ci_low=np.quantile(finite,.025),phi_ci_high=np.quantile(finite,.975),
                            bootstrap_valid_draws=len(finite),bootstrap_undefined_draws=len(delta)-len(finite)))
    pd.DataFrame(results).to_csv(root/"outputs/results/mention_sensitivity.csv",index=False)
    f[["source_id","edit_id","layout_id","target","E_D","semantic_proxy_E","I_D","new_proxy_match","edited_explanation"]].to_csv(root/"outputs/results/mention_sensitivity_rows.csv",index=False)
    note={"analysis":"post_hoc_nationality_attribute_mention_sensitivity","primary_tests_unchanged":True,
          "rule":"For French/Canadian insertions only, count an explicit nationality/nationalities reference in the new explanation in addition to the primary lexical rule",
          "new_matches":int(f.new_proxy_match.sum()),"new_matches_with_I1":int(((f.new_proxy_match)&f.I_D.eq(1)).sum()),
          "proxy_audit_rows":len(proxy_audit),"scope":"only two nationality adjectives; deterministic keyword proxy, not a full semantic mention gold standard",
          "interpretation":"Checks whether the primary group contrast depends on lexical mention operationalization. This exploratory check does not establish general semantic faithfulness."}
    (root/"outputs/results/sensitivity_manifest.json").write_text(json.dumps(note,indent=2)+"\n")
    print(json.dumps(note,indent=2))
    return note


if __name__=="__main__":review_examples(ROOT,json.loads((ROOT/"config/experiment.json").read_text()))
