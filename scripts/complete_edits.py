"""Add a second insertion where no screened deletion exists, without changing review IDs."""
import json
from pathlib import Path
import pandas as pd
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from prepare_edits import annotate,digest

ROOT=Path(__file__).resolve().parents[1]


def complete_edits(root):
    selected=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False)
    proposals=pd.read_csv(root/"data/edits/edit_proposals.csv",keep_default_na=False)
    manifest=json.loads((root/"data/edits/feature_manifest.json").read_text())
    reference=json.loads((root/"data/frozen/idf_reference.json").read_text())
    nlp=spacy.load("en_core_web_sm")
    nltk.data.path.insert(0,str(root/"cache/nltk"))
    sia=SentimentIntensityAnalyzer()
    added=[]
    for sid,g in selected.groupby("source_id",sort=False):
        if len(g)>=2:continue
        pool=proposals[(proposals.source_id==sid)&(proposals.family=="insertion")&(proposals.selection_status=="eligible_not_selected")].sort_values("proposal_rank")
        if pool.empty:continue
        row=pool.iloc[0].to_dict()
        text=row["edited_text"];start=int(row["edited_span_start"]);end=int(row["edited_span_end"])
        row.update(annotate(text,start,end,nlp(text),sia,reference,row["role"],row["edited_premise"],row["edited_hypothesis"]))
        row.update(target_span_start=start,target_span_end=end,target_span_reference="edited",selection_status="selected_pending_audit")
        a,b=manifest["idf_bin_boundaries"]
        row["idf_bin"]="common" if row["idf"]<=a else "middle" if row["idf"]<=b else "rare"
        added.append(row)
        proposals.loc[proposals.edit_id==row["edit_id"],"selection_status"]="selected_pending_audit"
    if added:
        selected=pd.concat([selected,pd.DataFrame(added)],ignore_index=True)
        selected.to_csv(root/"data/edits/selected_edits.csv",index=False)
        proposals.to_csv(root/"data/edits/edit_proposals.csv",index=False)
        manifest.update(n_edits=len(selected),family_counts=selected.family.value_counts().to_dict(),
          fallback_policy="Second screened insertion for sources without screened deletion; both families remain separate in analysis",
          additional_insertions=len(added),selected_edits_sha256=digest((root/"data/edits/selected_edits.csv").read_text()),
          idf_bins_unchanged_after_additions=True)
        manifest["feature_counts"]={c:selected.groupby(["family",c]).size().to_dict().__repr__() for c in ["named_entity","target_polarity","position_bin","idf_bin"]}
        (root/"data/edits/feature_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print({"additional_insertions":len(added),"selected_edits":len(selected),"sources_below_two_edits":int((selected.groupby("source_id").size()<2).sum())})
    return selected


if __name__=="__main__":
    complete_edits(ROOT)
