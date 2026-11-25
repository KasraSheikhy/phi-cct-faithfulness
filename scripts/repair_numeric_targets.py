"""Pre-outcome quality repair caught by synthetic integration testing; preserve all failures."""
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
import nltk,spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from prepare_edits import annotate,tokens,digest

ROOT=Path(__file__).resolve().parents[1]


if __name__=="__main__":
    root=ROOT
    selected=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False)
    proposals=pd.read_csv(root/"data/edits/edit_proposals.csv",keep_default_na=False)
    exclusions=pd.read_csv(root/"data/audit/quality_exclusions.csv",keep_default_na=False)
    bad=selected[~selected.target.map(lambda x:bool(tokens(str(x))))&~selected.edit_id.isin(exclusions.edit_id)]
    ref=json.loads((root/"data/frozen/idf_reference.json").read_text());m=json.loads((root/"data/edits/feature_manifest.json").read_text())
    nltk.data.path.insert(0,str(root/"cache/nltk"));sia=SentimentIntensityAnalyzer();nlp=spacy.load("en_core_web_sm")
    additions=[]
    for r in bad.to_dict("records"):
        excluded=dict(edit_id=r["edit_id"],source_id=r["source_id"],reason="digit-only target has no tokens under frozen alphabetic lexical/IDF rule",
                      reviewer="Codex pre-outcome integration check",user_decision="not_in_manual_review_packet",final_quality_decision="reject",outcomes_blinded=True)
        exclusions=pd.concat([exclusions,pd.DataFrame([excluded])],ignore_index=True)
        selected.loc[selected.edit_id==r["edit_id"],"selection_status"]="quality_excluded"
        proposals.loc[proposals.edit_id==r["edit_id"],["selection_status","rejection_reason"]]=["rejected","no_alphabetic_target_tokens"]
        pool=proposals[(proposals.source_id==r["source_id"])&proposals.selection_status.eq("eligible_not_selected")&proposals.target.map(lambda x:bool(tokens(str(x))))&~proposals.edit_id.isin(selected.edit_id)]
        ds=pool[pool.family=="deletion"]
        row=(ds if len(ds) else pool[pool.family=="insertion"]).sort_values("proposal_rank").iloc[0].to_dict()
        insertion=row["family"]=="insertion"
        text=row["edited_text"] if insertion else row["original_text"]
        start=int(row["edited_span_start"]) if insertion else text.find(row["target"],int(row["original_span_start"]))
        end=start+len(row["target"])
        premise=row["edited_premise"] if insertion else row["original_premise"]
        hypothesis=row["edited_hypothesis"] if insertion else row["original_hypothesis"]
        row.update(annotate(text,start,end,nlp(text),sia,ref,row["role"],premise,hypothesis))
        row.update(target_span_start=start,target_span_end=end,target_span_reference="edited" if insertion else "original",selection_status="selected_after_quality_review")
        a,b=m["idf_bin_boundaries"];row["idf_bin"]="common" if row["idf"]<=a else "middle" if row["idf"]<=b else "rare"
        additions.append(row)
        proposals.loc[proposals.edit_id==row["edit_id"],"selection_status"]="selected_after_quality_review"
    if additions:
        selected=pd.concat([selected,pd.DataFrame(additions)],ignore_index=True)
        selected.to_csv(root/"data/edits/selected_edits.csv",index=False)
        proposals.to_csv(root/"data/edits/edit_proposals.csv",index=False)
        exclusions.to_csv(root/"data/audit/quality_exclusions.csv",index=False)
        active=selected[~selected.edit_id.isin(exclusions.edit_id)]
        m.update(n_edits=len(selected),active_edits=len(active),family_counts=active.family.value_counts().to_dict(),
                 quality_excluded=exclusions.edit_id.tolist(),numeric_target_repairs=len(additions),
                 selected_edits_sha256=digest((root/"data/edits/selected_edits.csv").read_text()))
        m["feature_counts"]={c:active.groupby(["family",c]).size().to_dict().__repr__() for c in ["named_entity","target_polarity","position_bin","idf_bin"]}
        (root/"data/edits/feature_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print({"pre_outcome_repairs":len(additions),"active_families":m["family_counts"]})
