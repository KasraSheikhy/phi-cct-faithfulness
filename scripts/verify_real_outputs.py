"""Independent checks on completed real records, without modifying the frozen inputs."""
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]


def verify_real_outputs(root):
    cfg=json.loads((root/"config/experiment.json").read_text())
    protocol=json.loads((root/"config/protocol_manifest.json").read_text())
    assert all(hashlib.sha256((root/f).read_bytes()).hexdigest()==expected for f,expected in protocol["files"].items())
    manifest=json.loads((root/"outputs/full_run_manifest.json").read_text())
    raw=pd.DataFrame([json.loads(line) for line in (root/manifest["checkpoint"]).read_text().splitlines()])
    pairs=pd.read_csv(root/"outputs/results/paired_metrics.csv")
    sample=pd.read_csv(root/"data/frozen/frozen_sample.csv")
    assert len(raw)==raw.job_id.nunique()==9000 and raw.source_id.nunique()==1000
    assert raw.config_hash.nunique()==1 and raw.config_hash.iloc[0]==manifest["config_hash"]
    assert raw.generation_status.eq("ok").all() and raw.parse_status.eq("ok").all()
    for row in raw.itertuples():
        letter=re.search(r"\nAnswer:\s*([ABC])\s*$",row.raw_response).group(1)
        assert cfg["option_layouts"][row.layout_id][ord(letter)-65]==row.semantic_label
        assert row.raw_response.lstrip().startswith("Explanation:")
    indexed=raw.set_index("job_id")
    assert len(pairs)==6000 and pairs.validity_status.eq("valid").all()
    for row in pairs.itertuples():
        before=indexed.loc[row.baseline_job_id];after=indexed.loc[row.edited_job_id]
        assert before.source_id==after.source_id==row.source_id and before.layout_id==after.layout_id==row.layout_id
        assert int(before.semantic_label!=after.semantic_label)==row.I_D
        text=after.explanation if row.family=="insertion" else before.explanation
        assert text==(row.edited_explanation if row.family=="insertion" else row.baseline_explanation)
        original=row.original_text
        reconstructed=original[:int(row.original_span_start)]+(row.replacement if isinstance(row.replacement,str) else "")+original[int(row.original_span_end):]
        assert reconstructed==row.edited_text
    summary=pd.read_csv(root/"outputs/results/summary.csv")
    for family,g in pairs.groupby("family"):
        result=summary[summary.family==family].iloc[0]
        assert np.isclose(np.corrcoef(g.E_D,g.I_D)[0,1],result.phi)
        for e,i,name in [(0,0,"n00"),(0,1,"n01"),(1,0,"n10"),(1,1,"n11")]:
            assert ((g.E_D==e)&(g.I_D==i)).sum()==result[name]
    baseline=raw[raw.edit_id=="BASELINE"]
    gold=sample.set_index("source_id").gold_semantic_label
    accuracy=(baseline.semantic_label.to_numpy()==baseline.source_id.map(gold).to_numpy()).mean()
    assert np.isclose(accuracy,pd.read_csv(root/"outputs/results/accuracy.csv").valid_baseline_accuracy.iloc[0])
    tests=pd.read_csv(root/"outputs/results/tests.csv")
    independent_q=multipletests(tests.p_raw.fillna(1),method="fdr_bh")[1]
    assert np.allclose(tests.loc[tests.p_raw.notna(),"p_bh"],independent_q[tests.p_raw.notna()])
    assert len(tests)==10 and tests.status.eq("non_estimable").sum()==1
    record={"verified_at_utc":datetime.now(timezone.utc).isoformat(),"data_type":"actual_full_run",
       "passed":True,"n_raw_responses":len(raw),"n_source_examples":len(sample),"n_pairs":len(pairs),
       "checks":["frozen_protocol_checksums","unique_jobs_and_single_configuration","explanation_before_answer",
                 "all_option_letters_map_to_canonical_semantics","all_pair_keys_and_semantic_impacts",
                 "all_edit_spans_reconstruct_text","insertion_new_vs_deletion_baseline_convention",
                 "phi_independent_Pearson_and_contingency_counts","accuracy_independent_recalculation","BH_independent_statsmodels_check"],
       "checkpoint_sha256":hashlib.sha256((root/manifest["checkpoint"]).read_bytes()).hexdigest()}
    (root/"outputs/real_output_verification.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
    return record


if __name__=="__main__":verify_real_outputs(ROOT)
