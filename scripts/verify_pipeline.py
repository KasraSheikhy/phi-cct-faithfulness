"""Scientific correctness checks using synthetic data, never mixed with study outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_core import validate_core,contingency
from analyze_results import phi_array,bootstrap_group,interaction_test,phi_equality_test,bh_planned_family,make_pairs

ROOT=Path(__file__).resolve().parents[1]


def verify():
    validate_core()
    rng=np.random.default_rng(42)
    binary=rng.integers(0,2,size=(200,2))
    assert np.isclose(contingency(binary)["phi"],np.corrcoef(binary.T)[0,1])
    f=pd.DataFrame({"source_id":np.repeat([f's{i}' for i in range(80)],3),
                    "E_D":rng.integers(0,2,size=240),"I_D":rng.integers(0,2,size=240)})
    source_order=list(f.source_id.unique())
    weights=rng.multinomial(80,np.ones(80)/80,size=200)
    original=bootstrap_group(f,source_order,weights)
    duplicated=bootstrap_group(pd.concat([f,f],ignore_index=True),source_order,weights)
    assert original==duplicated, "Duplicating all rows within each cluster must not narrow its bootstrap CI"
    frame=pd.DataFrame({"source_id":[f'x{i}' for i in range(400)],"E_D":rng.integers(0,2,400),"Z":np.repeat(["a","b"],200)})
    probability=np.where(frame.Z=="b",np.where(frame.E_D==1,.9,.1),.5)
    frame["I_D"]=(rng.random(400)<probability).astype(int)
    test,contrasts=interaction_test(frame,"Z","SYNTHETIC_ONLY")
    assert test["status"]=="estimable" and test["p_raw"]<.001
    w=rng.multinomial(400,np.ones(400)/400,size=500)
    equality,effects=phi_equality_test(frame,"Z","SYNTHETIC_ONLY",frame.source_id.tolist(),w)
    assert equality["status"]=="estimable" and equality["p_raw"]<.01
    assert effects[0]["ci_low"]>0
    constant=frame.copy();constant["E_D"]=0
    assert interaction_test(constant,"Z","SYNTHETIC_ONLY")[0]["status"]=="non_estimable"
    bh=bh_planned_family(pd.DataFrame({"p_raw":[.001,.01,np.nan],"p_bh":[np.nan]*3}),10)
    assert np.allclose(bh.p_bh.iloc[:2],[.01,.05]) and np.isnan(bh.p_bh.iloc[2])
    layouts=[["entailment","neutral","contradiction"],["neutral","contradiction","entailment"]]
    sample=pd.DataFrame([{"source_id":"synthetic","gold_semantic_label":"entailment"}])
    edits=pd.DataFrame([{"source_id":"synthetic","edit_id":"edit","family":"insertion","target":"red"}])
    raw=pd.DataFrame([
       {"source_id":"synthetic","edit_id":"BASELINE","layout_id":0,"job_id":"b0","parse_status":"ok","explanation":"A synthetic red object.","semantic_label":"entailment"},
       {"source_id":"synthetic","edit_id":"edit","layout_id":0,"job_id":"e0","parse_status":"malformed","explanation":None,"semantic_label":None},
       {"source_id":"synthetic","edit_id":"BASELINE","layout_id":1,"job_id":"b1","parse_status":"ok","explanation":"A synthetic object.","semantic_label":"entailment"},
       {"source_id":"synthetic","edit_id":"edit","layout_id":1,"job_id":"e1","parse_status":"ok","explanation":"A synthetic red object.","semantic_label":"entailment"}])
    pairs=make_pairs(sample,edits,raw,layouts)
    assert np.isnan(pairs.iloc[0].E_D) and np.isnan(pairs.iloc[0].I_D)
    assert pairs.iloc[1].E_D==1 and pairs.iloc[1].I_D==0
    result={"data_type":"synthetic_checks_only_not_research_results","passed":True,
            "checks":["phi_equals_binary_Pearson","constant_phi_is_undefined","semantic_option_remapping",
                      "strict_explanation_then_answer","invalid_outputs_remain_missing","word_boundary_and_stem_matching",
                      "cluster_bootstrap_duplicate_invariance","moderation_model_detects_known_synthetic_interaction",
                      "constant_group_test_non_estimable","centered_cluster_bootstrap_phi_equality_detects_synthetic_difference",
                      "BH_uses_planned_family_size","pair_join_and_invalid_exclusion"]}
    (ROOT/"outputs/pipeline_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    verify()
