"""Coverage, source-cluster bootstrap, planned moderation tests and traceable exports."""
import argparse
from pathlib import Path
import json
import os
import warnings
import numpy as np
import pandas as pd
os.environ.setdefault("XDG_CACHE_HOME",str(Path.cwd()/".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
from experiment_core import contingency,mention_flags

ROOT=Path(__file__).resolve().parents[1]
VARIABLES={"named_entity":["non_entity","entity"],"target_polarity":["negative","neutral","positive"],
           "position_bin":["early","middle","late"],"idf_bin":["common","middle","rare"],
           "gold_label_option_position":[1,2,3]}
DISPLAY={"named_entity":"Named entity","target_polarity":"Target sentiment",
         "position_bin":"Target position","idf_bin":"Lexical rarity (training IDF)",
         "gold_label_option_position":"Position of original gold label"}


def phi_array(counts):
    a,b,c,d=np.moveaxis(np.asarray(counts,dtype=float),-1,0)
    denom=np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return np.divide(a*d-b*c,denom,out=np.full_like(denom,np.nan),where=denom>0)


def cluster_phi_draws(frame,source_order,weights):
    counts=np.zeros((len(source_order),4),dtype=np.int64)
    lookup={s:i for i,s in enumerate(source_order)}
    for row in frame.itertuples():
        counts[lookup[row.source_id],2*int(row.E_D)+int(row.I_D)] += 1
    return phi_array(weights @ counts)


def bootstrap_group(frame,source_order,weights):
    draws=cluster_phi_draws(frame,source_order,weights)
    finite=draws[np.isfinite(draws)]
    ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
    return dict(phi_ci_low=ci[0],phi_ci_high=ci[1],bootstrap_valid_draws=len(finite),
                bootstrap_undefined_draws=len(draws)-len(finite),
                bootstrap_ci_status="ok" if len(finite)>=.95*len(draws) else "unstable_many_undefined")


def phi_equality_test(frame,variable,family,source_order,weights,min_sources=20):
    groups=list(frame.groupby(variable,sort=True,observed=True))
    counts={str(level):dict(n_sources=g.source_id.nunique(),n_pairs=len(g)) for level,g in groups}
    record=dict(family=family,variable=variable,primary_or_secondary="primary" if variable=="named_entity" else "secondary",
        null_hypothesis="All subgroup phi-CCT coefficients for this variable are equal",
        formula="Contrasts of subgroup phi against the first observed level",
        test_name="centered source-cluster bootstrap omnibus phi equality test",
        n_sources=frame.source_id.nunique(),n_pairs=len(frame),n_groups=len(groups),group_counts=json.dumps(counts),
        statistic=np.nan,df=max(0,len(groups)-1),p_raw=np.nan,p_bh=np.nan,effect_max_abs_phi_difference=np.nan,
        bootstrap_joint_valid_draws=0,status="non_estimable",nonestimable_reason="")
    if len(groups)<2:
        record["nonestimable_reason"]="fewer_than_two_observed_groups"
        return record,[]
    if min(v["n_sources"] for v in counts.values())<min_sources:
        record["nonestimable_reason"]="sparse_group_below_prespecified_source_threshold"
        return record,[]
    points=np.array([contingency(zip(g.E_D,g.I_D))["phi"] for _,g in groups])
    if not np.isfinite(points).all():
        record["nonestimable_reason"]="undefined_subgroup_phi_constant_E_D_or_I_D"
        return record,[]
    draws=np.column_stack([cluster_phi_draws(g,source_order,weights) for _,g in groups])
    draws=draws[np.isfinite(draws).all(axis=1)]
    record["bootstrap_joint_valid_draws"]=len(draws)
    if len(draws)<.95*len(weights):
        record["nonestimable_reason"]="more_than_5_percent_joint_undefined_bootstrap_draws"
        return record,[]
    delta=points[1:]-points[0]
    boot_delta=draws[:,1:]-draws[:,[0]]
    covariance=np.atleast_2d(np.cov(boot_delta,rowvar=False))
    if np.linalg.matrix_rank(covariance)!=len(delta):
        record["nonestimable_reason"]="rank_deficient_bootstrap_contrast_covariance"
        return record,[]
    inverse=np.linalg.inv(covariance)
    statistic=float(delta@inverse@delta)
    centered=boot_delta-delta
    null_statistics=np.einsum("bi,ij,bj->b",centered,inverse,centered)
    p=(1+np.count_nonzero(null_statistics>=statistic))/(1+len(draws))
    contrasts=[]
    for j,(level,_) in enumerate(groups[1:]):
        lo,hi=np.quantile(boot_delta[:,j],[.025,.975])
        contrasts.append(dict(family=family,variable=variable,term=f'{level} minus {groups[0][0]}',
                         effect=delta[j],ci_low=lo,ci_high=hi,scale="phi_difference"))
    record.update(status="estimable",statistic=statistic,p_raw=p,effect_max_abs_phi_difference=float(np.abs(delta).max()))
    return record,contrasts


def make_pairs(sample,edits,raw,layouts):
    indexed={(r["source_id"],r["edit_id"],int(r["layout_id"])):r for r in raw.to_dict("records")}
    if len(indexed)!=len(raw):
        raise ValueError("Duplicate source/edit/layout outputs; select one run configuration")
    gold=sample.set_index("source_id").gold_semantic_label.to_dict()
    pairs=[]
    for e in edits.to_dict("records"):
        for layout_id,layout in enumerate(layouts):
            b=indexed.get((e["source_id"],"BASELINE",layout_id))
            x=indexed.get((e["source_id"],e["edit_id"],layout_id))
            record={**e,"layout_id":layout_id,"baseline_job_id":b["job_id"] if b else "",
                    "edited_job_id":x["job_id"] if x else "","gold_label_option_position":layout.index(gold[e["source_id"]])+1,
                    "E_D":np.nan,"I_D":np.nan,"exact_mention":np.nan,"stem_mention":np.nan,"paper_rule_mention":np.nan,
                    "validity_status":"invalid","exclusion_reason":"",
                    "baseline_explanation":b.get("explanation","") if b else "",
                    "edited_explanation":x.get("explanation","") if x else "",
                    "baseline_semantic_label":b.get("semantic_label","") if b else "",
                    "edited_semantic_label":x.get("semantic_label","") if x else ""}
            record.update(baseline_raw_response=b.get("raw_response","") if b else "",
                          edited_raw_response=x.get("raw_response","") if x else "",
                          raw_csv_file="outputs/raw/full_raw_outputs.csv")
            if b is None or x is None:
                record["exclusion_reason"]="missing_baseline_or_edited_job"
            elif b["parse_status"]!="ok" or x["parse_status"]!="ok":
                record["exclusion_reason"]=f'baseline:{b["parse_status"]};edited:{x["parse_status"]}'
            else:
                explanation=x["explanation"] if e["family"]=="insertion" else b["explanation"]
                record.update(mention_flags(explanation,e["target"]))
                record.update(I_D=int(b["semantic_label"]!=x["semantic_label"]),validity_status="valid")
            pairs.append(record)
    return pd.DataFrame(pairs)


def interaction_test(frame,variable,family,min_sources=20):
    counts=frame.groupby(variable,observed=True).agg(n_sources=("source_id","nunique"),n_pairs=("source_id","size"))
    record=dict(family=family,variable=variable,primary_or_secondary="primary" if variable=="named_entity" else "secondary",
        null_hypothesis="Mention-impact risk difference is equal across variable levels",
        formula=f'I_D ~ E_D * C({variable})',test_name="source-cluster-robust omnibus interaction Wald chi-square",
        n_sources=frame.source_id.nunique(),n_pairs=len(frame),n_groups=len(counts),
        group_counts=counts.to_json(orient="index"),statistic=np.nan,df=np.nan,p_raw=np.nan,p_bh=np.nan,
        interaction_effect_max_abs=np.nan,status="non_estimable",nonestimable_reason="")
    contrasts=[]
    if len(counts)<2:
        record["nonestimable_reason"]="fewer_than_two_observed_groups"
        return record,contrasts
    if counts.n_sources.min()<min_sources:
        record["nonestimable_reason"]="sparse_group_below_prespecified_source_threshold"
        return record,contrasts
    if any(g.E_D.nunique()<2 or g.I_D.nunique()<2 for _,g in frame.groupby(variable,observed=True)):
        record["nonestimable_reason"]="constant_E_D_or_I_D_in_at_least_one_group"
        return record,contrasts
    try:
        y,X=patsy.dmatrices(record["formula"],frame,return_type="dataframe")
        if np.linalg.matrix_rank(X)!=X.shape[1]:
            raise ValueError("rank_deficient_design")
        fit=sm.OLS(y,X).fit(cov_type="cluster",cov_kwds={"groups":frame.loc[X.index,"source_id"],"use_correction":True})
        terms=[i for i,name in enumerate(X.columns) if ":" in name and "E_D" in name]
        R=np.eye(X.shape[1])[terms]
        if not len(terms) or np.linalg.matrix_rank(R@fit.cov_params()@R.T)<len(terms):
            raise ValueError("rank_deficient_interaction_covariance")
        test=fit.wald_test(R,use_f=False,scalar=True)
        p=float(test.pvalue)
        if not np.isfinite(p): raise ValueError("nonfinite_test")
        ci=fit.conf_int()
        for i in terms:
            contrasts.append(dict(family=family,variable=variable,term=X.columns[i],effect=float(fit.params.iloc[i]),
                       ci_low=float(ci.iloc[i,0]),ci_high=float(ci.iloc[i,1]),scale="probability_point_interaction_relative_to_reference"))
        record.update(statistic=float(test.statistic),df=len(terms),p_raw=p,status="estimable",
                      interaction_effect_max_abs=max(abs(c["effect"]) for c in contrasts))
    except (ValueError,np.linalg.LinAlgError) as exc:
        record["nonestimable_reason"]=str(exc)
    return record,contrasts


def bh_planned_family(tests,m=10):
    valid=tests.p_raw.dropna().sort_values()
    if len(valid):
        adjusted=np.minimum.accumulate((valid.to_numpy()*m/np.arange(1,len(valid)+1))[::-1])[::-1]
        tests.loc[valid.index,"p_bh"]=np.minimum(1,adjusted)
    return tests


def export_coverage(sample,proposals,edits,raw,pairs,root):
    records=[]
    for sid in sample.source_id:
        er=edits[edits.source_id==sid]; rr=raw[raw.source_id==sid]; pr=pairs[pairs.source_id==sid]
        br=rr[rr.edit_id=="BASELINE"]
        done=rr[rr.edit_id!="BASELINE"]
        planned=3*(1+len(er))
        records.append(dict(source_id=sid,expected_baseline_jobs=3,expected_edited_jobs=3*len(er),
            minimum_target_edited_jobs=6,completed_baseline_jobs=len(br),completed_edited_jobs=len(done),
            completed_valid_baselines=int((br.parse_status=="ok").sum()),
            selected_edits=len(er),screened_insertion_edits=int((er.family=="insertion").sum()),
            screened_deletion_edits=int((er.family=="deletion").sum()),
            valid_insertion_edits=pr[(pr.family=="insertion")&(pr.validity_status=="valid")].edit_id.nunique(),
            valid_deletion_edits=pr[(pr.family=="deletion")&(pr.validity_status=="valid")].edit_id.nunique(),insufficient_edits=len(er)<2,
            rejected_proposals=int(((proposals.source_id==sid)&(proposals.selection_status=="rejected")).sum()),
            generation_failures=int((rr.generation_status!="ok").sum()),parsing_failures=int(((rr.generation_status=="ok")&(rr.parse_status!="ok")).sum()),
            valid_pairs=int((pr.validity_status=="valid").sum()),
            completion_status="complete_with_insufficient_edits" if len(rr)==planned and len(er)<2 else "complete" if len(rr)==planned else "incomplete"))
    coverage=pd.DataFrame(records)
    coverage.to_csv(root/"outputs/results/coverage.csv",index=False)
    return coverage


def analyze(root,config,mode="full"):
    root=Path(root)
    sample=pd.read_csv(root/"data/frozen/frozen_sample.csv",keep_default_na=False)
    edits=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False)
    proposals=pd.read_csv(root/"data/edits/edit_proposals.csv",keep_default_na=False)
    audit=pd.read_csv(root/"data/audit/edit_audit.csv",keep_default_na=False)
    rejected=set(audit.loc[audit.decision=="reject","edit_id"])
    exclusions=root/"data/audit/quality_exclusions.csv"
    if exclusions.exists(): rejected.update(pd.read_csv(exclusions,keep_default_na=False).edit_id)
    edits=edits[~edits.edit_id.isin(rejected)]
    manifest=json.loads((root/f"outputs/{mode}_run_manifest.json").read_text())
    path=root/manifest["checkpoint"]
    raw=pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    if raw.empty: raise ValueError("No raw model responses to analyze")
    if raw.config_hash.nunique()!=1: raise ValueError("Mixed inference configurations")
    raw["explanation_word_count"]=[len(str(r.explanation).split()) if r.parse_status=="ok" else np.nan for r in raw.itertuples()]
    raw.to_csv(root/f"outputs/raw/{mode}_raw_outputs.csv",index=False)
    verbosity=[]
    for family,g in raw[raw.parse_status=="ok"].groupby("family"):
        limit=config.get("requested_explanation_max_words")
        verbosity.append(dict(family=family,n_responses=len(g),word_count_rule="whitespace-separated words",
          requested_max_words=limit,mean_words=g.explanation_word_count.mean(),median_words=g.explanation_word_count.median(),
          max_words=g.explanation_word_count.max(),over_requested_limit=int((g.explanation_word_count>limit).sum()) if limit else np.nan))
    pd.DataFrame(verbosity).to_csv(root/"outputs/results/verbosity.csv",index=False)
    pairs=make_pairs(sample,edits,raw,config["option_layouts"])
    pairs.to_csv(root/"outputs/results/paired_metrics.csv",index=False)
    coverage=export_coverage(sample,proposals,edits,raw,pairs,root)
    valid=pairs[pairs.validity_status=="valid"].copy()
    valid[["E_D","I_D"]]=valid[["E_D","I_D"]].astype(int)
    weights=np.random.default_rng(config["bootstrap_seed"]).multinomial(len(sample),np.ones(len(sample))/len(sample),size=config["bootstrap_draws"])
    summary,estimates,tests,contrasts=[],[],[],[]
    baselines=raw[raw.edit_id=="BASELINE"]
    vb=baselines[baselines.parse_status=="ok"]
    accuracy=dict(valid_baseline_n=len(vb),scheduled_baseline_n=3*len(sample),
        valid_baseline_accuracy=float((vb.semantic_label==vb.gold_semantic_label).mean()) if len(vb) else np.nan,
        full_coverage_correct_rate=int((vb.semantic_label==vb.gold_semantic_label).sum())/(3*len(sample)),
        baseline_invalid_or_missing=3*len(sample)-len(vb))
    pd.DataFrame([accuracy]).to_csv(root/"outputs/results/accuracy.csv",index=False)
    for family in ("insertion","deletion"):
        f=valid[valid.family==family]
        summary.append(dict(family=family,n_sources=f.source_id.nunique(),**contingency(zip(f.E_D,f.I_D)),**bootstrap_group(f,sample.source_id.tolist(),weights),**accuracy,
                            phi_paper_rule=contingency(zip(f.paper_rule_mention,f.I_D))["phi"],
                            phi_exact_match=contingency(zip(f.exact_mention,f.I_D))["phi"]))
        for variable,levels in VARIABLES.items():
            test,cs=phi_equality_test(f,variable,family,sample.source_id.tolist(),weights,config["minimum_subgroup_sources_for_inference"])
            tests.append(test);contrasts.extend(cs)
            for level in levels:
                g=f[f[variable]==level]
                estimates.append(dict(family=family,variable=variable,level=level,n_sources=g.source_id.nunique(),
                   sparse=g.source_id.nunique()<config["minimum_subgroup_sources_for_inference"],
                   **contingency(zip(g.E_D,g.I_D)),**bootstrap_group(g,sample.source_id.tolist(),weights)))
    pd.DataFrame(summary).to_csv(root/"outputs/results/summary.csv",index=False)
    estimates=pd.DataFrame(estimates)
    estimates.to_csv(root/"outputs/results/plot_estimates.csv",index=False)
    tests=bh_planned_family(pd.DataFrame(tests),m=10)
    tests.to_csv(root/"outputs/results/tests.csv",index=False)
    pd.DataFrame(contrasts,columns=["family","variable","term","effect","ci_low","ci_high","scale"]).to_csv(root/"outputs/results/phi_contrasts.csv",index=False)
    for variable,levels in VARIABLES.items():
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,family in zip(axes,("insertion","deletion")):
            table=estimates[(estimates.family==family)&(estimates.variable==variable)]
            for k,row in enumerate(table.itertuples()):
                if np.isfinite(row.phi):
                    ax.plot(k,row.phi,"o",color="#28777b")
                    if np.isfinite(row.phi_ci_low): ax.vlines(k,row.phi_ci_low,row.phi_ci_high,color="#28777b",lw=2)
                else: ax.text(k,0,"undefined",ha="center",rotation=30,fontsize=8)
            ax.set_xticks(range(len(levels)),[f'{r.level}\nS={r.n_sources}, P={r.n_pairs}'+("\nsparse" if r.sparse else "")+("\nunstable CI" if r.bootstrap_ci_status!="ok" else "") for r in table.itertuples()])
            ax.set_title(family);ax.set_ylim(-1.08,1.08);ax.axhline(0,color="grey",lw=.6);ax.grid(axis="y",alpha=.2)
        axes[0].set_ylabel("phi-CCT (95% source-cluster bootstrap CI)")
        fig.suptitle(DISPLAY[variable]);fig.tight_layout()
        fig.savefig(root/f"outputs/figures/{variable}.png",dpi=180)
        fig.savefig(root/f"outputs/figures/{variable}.pdf")
        plt.close(fig)
    role_estimates=[]
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for row_index,role in enumerate(("premise","hypothesis")):
        for column_index,family in enumerate(("insertion","deletion")):
            ax=axes[row_index,column_index]
            for k,level in enumerate(VARIABLES["position_bin"]):
                g=valid[(valid.family==family)&(valid.role==role)&(valid.position_bin==level)]
                estimate=dict(family=family,role=role,level=level,n_sources=g.source_id.nunique(),
                    **contingency(zip(g.E_D,g.I_D)),**bootstrap_group(g,sample.source_id.tolist(),weights))
                role_estimates.append(estimate)
                if np.isfinite(estimate["phi"]):
                    ax.plot(k,estimate["phi"],"o",color="#28777b")
                    if np.isfinite(estimate["phi_ci_low"]):ax.vlines(k,estimate["phi_ci_low"],estimate["phi_ci_high"],color="#28777b")
                else:ax.text(k,0,"undefined",fontsize=8,ha="center")
            panel=role_estimates[-3:]
            ax.set_xticks(range(3),[f'{e["level"]}\nS={e["n_sources"]}, P={e["n_pairs"]}' for e in panel])
            ax.set_title(f'{family} / {role}');ax.set_ylim(-1.08,1.08);ax.axhline(0,color="grey",lw=.6);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Target position by edited role (95% source-cluster bootstrap CI)")
    fig.tight_layout();fig.savefig(root/"outputs/figures/position_by_role.png",dpi=180);plt.close(fig)
    pd.DataFrame(role_estimates).to_csv(root/"outputs/results/position_by_role.csv",index=False)
    # Preserve availability, avoid manufacturing missing discordance directions.
    qualitative=[];used=set();availability=[]
    for e,i in ((0,1),(1,0),(1,1),(0,0)):
        group=valid[(valid.E_D==e)&(valid.I_D==i)]
        availability.append(dict(case_type=f'E{e}_I{i}',n_pairs=len(group),available=bool(len(group))))
        for _,r in group.groupby("family",sort=True).head(1).iterrows():
            if r.source_id in used: continue
            d=r.to_dict();d.update(case_type=f'E{e}_I{i}',interpretation="REVIEW_REQUIRED",alternative_explanation="REVIEW_REQUIRED")
            qualitative.append(d);used.add(r.source_id)
    failures=pairs[pairs.validity_status!="valid"]
    if len(failures):
        d=failures.iloc[0].to_dict();d.update(case_type="failure",interpretation="Invalid pair excluded from metric; inspect exclusion_reason",alternative_explanation="Missing or malformed response is not a negative mention/impact.")
        qualitative.append(d)
    elif exclusions.exists():
        excluded=pd.read_csv(exclusions,keep_default_na=False).iloc[0]
        source_edits=pd.read_csv(root/"data/edits/selected_edits.csv",keep_default_na=False)
        failed=source_edits[source_edits.edit_id==excluded.edit_id]
        if len(failed):
            d=failed.iloc[0].to_dict()
            d.update(case_type="rejected_intervention",E_D=np.nan,I_D=np.nan,layout_id=0,
                     validity_status="rejected_before_inference",exclusion_reason=excluded.reason,
                     baseline_job_id="",edited_job_id="",baseline_explanation="",edited_explanation="",
                     baseline_semantic_label="",edited_semantic_label="",
                     interpretation="Dependency parsing produced an incomplete verb complement; excluded before study inference.",
                     alternative_explanation="A malformed edit must not be treated as evidence about explanation faithfulness.")
            qualitative.append(d)
    for _,r in valid.iterrows():
        if len(qualitative)>=5: break
        if r.source_id not in used:
            d=r.to_dict();d.update(case_type="additional_trace",interpretation="REVIEW_REQUIRED",alternative_explanation="REVIEW_REQUIRED")
            qualitative.append(d);used.add(r.source_id)
    qualitative=pd.DataFrame(sorted(qualitative,key=lambda r:0 if r["case_type"] in {"failure","rejected_intervention"} else 1))
    qualitative["example_id"]=[f'Q{i+1}' for i in range(len(qualitative))]
    review_path=root/"outputs/qualitative/reviewed_qualitative.csv"
    if review_path.exists():
        reviewed=pd.read_csv(review_path,keep_default_na=False)
        def review_key(row):
            return (row["edit_id"],int(float(row["layout_id"])),row["baseline_job_id"],row["edited_job_id"])
        lookup={review_key(row):row for row in reviewed.to_dict("records")}
        existing_keys={review_key(row) for row in qualitative.to_dict("records")}
        pair_lookup={review_key(row):row for row in pairs.to_dict("records")}
        additions=[]
        for key,row in lookup.items():
            if key not in existing_keys and key in pair_lookup:
                additions.append({**pair_lookup[key],"example_id":row["example_id"],"case_type":row["case_type"],
                                  "interpretation":"REVIEW_REQUIRED","alternative_explanation":"REVIEW_REQUIRED"})
        if additions:qualitative=pd.concat([qualitative,pd.DataFrame(additions)],ignore_index=True)
        for index,row in qualitative.iterrows():
            match=lookup.get(review_key(row))
            if match is not None:
                for col in ("interpretation","alternative_explanation","reviewer","reviewed_at_utc"):
                    if col in match:qualitative.at[index,col]=match[col]
    qualitative.to_csv(root/"outputs/qualitative/qualitative_examples.csv",index=False)
    pd.DataFrame(availability).to_csv(root/"outputs/qualitative/case_availability.csv",index=False)
    status=dict(status="complete" if mode=="full" and not coverage.completion_status.eq("incomplete").any() else "incomplete",
       mode=mode,n_frozen_sources=len(sample),n_sources_with_valid_pairs=valid.source_id.nunique(),
       expected_jobs=manifest["expected_jobs"],completed_jobs=len(raw),valid_pairs=len(valid),
       sources_with_insufficient_edits=int(coverage.insufficient_edits.sum()),planned_tests=len(tests),
       tests_estimable=int(tests.status.eq("estimable").sum()),bootstrap_draws=config["bootstrap_draws"],
       bootstrap_unit="source_id, all 1000 frozen clusters; repeated layouts and edits kept together",
       undefined_draws="omitted from percentile calculation; counts and unstable status exported",
       bh_family_size=10,qualitative_review="completed_with_analyst_review" if not qualitative.interpretation.str.contains("REVIEW_REQUIRED").any() else "pending")
    (root/"outputs/results/analysis_manifest.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))
    return status


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["full","smoke"],default="full");a=p.parse_args()
    analyze(ROOT,json.loads((ROOT/"config/experiment.json").read_text()),a.mode)
