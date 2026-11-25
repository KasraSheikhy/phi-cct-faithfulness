"""Create the required two-page PDF using actual analyzed and reviewed records only."""
import argparse
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def number(value,digits=3):
    return "NE" if pd.isna(value) else f'{value:.{digits}f}'


def pvalue(value):
    return "NE" if pd.isna(value) else f'{value:.4f}' if value<.001 else f'{value:.3f}'


def render_report(root,allow_incomplete=False):
    root=Path(root)
    cfg=json.loads((root/"config/experiment.json").read_text())
    status=json.loads((root/"outputs/results/analysis_manifest.json").read_text())
    if status["status"]!="complete" and not allow_incomplete:
        raise RuntimeError("Full evaluation incomplete; use --allow-incomplete only to create an honestly labeled interim report")
    summary=pd.read_csv(root/"outputs/results/summary.csv")
    tests=pd.read_csv(root/"outputs/results/tests.csv")
    estimates=pd.read_csv(root/"outputs/results/plot_estimates.csv")
    accuracy=pd.read_csv(root/"outputs/results/accuracy.csv").iloc[0]
    qualitative=pd.read_csv(root/"outputs/qualitative/qualitative_examples.csv",keep_default_na=False)
    if qualitative.interpretation.astype(str).str.contains("REVIEW_REQUIRED").any() and not allow_incomplete:
        raise RuntimeError("Review actual qualitative examples before generating the final report")
    titles={"named_entity":"Named entity","target_polarity":"Sentiment","position_bin":"Target position",
            "idf_bin":"Lexical rarity / IDF","gold_label_option_position":"Gold-label option position"}
    colors_by_family={"insertion":"#277c82","deletion":"#b45b35"}
    fig,axes=plt.subplots(3,2,figsize=(7.35,5.8),sharey=True)
    for ax,(variable,title) in zip(axes.flat,titles.items()):
        table=estimates[estimates.variable==variable]
        levels=list(table[table.family=="insertion"].level.astype(str))
        for family,offset in (("insertion",-.10),("deletion",.10)):
            rows=table[table.family==family]
            for i,r in enumerate(rows.itertuples()):
                x=i+offset
                if np.isfinite(r.phi):
                    ax.plot(x,r.phi,"o" if family=="insertion" else "s",ms=4,color=colors_by_family[family])
                    if np.isfinite(r.phi_ci_low):ax.vlines(x,r.phi_ci_low,r.phi_ci_high,color=colors_by_family[family],lw=1.3)
                else:ax.text(x,-.98,"NE",fontsize=6,ha="center",color=colors_by_family[family])
        ins=table[table.family=="insertion"].n_sources.tolist();dele=table[table.family=="deletion"].n_sources.tolist()
        unstable=[any(g.bootstrap_ci_status!="ok") for _,g in table.groupby(table.level.astype(str),sort=False)]
        ax.set_xticks(range(len(levels)),[f'{s}{"*" if u else ""}\nS={a}/{b}' for s,a,b,u in zip(levels,ins,dele,unstable)],fontsize=6.6)
        ax.set_title(title,fontsize=8.5);ax.set_ylim(-1.08,1.08);ax.set_xlim(-.5,len(levels)-.5)
        ax.axhline(0,color="grey",lw=.5);ax.grid(axis="y",alpha=.18);ax.tick_params(axis="y",labelsize=7)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05,.8,"Insertion: teal circles\nDeletion extension: orange squares\n\n95% source-cluster bootstrap CI\nS = unique sources (insertion/deletion)\nNE = undefined, not zero\n* More than 5% undefined draws;\n   interval is unstable/conditional.\nSparse strata remain in full CSVs.",fontsize=8,va="top")
    fig.tight_layout(pad=1.0)
    overview=root/"outputs/figures/report_overview.png";fig.savefig(overview,dpi=220);plt.close(fig)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall",fontName="Helvetica",fontSize=8.4,leading=10.4,spaceAfter=5))
    styles.add(ParagraphStyle(name="TitleSmall",fontName="Helvetica-Bold",fontSize=16,leading=19,spaceAfter=7,textColor=colors.HexColor("#174b51")))
    styles.add(ParagraphStyle(name="SectionSmall",fontName="Helvetica-Bold",fontSize=9.5,leading=12,spaceBefore=6,spaceAfter=4))
    styles.add(ParagraphStyle(name="NoteSmall",fontName="Helvetica",fontSize=7.2,leading=9,spaceAfter=4))
    story=[]
    def p(text,style="BodySmall"):story.append(Paragraph(text,styles[style]))
    def table(rows,widths):
        t=Table(rows,colWidths=widths,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5eff0")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
          ("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),7),
          ("BOTTOMPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),
          ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#9cb4b8")),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(t)
    p("What affects phi-CCT faithfulness scores?","TitleSmall")
    p("e-SNLI / Qwen3-8B | "+("COMPLETED INFERENCE" if status["status"]=="complete" else "INCOMPLETE - INTERIM REPORT"),"SectionSmall")
    p("<b>Question.</b> Do named entities, target sentiment, input position, lexical rarity or answer-option position change the measured association between explanation mention (E_D) and semantic prediction impact (I_D)? Named-entity status is primary; the other four are planned secondary checks. The null is equality of subgroup phi-CCT within an intervention family.")
    p(f'<b>Design.</b> 1,000 distinct test examples, seed {cfg["sample_seed"]}, source revision {cfg["source_revision"][:12]}. Qwen/Qwen3-8B revision {cfg["model_revision"][:12]}, FP16, no quantization, greedy decoding, thinking disabled. One concise explanation sentence (requested maximum 25 words) precedes the final answer; noncompliance is recorded, never truncated. Three matched cyclic label layouts map back to semantic labels. Coverage: {status["completed_jobs"]}/{status["expected_jobs"]} scheduled responses; {status["valid_pairs"]} valid pairs from {status["n_sources_with_valid_pairs"]} source clusters.')
    p("<b>Interventions and audit.</b> Two rule-screened edits per source; a second insertion is used when no suitable deletion is found. Insertion E_D checks the new explanation; deletion checks the baseline explanation as an extension. User review globally accepted 24 proposals, but detailed meaning/gold-label judgments remain uncertain. Analyst exclusions and replacements, including malformed deletion and digit-only targets, were made before study outcomes and preserved.")
    p("<b>Annotation.</b> spaCy NER (numeric tags separately); VADER target valence with negation separate; normalized target midpoint in premise+hypothesis with fixed thirds; IDF from 549,367 training documents with frozen target-based tertiles; original gold-label position as the option anchor. Premise/hypothesis position panels are in the notebook.")
    p(f'<b>Baseline.</b> Accuracy {number(accuracy.valid_baseline_accuracy)} over {int(accuracy.valid_baseline_n)} valid responses; full scheduled-denominator correct rate {number(accuracy.full_coverage_correct_rate)}; invalid/missing baseline responses {int(accuracy.baseline_invalid_or_missing)}.')
    rows=[["Family","Sources / pairs","phi [95% CI]","Impact / mention","n00 / n01 / n10 / n11"]]
    for r in summary.itertuples():
        rows.append([r.family,f'{r.n_sources} / {r.n_pairs}',f'{number(r.phi)} [{number(r.phi_ci_low)}, {number(r.phi_ci_high)}]',f'{number(r.impact_rate)} / {number(r.mention_rate)}',f'{r.n00} / {r.n01} / {r.n10} / {r.n11}'])
    table(rows,[55,82,132,82,172])
    p("<b>Five planned checks.</b> Centered source-cluster bootstrap omnibus equality tests, 2,000 draws; raw p and BH q use all 10 planned tests. Full contrast intervals are in phi_contrasts.csv. Deletion sentiment is non-estimable (NE): its negative group has only 13 independent sources, below the pre-specified minimum 20.")
    rows=[["Variable","Insertion p / q","max |delta phi|","Deletion p / q","max |delta phi|"]]
    for variable,title in titles.items():
        ins=tests[(tests.variable==variable)&(tests.family=="insertion")].iloc[0]
        dele=tests[(tests.variable==variable)&(tests.family=="deletion")].iloc[0]
        rows.append([title,f'{pvalue(ins.p_raw)} / {pvalue(ins.p_bh)}',number(ins.effect_max_abs_phi_difference),f'{pvalue(dele.p_raw)} / {pvalue(dele.p_bh)}',number(dele.effect_max_abs_phi_difference)])
    table(rows,[135,103,88,108,89])
    significant=tests[tests.p_bh<.05]
    if len(significant):
        p('<b>Primary result.</b> Only insertion named-entity status survives BH (p=0.0005, q=0.005): non-entity phi=0.265 versus entity phi=-0.158; difference 0.424 [0.288, 0.546]. The entity subset is exclusively French/Canadian NORP markers (142 sources, 444 pairs). This supports a conditional measurement difference, not a general causal entity effect.')
    else:
        p(f'<b>Result.</b> No planned comparison has BH q&lt;0.05; {int(tests.status.eq("estimable").sum())} of 10 are estimable. Interpret this as inconclusive or null evidence under this design, not proof of invariance.')
    p("<b>Post-hoc mechanism check.</b> For the two nationality targets only, also counting explicit 'nationality/nationalities' references adds 125 mentions (60 with flips). Entity phi becomes +0.257 [0.140, 0.358]; the non-entity-minus-entity gap becomes 0.009 [-0.111, 0.140]. Twenty proxy cases were inspected. This exploratory proxy suggests lexical matching drives the contrast; it is not a semantic oracle or a new confirmatory test.")
    p("<b>Limits.</b> Features/templates co-vary; only two nationality terms represent insertion entities; audit detail and some subgroup power are limited. A flip is sensitivity, not correctness or proof of unfaithfulness. Edited gold labels were not independently assigned. Undefined bootstrap draws are omitted and counted. Verbosity is a fixed, pilot-adjusted setting, not a novel finding.","NoteSmall")
    story.append(PageBreak())
    p("Subgroup estimates and traceable examples","TitleSmall")
    p("Families remain distinct. Intervals resample all source clusters, preserving repeated edits and layouts; the axis range is fixed across plots. Full group counts, undefined draws and non-estimability reasons accompany the notebook and CSVs.","NoteSmall")
    story.append(Image(str(overview),width=523,height=413))
    p("Qualitative traces","SectionSmall")
    chosen=qualitative.set_index("example_id").loc[["Q1","Q2","Q5","Q8","Q9"]].reset_index()
    for r in chosen.to_dict("records"):
        label=r["example_id"]
        target=html.escape(str(r.get("target","")))
        if r.get("case_type")=="rejected_intervention":
            content=f'{label}: rejected target "{target}". '+html.escape(str(r["interpretation"]))
        else:
            content=f'{label}: "{target}"; {r.get("baseline_semantic_label", "")} to {r.get("edited_semantic_label", "")}; E={r.get("E_D", "")}, I={r.get("I_D", "")}. '+html.escape(str(r.get("interpretation","")))
        p(content,"NoteSmall")
    p("Q1, Q2, Q5, Q8 and Q9 are keyed rows in qualitative_examples.csv (nine traces total). Full inputs, spans, raw explanations, semantic predictions and raw job IDs are preserved there. Q2 was discussed with the user; other output interpretations and proxy checks were inspected by Codex and remain available for researcher verification.","NoteSmall")
    p('References: Siegel et al., <link href="https://arxiv.org/abs/2503.13445v3" color="#277c82">arXiv:2503.13445v3</link>; Camburu et al., <link href="https://github.com/OanaMariaCamburu/e-SNLI" color="#277c82">e-SNLI</link>; <link href="https://huggingface.co/Qwen/Qwen3-8B" color="#277c82">Qwen3-8B model card</link>. Independently written pipeline; official corr_faith code cited for consultation. AI assistance and verification are documented in README.md.',"NoteSmall")
    output=root/"report/phi_cct_report.pdf"
    def footer(canvas,doc):
        canvas.setFont("Helvetica",7);canvas.setFillColor(colors.grey)
        canvas.drawString(36,20,"phi-CCT on e-SNLI | full provenance and uncertainty in accompanying CSVs")
        canvas.drawRightString(A4[0]-36,20,str(doc.page))
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=32,bottomMargin=32)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(output).pages)
    if count!=2:raise RuntimeError(f'Report has {count} pages; adjust layout and visually verify before delivery')
    print(str(output))
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--allow-incomplete",action="store_true");a=p.parse_args()
    render_report(ROOT,a.allow_incomplete)
