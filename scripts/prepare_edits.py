"""Outcome-blind, deterministic edits and feature annotations for every frozen source."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = set("man woman person people boy girl child children men women kid kids teenager adult worker player runner skier biker cyclist farmer cook chef soldier officer dancer musician singer crowd couple family pedestrian hiker baby toddler students student teacher".split())
ANIMALS = set("dog cat horse bird puppy kitten cow elephant animal duck pig bear monkey tiger lion sheep goat fish rabbit dogs cats horses birds animals".split())
CLOTHING = set("shirt pants coat jacket dress hat cap sweater shorts jeans skirt suit shoe shoes boot boots glove gloves uniform helmet scarf apron".split())
OBJECTS = set("car truck vehicle bus bike bicycle motorcycle boat train plane airplane building house room table chair bench ball bag box door window cup bottle sign umbrella surfboard skateboard sled tractor street road wall fence pole basket book backpack tent bridge tower cart machine canister".split())
ACTION_VERBS = set("walk run move dance ride climb swim jump play work cook eat drink speak talk drive sing watch read write carry sweep skate ski hike cross reach enter leave".split())
NAMED_ENTITY_TYPES = {"PERSON","ORG","GPE","LOC","FAC","NORP","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE"}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def proposals_for(doc, role):
    candidates = []
    for t in doc:
        choices = []
        if t.pos_ == "NOUN":
            if t.lemma_.lower() in PEOPLE:
                choices = ["happy", "angry", "calm", "French", "Canadian"]
            elif t.lemma_.lower() in ANIMALS:
                choices = ["small", "aggressive", "playful", "quiet"]
            elif t.lemma_.lower() in CLOTHING:
                choices = ["red", "blue", "dirty", "clean"]
            elif t.lemma_.lower() in OBJECTS:
                choices = ["red", "old", "new", "large", "damaged"]
            else:
                choices = ["strange", "unusual", "ordinary"]
        if choices:
            # Insert before existing adjectival modifiers to preserve adjective/noun chunks.
            left = min([t.i] + [x.left_edge.i for x in t.children if x.dep_ in {"amod","compound"} and x.lower_ not in {"other","same","such"}])
            offset = doc[left].idx
            for target in choices:
                if left and doc[left-1].lower_ == "a" and target[0].lower() in "aeiou":
                    continue  # changing a->an would be a second, non-insertion edit
                candidates.append(dict(family="insertion", role=role, method="noun_modifier",
                                       original_span_start=offset, original_span_end=offset,
                                       replacement=target + " ", target=target,
                                       edited_span_start=offset, edited_span_end=offset+len(target)))
        if t.pos_ == "VERB" and t.lemma_.lower() in ACTION_VERBS and t.dep_ != "amod":
            # Before the lexical verb, after any auxiliary: "is slowly walking".
            for target in ("slowly", "quickly", "happily", "angrily", "quietly"):
                candidates.append(dict(family="insertion", role=role, method="verb_modifier",
                                       original_span_start=t.idx, original_span_end=t.idx,
                                       replacement=target + " ", target=target,
                                       edited_span_start=t.idx, edited_span_end=t.idx+len(target)))
        deletable = t.dep_ == "amod" and t.pos_ == "ADJ"
        deletable |= t.dep_ == "advmod" and t.pos_ == "ADV" and t.lower_ not in {
            "not", "n't", "there", "here", "how", "when", "where", "why", "as", "too", "so"}
        deletable |= t.dep_ == "nummod" and t.head.tag_ == "NNS"
        if deletable and not any(x.dep_ == "punct" for x in t.subtree):
            subtree = list(t.subtree)
            if max(x.i for x in subtree)-min(x.i for x in subtree)+1 == len(subtree):
                first, last = min(subtree,key=lambda x:x.i), max(subtree,key=lambda x:x.i)
                start, end = first.idx, last.idx + len(last)
                if end < len(doc.text) and doc.text[end:end+1].isspace():
                    candidates.append(dict(family="deletion", role=role, method="modifier_deletion",
                                           original_span_start=start, original_span_end=end+1,
                                           replacement="", target=doc.text[start:end],
                                           edited_span_start=start, edited_span_end=start))
        # Optional PP adjuncts, avoiding common required verb complements and partitives.
        if t.dep_ == "prep" and t.lower_ in {"with", "in", "on", "near", "under", "beside", "behind", "outside", "inside", "through", "along", "during"} and t.head.lemma_.lower() not in {"put", "place", "set", "live", "reside", "belong", "consist", "depend", "rely", "look"}:
            subtree = list(t.subtree)
            first, last = min(subtree,key=lambda x:x.i), max(subtree,key=lambda x:x.i)
            if first.i > 0 and last.i-first.i+1 == len(subtree) and not any(x.is_punct or x.dep_ in {"conj", "cc"} for x in subtree):
                start, end = first.idx, last.idx+len(last)
                if doc.text[start-1].isspace():
                    candidates.append(dict(family="deletion", role=role, method="adjunct_pp_deletion",
                                           original_span_start=start-1, original_span_end=end,
                                           replacement="", target=doc.text[start:end],
                                           edited_span_start=start-1, edited_span_end=start-1))
    return candidates


def annotate(text, target_start, target_end, doc, sia, reference, role, premise, hypothesis):
    target = text[target_start:target_end]
    all_overlapping = sorted({e.label_ for e in doc.ents if e.start_char < target_end and e.end_char > target_start})
    overlapping = [label for label in all_overlapping if label in NAMED_ENTITY_TYPES]
    ws = list(re.finditer(r"\S+", text))
    selected = [i for i,m in enumerate(ws) if m.start() < target_end and m.end() > target_start]
    if not selected:
        raise ValueError("Target has no tokens")
    within = (selected[0]+selected[-1]+1)/2
    premise_n, hypothesis_n = len(premise.split()), len(hypothesis.split())
    normalized = ((premise_n if role == "hypothesis" else 0)+within)/(premise_n+hypothesis_n)
    tt = tokens(target)
    scores = [sia.lexicon.get(t, 0.0) for t in tt]
    score = sum(scores)/len(scores) if scores else 0.0
    idfs = [math.log((reference["n_documents"]+1)/(reference["document_frequencies"].get(t,0)+1))+1 for t in tt]
    return dict(named_entity="entity" if overlapping else "non_entity", entity_type="|".join(overlapping) or "NONE", ner_all_types="|".join(all_overlapping) or "NONE",
                target_polarity="positive" if score > 0 else "negative" if score < 0 else "neutral",
                polarity_score=score, context_polarity=sia.polarity_scores(text)["compound"],
                negation_flag=any(t in {"not", "no", "never", "neither", "nor"} for t in tt),
                position_normalized=normalized, position_within_role=within/len(ws),
                position_bin="early" if normalized < 1/3 else "middle" if normalized < 2/3 else "late",
                idf=float(np.mean(idfs)) if idfs else float("nan"), unseen_target=any(t not in reference["document_frequencies"] for t in tt))


def prepare_edits(root, config):
    sample = pd.read_csv(root / "data/frozen/frozen_sample.csv", keep_default_na=False)
    reference = json.loads((root / "data/frozen/idf_reference.json").read_text())
    nltk.data.path.insert(0, str(root / "cache/nltk"))
    sia = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
    rows = sample.to_dict("records")
    docs = list(nlp.pipe([r[role] for r in rows for role in ("premise", "hypothesis")], batch_size=64))
    all_proposals, chosen = [], []
    for n, source in enumerate(rows):
        source_id = source["source_id"]
        cands = [c for j,role in enumerate(("premise", "hypothesis")) for c in proposals_for(docs[2*n+j],role)]
        # Hash ranking is deterministic and independent of evaluated model outputs and gold label.
        cands.sort(key=lambda c: digest(json.dumps([config["sample_seed"], source_id, c],sort_keys=True)))
        selected_families = set()
        for rank, c in enumerate(cands):
            original = source[c["role"]]
            start, end = c["original_span_start"], c["original_span_end"]
            edited = original[:start]+c["replacement"]+original[end:]
            reasons = []
            if not tokens(c["target"]):
                reasons.append("no_alphabetic_target_tokens")
            if edited == original or len(tokens(edited)) < 3:
                reasons.append("unchanged_or_too_short")
            if re.search(r"\b(?:a|an|the)\s*[.!?]", edited, re.I):
                reasons.append("dangling_determiner")
            for article,word in re.findall(r"\b(a|an)\s+([a-z]+)",edited,re.I):
                word=word.lower()
                vowel=word[0] in "aeiou" and not word.startswith(("uni","use","user","usual","euro","one","once"))
                vowel |= word.startswith(("hour","honest","honor","heir"))
                if (article.lower()=="a" and vowel) or (article.lower()=="an" and not vowel):
                    original_pattern=re.search(r"\b"+re.escape(article)+r"\s+"+re.escape(word)+r"\b",original,re.I)
                    if not original_pattern:
                        reasons.append("introduced_article_agreement")
            row = dict(source_id=source_id, edit_id=f'{source_id}:{c["family"]}:{rank:03d}', proposal_rank=rank,
                       **c, original_text=original, edited_text=edited,
                       original_premise=source["premise"], original_hypothesis=source["hypothesis"],
                       edited_premise=edited if c["role"]=="premise" else source["premise"],
                       edited_hypothesis=edited if c["role"]=="hypothesis" else source["hypothesis"])
            row.update(selection_status="rejected" if reasons else "eligible_not_selected", rejection_reason="|".join(reasons))
            if not reasons and c["family"] not in selected_families:
                edoc = nlp(edited)
                reftext = edited if c["family"] == "insertion" else original
                refdoc = edoc if c["family"] == "insertion" else docs[2*n+(c["role"]=="hypothesis")]
                ts = c["edited_span_start"] if c["family"]=="insertion" else original.find(c["target"],start)
                te = ts+len(c["target"])
                assert reftext[ts:te] == c["target"]
                p = row["edited_premise"] if c["family"]=="insertion" else source["premise"]
                h = row["edited_hypothesis"] if c["family"]=="insertion" else source["hypothesis"]
                row.update(target_span_start=ts,target_span_end=te,target_span_reference="edited" if c["family"]=="insertion" else "original")
                row.update(annotate(reftext,ts,te,refdoc,sia,reference,c["role"],p,h))
                row["selection_status"] = "selected_pending_audit"
                chosen.append(row)
                selected_families.add(c["family"])
            all_proposals.append(row)
        if (n+1) % 100 == 0:
            print("Prepared sources",n+1,flush=True)
    edits = pd.DataFrame(chosen)
    # Cut points depend on training-only IDF values of frozen targets, never model outcomes.
    boundaries = np.quantile(edits.idf.dropna(), [1/3,2/3]).tolist()
    edits["idf_bin"] = np.where(edits.idf <= boundaries[0],"common",np.where(edits.idf <= boundaries[1],"middle","rare"))
    all_frame = pd.DataFrame(all_proposals)
    audit_path = root / "data/audit/edit_audit.csv"
    if audit_path.exists():
        prior=pd.read_csv(audit_path,keep_default_na=False)
        if prior.decision.astype(str).str.strip().ne("").any():
            raise RuntimeError("Human decisions exist; do not regenerate proposals or overwrite review")
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive=root/"data/audit"/f"superseded_{stamp}"
        archive.mkdir(parents=True,exist_ok=False)
        for path in (audit_path,root/"data/edits/edit_proposals.csv",root/"data/edits/selected_edits.csv",root/"data/edits/feature_manifest.json",root/"docs/blinded_edit_review.md"):
            if path.exists(): path.rename(archive/path.name)
    root.joinpath("data/edits/edit_proposals.csv").write_text(all_frame.to_csv(index=False))
    edits.to_csv(root / "data/edits/selected_edits.csv", index=False)
    audit_ids = []
    for family in ("insertion", "deletion"):
        pool = edits[edits.family == family].copy()
        # Cover distinct methods/features before filling deterministic random order.
        selected = []
        for col in ("method","named_entity","target_polarity","position_bin","idf_bin","role"):
            for _,group in pool.groupby(col,sort=True):
                available = group[~group.edit_id.isin(selected)]
                if len(available) and len(selected)<12:
                    selected.append(available.iloc[0].edit_id)
        selected += [x for x in pool.edit_id if x not in selected][:12-len(selected)]
        audit_ids += selected
    review = edits.set_index("edit_id").loc[audit_ids].reset_index()
    for col in ("grammaticality", "coherence", "meaning_changed", "gold_label_may_change", "decision", "rejection_reason", "reviewer", "reviewed_at", "notes"):
        review[col] = ""
    review["outcomes_blinded"] = True
    review["audit_number"] = np.arange(1,len(review)+1)
    review.to_csv(audit_path,index=False)
    lines = ["# Blinded edit review — 24 proposals", "", "No Qwen responses have been generated. Review each edit for grammaticality, coherence, meaning change and possible gold-label change. A semantic change can be valid; malformed or incoherent text is not.", "", "Reply using the numbered IDs, for example: `1 accept; meaning changes; gold may change` or `2 reject: dangling modifier`. Uncertain cases should be marked uncertain. Human decisions have not been filled in by AI.", ""]
    for r in review.to_dict("records"):
        lines += [f'## {r["audit_number"]}. {r["family"]} — {r["role"]}',f'ID: `{r["edit_id"]}`',"",f'Premise: {r["original_premise"]}',f'Hypothesis: {r["original_hypothesis"]}',"",f'**Original {r["role"]}:** {r["original_text"]}',f'**Edited {r["role"]}:** {r["edited_text"]}',f'**Target:** `{r["target"]}`; target span [{r["target_span_start"]}, {r["target_span_end"]}) in {r["target_span_reference"]} text.',f'Automatic annotations to check: entity={r["named_entity"]} ({r["entity_type"]}); polarity={r["target_polarity"]}; position={r["position_bin"]}; IDF={r["idf_bin"]}.',"", "Grammaticality: ___; coherence: ___; meaning change: ___; gold label may change: ___; accept/reject/uncertain: ___", ""]
    (root / "docs/blinded_edit_review.md").write_text("\n".join(lines)+"\n")
    features = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"model_outputs_seen":False,
                "ner_model":"en_core_web_sm", "ner_model_version":nlp.meta["version"], "named_entity_types":sorted(NAMED_ENTITY_TYPES),
                "sentiment":"VADER lexicon mean target token valence by sign; context VADER compound separately",
                "idf_bin_rule":"tertiles over training-derived IDF values of all frozen selected targets; <= lower common, <=upper middle, else rare",
                "idf_bin_boundaries":boundaries,"n_sources":len(sample),"n_edits":len(edits),
                "family_counts":edits.family.value_counts().to_dict(),"audit_n":len(review),
                "selected_edits_sha256":digest((root / "data/edits/selected_edits.csv").read_text()),
                "feature_counts":{c:edits.groupby(["family",c]).size().to_dict().__repr__() for c in ["named_entity","target_polarity","position_bin","idf_bin"]}}
    (root / "data/edits/feature_manifest.json").write_text(json.dumps(features,indent=2)+"\n")
    print(json.dumps(features,indent=2),flush=True)


if __name__ == "__main__":
    prepare_edits(ROOT,json.loads((ROOT/"config/experiment.json").read_text()))
