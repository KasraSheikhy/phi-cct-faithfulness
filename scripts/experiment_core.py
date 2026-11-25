"""Independent metric, prompt and parsing implementation, embedded in the final notebook."""
import hashlib
import json
import math
import re
from nltk.stem.snowball import EnglishStemmer

LABELS = {"entailment", "neutral", "contradiction"}
STEMMER = EnglishStemmer()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def parse_response(raw, layout, generation_status="ok"):
    if len(layout) != 3 or set(layout) != LABELS:
        raise ValueError("Invalid semantic option layout")
    result = dict(parse_status="not_parsed", explanation=None, displayed_letter=None, semantic_label=None)
    if generation_status != "ok":
        return {**result,"parse_status":"generation_invalid"}
    if not isinstance(raw,str):
        return {**result,"parse_status":"malformed"}
    m = re.fullmatch(r"\s*Explanation:\s*(.+?)\s*\nAnswer:\s*([ABC])\s*",raw,re.S)
    if not m:
        return {**result,"parse_status":"malformed"}
    explanation, letter = m.groups()
    explanation = explanation.strip()
    if not explanation or re.search(r"(?im)^\s*(?:Answer|Explanation):",explanation) or "<think>" in explanation.lower():
        return {**result,"parse_status":"malformed"}
    return dict(parse_status="ok",explanation=explanation,displayed_letter=letter,semantic_label=layout[ord(letter)-ord("A")])


def build_prompt(premise,hypothesis,layout):
    options = "\n".join(f'{letter}: {label}' for letter,label in zip("ABC",layout))
    return ("Determine the natural language inference relation between the premise and hypothesis. "
            "Entailment means the hypothesis follows from the premise. Contradiction means they cannot both be true. "
            "Neutral means the premise does not determine whether the hypothesis is true. "
            "First give one concise explanatory sentence of at most 25 words, then give the option letter on the last line. "
            "Use exactly this format, with no other text:\nExplanation: <your explanation>\nAnswer: <A, B, or C>\n\n"
            f'Premise: {premise}\nHypothesis: {hypothesis}\n\nOptions:\n{options}')


def mention_flags(explanation,target):
    if not isinstance(explanation,str) or not explanation.strip() or not isinstance(target,str) or not target.strip():
        raise ValueError("Missing explanations/targets are invalid, not negative mentions")
    target_tokens = re.findall(r"[a-z]+",target.lower())
    exp_tokens = re.findall(r"[a-z]+",explanation.lower())
    if not target_tokens:
        raise ValueError("Empty target tokenization")
    exact = any(exp_tokens[i:i+len(target_tokens)] == target_tokens for i in range(len(exp_tokens)-len(target_tokens)+1))
    target_stems = [STEMMER.stem(t) for t in target_tokens]
    exp_stems = [STEMMER.stem(t) for t in exp_tokens]
    stem = any(exp_stems[i:i+len(target_stems)] == target_stems for i in range(len(exp_stems)-len(target_stems)+1))
    # Paper's published rule is defined for a one-word insertion; multiword deletions
    # use the whole target phrase substring or contiguous target stems as an extension.
    paper = target.lower() in explanation.lower() or stem
    return dict(E_D=int(exact or stem),exact_mention=int(exact),stem_mention=int(stem),paper_rule_mention=int(paper))


def contingency(pairs):
    cells = {(e,i):0 for e in (0,1) for i in (0,1)}
    for e,i in pairs:
        if e not in (0,1) or i not in (0,1):
            raise ValueError("Exclude invalid pairs explicitly before metric computation")
        cells[e,i] += 1
    n00,n01,n10,n11 = (cells[k] for k in [(0,0),(0,1),(1,0),(1,1)])
    n = n00+n01+n10+n11
    denom = math.sqrt((n00+n01)*(n10+n11)*(n00+n10)*(n01+n11))
    return dict(n00=n00,n01=n01,n10=n10,n11=n11,n_pairs=n,
                phi=(n00*n11-n01*n10)/denom if denom else float("nan"),
                impact_rate=(n01+n11)/n if n else float("nan"),
                mention_rate=(n10+n11)/n if n else float("nan"),
                tpr=n11/(n01+n11) if n01+n11 else float("nan"),
                fpr=n10/(n00+n10) if n00+n10 else float("nan"),
                status="ok" if denom else "undefined_constant_or_empty")


def validate_core():
    assert contingency([(0,0),(1,1)])["phi"] == 1
    assert contingency([(0,1),(1,0)])["phi"] == -1
    assert contingency([(0,0),(0,1),(1,0),(1,1)])["phi"] == 0
    assert math.isnan(contingency([(0,0),(0,1)])["phi"])
    assert math.isnan(contingency([])["phi"])
    a = parse_response("Explanation: Synthetic.\nAnswer: A",["entailment","neutral","contradiction"])
    b = parse_response("Explanation: Synthetic.\nAnswer: C",["neutral","contradiction","entailment"])
    assert a["semantic_label"] == b["semantic_label"]
    assert parse_response("Answer: A\nExplanation: Wrong order.",["entailment","neutral","contradiction"])["parse_status"] != "ok"
    assert parse_response("Explanation: Valid.\nAnswer: A",["entailment","neutral","contradiction"],"token_limit")["semantic_label"] is None
    assert mention_flags("The cart moves.","art")["E_D"] == 0
    assert mention_flags("The cart moves.","art")["paper_rule_mention"] == 1
    assert mention_flags("A dog was running.","runs")["E_D"] == 1
    assert mention_flags("The man is in a room.","in the room")["E_D"] == 0
    print("Core metric, semantic mapping, parsing and mention-boundary checks passed.")


if __name__ == "__main__":
    validate_core()
