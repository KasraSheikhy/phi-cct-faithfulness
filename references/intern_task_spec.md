# One-week research task: what affects phi-CCT faithfulness scores?

**Period:** 11-16 November 2026. **Deadline:** 16 November 2026, 23:59 Tehran time.

## Research question

We want to investigate whether properties of an input, intervention, or prompt affect a correlation-based measurement of LLM explanation faithfulness. Implement a reproducible experiment and examine plausible alternative explanations for its results. The goal is to test the hypothesis, not to obtain a predetermined positive result.

Read Siegel et al., *Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations*, particularly the phi-CCT definition, intervention procedure, and limitations:

- Assigned version: https://arxiv.org/abs/2503.13445
- OpenReview: https://openreview.net/forum?id=nkJIMejfaN
- Official code for consultation and citation: https://github.com/google-deepmind/corr_faith

The paper already discusses sensitivity to verbosity and intervention prevalence. Reproducing that property is a useful sanity check; it is not itself a new finding. Your contribution in this task is a careful implementation and analysis of the specified variables.

## Required and optional scope

| Component | Required | Optional, after completing the required experiment |
|---|---|---|
| Dataset | ENSLI/e-SNLI | ECQA |
| Model | One modern Qwen checkpoint: `Qwen/Qwen3-8B` | `google/gemma-4-12B-it` or `Qwen/Qwen3.5-4B` |
| Implementation | Python, PyTorch, and your own experimental pipeline | Additional intervention methods or sensitivity experiments |
| Variables | All five checks described below | Well-motivated additional variables |

Implement the pipeline from scratch; no starter notebook is provided. Standard libraries and cited components are allowed. You may consult the official paper code and must identify anything adapted from it. Use Kaggle, Colab, another GPU provider, or a local machine. (Kaggle is strongly suggested) No fine-tuning is required.

Evaluate on **1,000 distinct sampled examples from each dataset used**: 1,000 e-SNLI examples are required; if you attempt optional ECQA, use a separate 1,000-example ECQA sample. Use a fixed seed, preserve official evaluation splits, record source IDs and label proportions, and freeze the sample before model outputs are inspected. Use the same frozen examples and edits for model/layout comparisons. First verify execution on small smoke examples; these smaller debugging runs do not satisfy the final evaluation requirement. Create at least two valid counterfactual edits per example where feasible (e.g. insertion and deletion); report examples with insufficient valid edits instead of silently replacing them after seeing outcomes. Report all 1,000 original examples in coverage, including invalid proposals and failed or blocked generations. A run that does not complete is an incomplete submission and must be labeled accordingly. Do not replace the main model with an older tiny checkpoint without documenting the limitation. Completing the required experiment takes priority over adding models or datasets.

Plan the compute budget early. Estimate runtime from the smoke run and save resumable checkpoints. Multiple interventions do not increase the number of independent source examples beyond 1,000, and a 1,000-example sample does not guarantee adequate power for every subgroup; report uncertainty and sparse strata.

Record exact model revisions, dataset sources/splits/IDs, package versions, hardware, and numerical precision. Use the model's chat template and explicitly document its thinking setting. For the primary controlled experiment use greedy decoding (`do_sample=False`) with thinking disabled; keep this fixed across original and counterfactual inputs. Do not silently truncate inputs or explanations or interpret a malformed response as a prediction.

## Metric and counterfactuals

Generate an explanation and predicted label for each original input (explanation-then-predict not predict-then-explanation). Edit the input and run it again in a fresh context with the same settings:

- `I_D = 1` if the predicted semantic label changes; otherwise `0`. This measures model sensitivity, not a change in correctness.
- For the paper's insertion setup, `E_D = 1` if the post-intervention explanation mentions the inserted target; otherwise `0`. Document the lexical matching rule and inspect paraphrase/false-match errors.
- `phi-CCT = corr(E_D, I_D)`. If either variable is constant, the correlation is undefined; report `NaN` and the counts, not zero.

You may use deletion, insertion, or another justified counterfactual construction method, including LLM-generated proposals, masked language models, controlled replacements, or methods found in the literature. A simple baseline is one-word deletion or adjective/adverb insertion. Grammatical minimal edits are more useful than arbitrary text corruption.

**Two intervention family is the minimum requirement e.g. implementing both deletion and insertion.** 

Deletion is an extension of the paper's insertion setup: check the deleted target against the matching original/baseline explanation, state this convention, and report deletion separately from insertion. For replacements, define which explanation and target(s) are checked before running. Do not present an undocumented variant as an exact reproduction of the paper.

Choose edits independently of whether the evaluated Qwen flips or mentions the target. Keep premise/hypothesis roles and label-option identities intact. Save original and edited text, exact edit spans, target, method, features, and rejection reasons. Inspect at least 20 edits per dataset manually before the main run, without seeing model outcomes, covering each intervention family you use. Record grammaticality, coherence, and whether task semantics or the correct label may change. Legitimate semantic changes are not automatically invalid, but must be distinguished from malformed or ambiguous inputs and discussed in interpretation.

## Five mandatory variable checks

Implement and report all five. These are candidate confounders or moderators to investigate, not established causal confounders. Pre-specify one primary hypothesis and label the other planned tests as secondary. A lack of variation or insufficient sample size is a valid documented outcome; omitting a check is not.

1. **Named entity:** annotate the edited span's named-entity status and, where possible, entity type using an NER method. Check some annotations manually. Capitalization alone is not a reliable NER method. Report entity/non-entity coverage and any sparse groups.
2. **Polarity:** define and implement positive/negative/neutral polarity for the edited target or context. State the lexicon/classifier and its scope. Distinguish sentiment polarity from grammatical negation, such as `not`; they are not interchangeable.
3. **Prompt position:** record the edited target's normalized position in the input and use explicit early/middle/late bins or another justified design. Define the reference span and report premise versus hypothesis separately where relevant. If using a cue, distinguish cue placement from target location. Fixed instructions should not be edited.
4. **Lexical rarity/IDF:** compute target rarity with a documented tokenization rule and training-only reference corpus. State the IDF formula, corpus size, and treatment of unseen words. Freeze bins without selecting thresholds from favorable outcomes.
5. **Answer-option position:** e-SNLI has three semantic labels. Display them in controlled permutations with letter codes, keeping the underlying task fixed, and map predictions back to `entailment`, `neutral`, or `contradiction` before computing `I_D`. Use matched original/edited option orders. A balanced three-layout cyclic design is sufficient; six permutations are a stronger extension. Keep label position separate from answer/explanation response order. For optional ECQA, permute the five choices and preserve their original identities when comparing predictions.

Think beyond these five: propose an additional variable and explain its plausible mechanism. Implementing a well-designed extra analysis, especially a convincing and checked result, earns bonus credit.

## Required analysis

The main goal is to evaluate the hypothesis that each of the five specified variables affects the interpretation of phi-CCT as a faithfulness measure. Analyze each dataset, model and intervention family separately. The required experiment uses **Qwen/Qwen3-8B without quantization**, greedy decoding, thinking disabled, and **explanation-then-predict** for both original and counterfactual inputs.

Report the following:

1. **Coverage and validity:** account for all 1,000 sampled source examples per dataset, including rejected edits, failed or blocked generations, and valid original/counterfactual pairs. Report expected and completed jobs and the actual number of valid edits per example. Identify rows by source example, intervention, model and option layout. Label unfinished runs as **incomplete**; do not present only successful rows as full coverage.
2. **Model and metric summary:** report baseline classification accuracy, overall phi-CCT for each intervention family with a bootstrap 95% confidence interval, impact rate `P(I_D=1)`, mention rate `P(E_D=1)`, and 2*2 table related to `E_D` & `I_D`. Accuracy must use canonical semantic labels, not displayed option letters. If comparing counterfactual predictions with the original gold label, state that this is not counterfactual accuracy unless the edited example's correct label has been checked.
3. **Statistical tests for all five variables:** state a null hypothesis and a justified test for each variable, and report its p-value, effect size, uncertainty and sample/group counts. Explain what the test measures. An interaction model such as `I_D ~ E_D * Z`, an appropriately clustered comparison of subgroup phi values, or another justified method may be used. For categorical variables with several levels, distinguish an overall test from individual level comparisons. Account for dependence among edits and layouts from the same source example. Report non-estimable tests explicitly, with a reason, rather than inventing p-values or omitting a variable. Report all planned tests; include a multiple-testing correction such as Benjamini-Hochberg alongside the raw p-values.
4. **Plots for all five variables:** plot phi-CCT against the variable's categories or pre-specified values/bins, with **bootstrap 95% confidence intervals**. For continuous variables such as IDF or normalized position, document the binning or estimation method. Include group counts and label sparse or undefined estimates. Keep intervention families separate, using separate panels or clearly distinguished series where helpful. Resample complete source examples so their repeated edits and option layouts remain together, and state the number of bootstrap draws and how undefined draws were handled.
5. **Qualitative examples:** provide at least five traceable examples explaining important results, including both discordance directions (`I_D=1,E_D=0` and `I_D=0,E_D=1`) when available, and a counterexample or failure. For statistically supported findings, include examples illustrating the relevant variable and discuss plausible alternative explanations. If results are inconclusive, explain that uncertainty. Include original and counterfactual inputs, target/edit, baseline and new explanations, semantic predictions, `E_D`, `I_D`, and a concise interpretation. Link each example to its raw CSV record. Report unavailable case types honestly rather than inventing or duplicating examples.

Additional metrics or analyses are welcome if they help answer the question. Examples include the four `(E_D, I_D)` contingency counts, TPR/FPR, prevalence-standardized phi-CCT, or robustness to alternative mention rules. These are supplementary; prioritize the required five-variable tests and plots. Explain what each added analysis contributes.

Interpret p-values and plots together with effect sizes, uncertainty, intervention quality and coverage. A significant association or interaction alone does not prove that a variable is a causal confounder or that phi-CCT is universally invalid. State whether the evidence supports the proposed mechanism, weakens it, or remains inconclusive. Never count parsing/generation failures as valid negatives or equate a label flip with unfaithfulness.

## Deliverables

Submit one archive or repository containing:

- A self-contained **Jupyter notebook implemented from scratch in Python/PyTorch** that runs with **Restart & Run All**, with configuration at the top, a small smoke test, the full evaluation path, readable functions and recorded outputs. Include the required tests and plots for all five variables. It should not depend on hidden local files or hardcoded secrets. Estimate runtime from the smoke test and use resumable checkpoints if needed.
- A concise **1-2 page report in PDF format** containing the hypothesis, experimental design, model/data settings, intervention audit summary, p-values and main plots/results for all five variables, limitations and conclusion. Include compact qualitative examples; full transcripts can be linked by IDs to the accompanying CSVs. Put additional plots or details in the notebook when needed to keep the PDF concise.
- **CSV files** containing the frozen sampled data and counterfactual proposals, raw outputs and parsing/generation statuses, intervention audit records, coverage for all 1,000 source examples, overall accuracy/phi-CCT/impact/mention results, all five variables' statistical tests and plotted estimates with confidence intervals, and qualitative examples. Preserve source IDs and matching keys so results can be traced back to the original outputs.
- A **short README** with reproduction steps, runtime and resource use, exact model revisions, numerical precision, package versions, and any limitations or incomplete coverage. Document that the required Qwen3-8B run uses no quantization and follows explanation-then-predict with thinking disabled.
- An account of **AI assistance in approximately two paragraphs**, including what tools were used, which parts they helped with, how their output was checked, and any reused components and sources. This can be included in the README so it does not crowd the PDF report. If no AI assistance was used, state that briefly.

## AI use and evaluation

Using AI tools for coding, debugging, literature exploration, and writing is welcome. You must understand the code, concepts, research intent and reasons behind your design choices, and verify the outputs rather than simply passing them on. You are responsible for every submitted component, including generated claims, references and statistical interpretations. Document AI use as described above; do not submit unexamined AI output or act only as a relay between an AI tool and the evaluator.

During review you may be asked to trace an example from raw output to metric, explain a function and hypothesis, diagnose a deliberately introduced bug, or make a small live modification. You should be able to do this in your own words. AI use itself is not penalized; unsupported claims or lack of ownership are.

Correctness, understanding, reproducibility, thoughtful auditing and clear interpretation matter more than a superficially positive result. The 1,000-example evaluation remains required; smaller debugging runs do not replace it. A null result with a sound analysis is acceptable.

Only after completing e-SNLI with Qwen3-8B and all five mandatory checks, use any remaining time for **ECQA with its own 1,000 sampled examples**, **google/gemma-4-12B-it or Qwen/Qwen3.5-4B**, or additional variables/intervention methods. Well-motivated, carefully checked extra findings earn bonus credit.
