# Experiment plan — draft before any model outputs

## Question and interpretation

Do input, intervention, and prompt properties moderate the relation between a target's explanation mention and the intervention's semantic prediction impact? Treat the five requested variables as candidate moderators/confounders. An association does not establish causal confounding or universal invalidity of phi-CCT. A null or non-estimable result is acceptable when reported accurately.

Pre-specify named-entity status as the primary hypothesis; polarity, normalized target location, training-corpus IDF and answer-option position are planned secondary hypotheses. Analyze insertion and deletion separately. Any post-outcome changes must be dated and labeled exploratory.

## Data freeze and intervention audit

1. Resolve and record an immutable original e-SNLI repository revision. Sample 1,000 distinct IDs uniformly without replacement from the official test split with seed 20261115. Validate ID and text uniqueness, save source rows and label proportions, and hash the manifest. Use training data only for IDF; do not tune choices using test labels or model outcomes.
2. Construct at least one valid insertion and one valid deletion per source where feasible. Preserve all proposals and deterministic selection order. Use grammatical modifier insertion/deletion and broader entity-compatible templates if justified by a blinded audit. Do not select an edit because Qwen flips or mentions its target. Preserve premise/hypothesis roles. If safe candidates do not exist, record insufficient coverage without replacing the frozen source.
3. Before model outputs, a human reviews at least 20 proposed edits, including at least 10 per family, for grammaticality, coherence and semantic/gold-label change. Review sparse entity and polarity groups as available. Record reviewer identity, timestamps, exact texts and spans, accept/reject decision, uncertainty, reasons and any template revisions. AI suggestions do not count as completed human review. Freeze proposals, selection rules, annotations and audit records before the main run.
4. Run a small smoke test on fixed source IDs from the frozen sample after the audit. Final sample selection never depends on smoke outcomes. If smoke causes prompt/parser/procedure changes, invalidate those smoke outputs and use the final frozen procedure for every final comparison.

## Inference and compute

- Qwen/Qwen3-8B with exact revision; FP16, no quantization, greedy `do_sample=False`, chat template with `enable_thinking=False`.
- Generate one response containing an explanation first and the final answer letter last. Baseline and edited contexts are fresh and use identical instructions and layouts. The generated explanation is distinct from Qwen's disabled thinking mode.
- Three cyclic option layouts per source and edit. Map A/B/C to canonical labels before accuracy and flip checks. Baseline is reused across insertion/deletion only within the same source and layout.
- Nominal workload: 1000 × 3 × (1 baseline + 1 insertion + 1 deletion) = 9000 responses, 6000 pairs. Invalid proposals and failed outputs reduce valid pairs, not the frozen denominator.
- Measure loading, prompt and generation time, output lengths, peak memory and valid-response rate on smoke. Project remaining time from observed throughput with a stated margin; do not invent runtime estimates. Save per-job checkpoints and resume by source/edit/layout/model/configuration hashes. Prevent mixing settings or duplicate jobs.
- No silent input truncation; reject an overlong prompt explicitly. A response hitting the token limit without termination is a generation failure, even if a label can be extracted. Preserve raw malformed, blocked and failed responses; do not assign E_D or I_D to invalid pairs.
- Local machine: Apple M4 Pro, 24 GB RAM. Start with batch size 1 on MPS. If unsuccessful, use a suitable Colab GPU; at least 24 GB VRAM is preferred. Keep final hardware/backend consistent or disclose and analyze any change.

## Variable definitions to freeze

| Variable | Proposed operationalization | Important check |
|---|---|---|
| Named entity | spaCy NER span overlap; deletion in original, insertion in edited text; retain entity type | Entity coverage may be very low in e-SNLI. Manually check labels; never use capitalization alone or manufacture counts. |
| Polarity | NLTK VADER lexicon score of edited target, positive/negative/neutral by sign; retain separate context score and grammatical negation flag | Sentiment and negation are different. Multiword targets need a documented aggregation rule. |
| Prompt position | Target token midpoint / total tokens in premise + hypothesis; early [0,1/3), middle [1/3,2/3), late [2/3,1]; whitespace tokenization | Deletion position measured on original; insertion on edited text. Retain role and within-role position; report roles separately. This measures target location, not randomized instruction placement. |
| Lexical rarity/IDF | Lowercase alphabetic word tokens; document = training premise + hypothesis; IDF = log((N+1)/(df+1))+1; unseen df=0; target token mean for phrases | Learn tertile boundaries from training vocabulary IDFs before outputs; preserve ties and empty bins, and report unseen rate. No test-output-driven thresholds. |
| Answer-option position | Position (1/2/3) of original gold semantic label in each cyclic layout, a fixed anchor available before generation | This is not the model-selected option position. Gold anchor balances positions within each source; provide layout IDs and label mappings. |

An optional exploratory variable is target lexical overlap between premise and hypothesis. It may influence both explanation copying and inference sensitivity. Defer implementation until required work is complete. Verbosity is a known paper sensitivity; recording it is a sanity check, not a novel finding.

## Metric and mention matching

I_D is a change of canonical semantic prediction, not a correctness change. Insertion E_D uses the edited explanation and inserted target. Deletion E_D uses the matched baseline explanation and deleted target, explicitly an extension rather than an exact paper replication.

Proposed primary mention rule: case-insensitive whole-word match or English Snowball stem match, with contiguous token matching for multiword targets. Save exact versus stem flags separately. The paper uses substring OR stem matching (section 4.2); replacing substring with word boundaries is a documented modification. Also compute the paper rule as sensitivity analysis and manually inspect paraphrases/false matches. Empty/ambiguous targets are invalid; missing explanations are not non-mentions.

Compute phi from the binary 2×2 counts; constant E_D or I_D yields NaN with reason and counts. Report baseline semantic accuracy on valid predictions with its denominator plus full coverage and invalid rate. Also provide correct / 3000 scheduled baseline jobs as a full-coverage rate. Never call accuracy against original labels on edited text counterfactual accuracy without human checked new gold labels.

## Statistical analysis

- For every family: overall phi, 2×2 counts, mention and impact rates, subgroup counts, and all five variable plots with 95% cluster-bootstrap intervals (2,000 draws, fixed seed). Resample full source clusters, retaining repeated layouts and edits; count undefined draws. Plot undefined and sparse strata explicitly. Report number of unique sources as well as pair rows.
- Planned tests (finalized before study outputs): a centered source-cluster bootstrap omnibus test of equality of subgroup phi values, one per variable and family. Form differences against the first observed level, estimate their joint covariance from the same cluster-bootstrap draws, and compute an omnibus quadratic statistic. Center bootstrap differences at their observed contrasts to approximate the null distribution, then use `(1 + number of null statistics >= observed)/(1 + valid draws)` as the p-value. This is an approximate bootstrap test under the regularity of nondegenerate binary margins, not an exact randomization test or proof of causal confounding. Rank-deficient covariance, undefined point phi, fewer than 20 sources in a group, or more than 5% jointly undefined draws makes the test non-estimable.
- Report phi contrasts, percentile confidence intervals, raw p-values, BH adjusted p-values across the 10 planned tests (five variables × two families), counts and failure reasons. Keep non-estimable tests as rows with NaN p-values; conservatively use planned family size 10 in BH. A risk-difference interaction implementation is retained for synthetic validation/possible supplementary work but is not the primary reported test.
- Groups with fewer than 20 unique sources are labeled sparse; report descriptive estimates but no subgroup inference. This is a pre-specified practical reporting threshold, not a power guarantee. Collinearity, constant outcomes or too few informative clusters also prevent inference.
- Obtain at least five traceable qualitative records, covering both discordance directions if they occur plus a failure/counterexample. Include source/edit/layout/job keys, raw responses, semantic predictions, E_D/I_D and alternative explanations. State unavailable case types honestly.

## Completion checklist

- [x] Read supplied assignment and identify required scope.
- [x] Inspect local hardware and choose local-first, Colab fallback.
- [x] Install isolated dependencies, check dependency consistency, verify MPS availability and record installed versions (model compatibility still pending).
- [x] Freeze 1000 evaluation IDs, checksum and class counts; create initial all-source coverage ledger.
- [ ] Freeze training-only feature rules and intervention/analysis protocol.
- [ ] Produce proposals and complete blinded human audit of at least 20 edits.
- [ ] Complete full notebook implementation and correctness checks.
- [ ] Run smoke with required Qwen and estimate time/memory.
- [ ] Freeze final protocol and execute all scheduled jobs with resume support.
- [ ] Reconcile all 1000 sources, failures and rejected edits.
- [ ] Produce ten planned test rows and five plots with both family panels.
- [ ] Inspect mention errors and export at least five qualitative examples.
- [ ] Create verified 1–2 page report PDF and complete AI-assistance account.
- [ ] Verify Restart & Run All and package notebook, README and real CSV outputs.

Source references and fixed paper version are in the README. No empirical results exist at setup.

## Amendments before study model outputs (16 November 2026, Tehran)

- IDF values use all 549,367 training rows. Bins use the tertiles of these training-derived IDF values across frozen selected targets, rather than vocabulary-type IDF tertiles. This is outcome-blind; vocabulary-type tertiles would put most ordinary intervention targets into a single group. The two cut points are frozen in the feature manifest and remain unchanged when adding fallback edits.
- Named-entity status includes conventional named entity classes such as PERSON, GPE, ORG, NORP and EVENT. Numeric/date recognition alone is not a named entity; raw NER types, including CARDINAL, are saved separately.
- Eighty sources without a screened deletion receive a second screened insertion. This maintains two active proposals per source; insertion and deletion estimates still use separate family-specific coverage and source-cluster uncertainty.
- User reviewed the 24 displayed proposals globally with the response "they are all ok in my opinion". Record global grammaticality/coherence acceptance and explicit uncertainty for meaning/gold-label changes; do not imply human relabeling or detailed judgments not supplied. A clear malformed deletion in review item 20 is excluded by the analyst, preserved as a failure, and replaced by the next screened deletion before study inference. The audit is a limited quality check, not a guarantee all 2,000 proposals are natural.
- Primary p-values directly test subgroup phi equality using a centered source-cluster bootstrap. This choice was made before any study outputs, to align the test with the stated correlation-based question.
- A synthetic end-to-end check caught seven digit-only deletion targets with no tokens under the pre-specified alphabetic lexical rule. Before any study inference, these proposals were quality-excluded and replaced by the next screened alphabetic candidates. All failed proposals are retained. This changes neither the 1,000 sampled IDs nor the IDF boundaries. The earlier protocol manifest is preserved as v1; the corrected final manifest is the one validated before inference.

## Runtime pilot adjustment before the full evaluation

The first frozen five-source smoke completed 45/45 valid responses, averaging 8.529 seconds and 60.84 generated tokens per response. Its 9,000-response projection was 21.32 hours, or 26.65 hours with the planned 25% margin, exceeding the available budget. Only format and runtime were inspected; no subgroup p-values were computed and no examples/edits/features were changed.

Before any full-run output, the common instruction was changed from one or two explanatory sentences to one concise sentence of at most 25 words. Repeat the smoke with this final prompt, discard the old pilot from final evaluation counts, and run all 1,000 sources/layouts/edits under the final same setting. The word cap is a prompt instruction, not output truncation or a filter: preserve and report noncompliance. Verbosity is a known paper sensitivity, so results describe this concise setting and do not establish a new verbosity finding. All hypothesis tests, bins, sample IDs and intervention selections remain as specified before pilot outcomes.
