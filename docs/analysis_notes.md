# Analysis interpretation

## Planned analysis

The question is whether five candidate moderators affect the operationalized phi-CCT score, not whether all five must be significant. The named-entity insertion contrast is the only primary family/variable comparison with BH q<0.05. Raw insertion position p=0.0275 and IDF p=0.0105 do not survive the ten-test correction (q=0.0916 and 0.0525). All planned rows remain visible; deletion polarity is explicitly non-estimable because its negative subgroup contains only 13 source clusters. These outcomes do not establish equality or causal invariance.

For each categorical variable, contrasts are formed against the first observed level. With observed contrast vector d and cluster-bootstrap contrast covariance S, the statistic is d' S^-1 d. Bootstrap contrasts are centered at d to approximate a null distribution of that quadratic statistic; p=(1+exceedances)/(1+joint-valid draws). This is an approximate bootstrap test under nondegenerate binary margins, not an exact permutation test. It uses the same resampled complete-source weights for every subgroup. Constant/undefined phi, too few sources, rank-deficient covariance, or more than 5% jointly undefined draws prevents a reported p-value. Group phi intervals use valid percentile draws and report the number omitted.

## Supported conditional finding and mechanism

Insertion named-entity phi is -0.158 [ -0.261, -0.035 ] versus 0.265 [ 0.200, 0.328 ] for non-entity spans. The difference (non-entity minus entity) is 0.424 [0.288, 0.546], p=0.0005, BH q=0.005. This compares an intervention-weighted distribution; multiple edits increase a source's contribution, not its independent cluster count. Entity spans here are exclusively NORP adjectives French/Canadian, not a diverse test of person or organization names. Groups also differ in lexical content, target rarity, role and template; these are not randomized matched entity/non-entity minimal pairs.

The primary lexical rule misses a clear alternative phrasing: the model can explain that nationality is unspecified without repeating Canadian or French (Q8). Conversely, it can quote Canadian while a color contradiction already determines the unchanged prediction (Q9). This pattern plausibly depresses E/I correlation without showing that the explanations hide the nationality intervention.

## Post-hoc sensitivity, separate from confirmatory tests

After inspecting Q8, the restricted proxy counts nationality/nationalities in the new explanation for French/Canadian insertions. The frozen interventions, primary E_D and all ten planned p-values remain unchanged. It adds 125 mentions, 60 with I_D=1. A twenty-case output/context inspection supports the intended attribute-level reading, but it is Codex analyst review, not an independent human validation study. The proxy changes entity phi to 0.257 [0.140,0.358]; the group difference becomes 0.009 [-0.111,0.140]. No new confirmatory p-value is claimed for this post-hoc rule.

This sensitivity supports a measurement-operationalization explanation for the primary contrast. It does not establish a general semantic oracle, a causal effect of named-entity status, universal invalidity of phi-CCT, or ground-truth internal faithfulness.

## Qualitative and mention limitations

Nine traceable cases cover both discordance directions, aligned and unchanged cases, and a rejected malformed edit. Q2 includes a human observation that hillside-versus-woods ambiguity already existed; a change in expressed rationale does not identify the internal cause. Deletion checks the baseline explanation and is an extension. Full-phrase matching misses on a hillside versus on the hillside; substring matching can count red in occurred or black in blackboard. The twenty-four mention-review rows retain these errors and uncertainty rather than silently changing the primary measurements.

The pre-run human judgment was global acceptability only. Detailed meaning/gold-label changes were not independently assigned. Additional qualitative interpretations are labeled Codex analyst inspections. All 1,000 sources and all generation/parse outcomes remain in coverage; no post-outcome edit exclusions were made. The fixed concise prompt was chosen solely from runtime before the main evaluation, and word-count noncompliance was retained.
