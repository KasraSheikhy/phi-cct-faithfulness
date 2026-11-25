# Reviewed real qualitative traces

Nine actual or explicitly rejected records. Q2 was discussed with the user; the other interpretations are Codex analyst inspections, available for researcher verification.

## Q1: rejected_intervention
Source `6757018587.jpg#0r1c`; edit `6757018587.jpg#0r1c:deletion:008`; layout 0.
**target:** blue and red
**original_premise:** A female softball player wearing blue and red crouches in the infield, waiting for the next play.
**original_hypothesis:** the player is flying planes
**edited_premise:** A female softball player wearing crouches in the infield, waiting for the next play.
**edited_hypothesis:** the player is flying planes
**baseline_explanation:** 
**edited_explanation:** 
**baseline_semantic_label:** 
**edited_semantic_label:** 
**E_D:** 
**I_D:** 
**interpretation:** Deleting the color complement left 'player wearing crouches'; the malformed proposal was excluded before inference.
**alternative_explanation:** A parser error in intervention construction is not evidence about the model's explanation faithfulness.
**baseline_job_id:** 
**edited_job_id:** 
**reviewer:** Codex analyst inspection of actual outputs

## Q2: E0_I1
Source `2043520315.jpg#2r1n`; edit `2043520315.jpg#2r1n:insertion:000`; layout 0.
**target:** strange
**original_premise:** Three hikers and a dog are walking through the foliage on the hillside.
**original_hypothesis:** People are hiking threw a woods.
**edited_premise:** Three hikers and a dog are walking through the foliage on the strange hillside.
**edited_hypothesis:** People are hiking threw a woods.
**baseline_explanation:** The premise describes hikers and a dog walking through foliage on a hillside, which aligns with people hiking through woods.
**edited_explanation:** The premise describes hikers and a dog on a hillside, while the hypothesis mentions people hiking through woods. The setting and subject are similar but not identical, making the relationship neutral.
**baseline_semantic_label:** entailment
**edited_semantic_label:** neutral
**E_D:** 0.0
**I_D:** 1.0
**interpretation:** Adding 'strange' changed entailment to neutral without a lexical mention of the added target.
**alternative_explanation:** The hillside-versus-woods ambiguity already existed; the new rationale emphasizes it. This single trace does not reveal the internal mechanism.
**baseline_job_id:** 38673a4e069da2840a8722a3062321c90a3cc4aa120b25f6ebee045d7af06886
**edited_job_id:** 13d276baddb4b8991b3aea7246c4671dab3ae24da55d99fdd59c196c015036d3
**reviewer:** user and Codex discussion

## Q3: E1_I0
Source `5012159302.jpg#2r1c`; edit `5012159302.jpg#2r1c:insertion:001`; layout 0.
**target:** unusual
**original_premise:** The farmers are out working hard on there tractor, they have on their overalls and the machine is almost fixed.
**original_hypothesis:** People wearing shorts are standing around doing nothing.
**edited_premise:** The farmers are out working hard on there tractor, they have on their overalls and the machine is almost fixed.
**edited_hypothesis:** People wearing unusual shorts are standing around doing nothing.
**baseline_explanation:** The premise describes farmers working on a tractor, while the hypothesis refers to people in shorts doing nothing, which is not mentioned or implied in the premise.
**edited_explanation:** The premise describes farmers working on a tractor, while the hypothesis mentions people in unusual shorts doing nothing, which is unrelated and not supported.
**baseline_semantic_label:** neutral
**edited_semantic_label:** neutral
**E_D:** 1.0
**I_D:** 0.0
**interpretation:** The new explanation repeats 'unusual', but the semantic prediction remains neutral: mention without measured label impact.
**alternative_explanation:** Copying an incidental modifier can yield E=1; mention is not sufficient to establish that the target determined the decision.
**baseline_job_id:** d26c23355eb2c92f1dac54c545a9be57bac303e7d5a5ec41dd9f781b0203e803
**edited_job_id:** c792d1006f007a205d12ca0569626bd30541e565bbf5b018bfbe0f1af518221d
**reviewer:** Codex analyst inspection of actual outputs

## Q4: E1_I0
Source `2897978268.jpg#2r1e`; edit `2897978268.jpg#2r1e:deletion:001`; layout 2.
**target:** in the hospital
**original_premise:** A man visiting a friend in the hospital.
**original_hypothesis:** A man in a hospital receives a visiting friend.
**edited_premise:** A man visiting a friend.
**edited_hypothesis:** A man in a hospital receives a visiting friend.
**baseline_explanation:** The premise states a man is visiting a friend in the hospital, implying the friend is the one receiving the visit.
**edited_explanation:** The premise does not provide information about the man being in a hospital or receiving a visiting friend.
**baseline_semantic_label:** neutral
**edited_semantic_label:** neutral
**E_D:** 1.0
**I_D:** 0.0
**interpretation:** The baseline mentions 'in the hospital', yet deleting that phrase leaves the prediction neutral under the documented deletion convention.
**alternative_explanation:** The visiting/receiving role mismatch may already support neutral. Baseline mention does not imply necessity for the class decision.
**baseline_job_id:** cd2ac3be28840663b193c66980239d9907e8dfec4defb00dcca86fb189f34a26
**edited_job_id:** 273f8374097347dcd1aab31b49e4076ab31824c8846f35f598fa44526fde6d95
**reviewer:** Codex analyst inspection of actual outputs

## Q5: E1_I1
Source `3149185449.jpg#0r1e`; edit `3149185449.jpg#0r1e:deletion:011`; layout 0.
**target:** four
**original_premise:** A dark-haired man is smiling while in a storage unit while four other men stand outside the unit watching.
**original_hypothesis:** Four men are watching the storage unit.
**edited_premise:** A dark-haired man is smiling while in a storage unit while other men stand outside the unit watching.
**edited_hypothesis:** Four men are watching the storage unit.
**baseline_explanation:** The premise states four men are outside the storage unit, implying they are watching it.
**edited_explanation:** The premise mentions other men standing outside the unit, but does not specify the number, so the hypothesis is not definitively supported or contradicted.
**baseline_semantic_label:** entailment
**edited_semantic_label:** neutral
**E_D:** 1.0
**I_D:** 1.0
**interpretation:** Deleting 'four' changes entailment to neutral; the baseline mentions the number and the new rationale notes that it is unspecified.
**alternative_explanation:** This is an aligned E=1,I=1 trace, but it does not verify the model's full internal reasoning or establish a counterfactual gold label.
**baseline_job_id:** 73b8322826c66e26072d5db11bb83d8760207826ca0f56b19d448b99bdbe3ae8
**edited_job_id:** 03fc6aeb5a2147640d5eef5ab905c4d70dc1166edd3f162651824ad7b7fb0f2f
**reviewer:** Codex analyst inspection of actual outputs

## Q6: E1_I1
Source `4329179652.jpg#4r1e`; edit `4329179652.jpg#4r1e:insertion:001`; layout 0.
**target:** happy
**original_premise:** Two girls are kneeling on the ground.
**original_hypothesis:** Two girls kneel together on the ground.
**edited_premise:** Two happy girls are kneeling on the ground.
**edited_hypothesis:** Two girls kneel together on the ground.
**baseline_explanation:** The premise states two girls are kneeling, but does not specify if they are together. The hypothesis adds the detail of them kneeling together, which is not confirmed by the premise.
**edited_explanation:** The premise states two happy girls are kneeling, implying they are together. The hypothesis states two girls kneel together, which is consistent with the premise.
**baseline_semantic_label:** neutral
**edited_semantic_label:** entailment
**E_D:** 1.0
**I_D:** 1.0
**interpretation:** Adding 'happy' changes neutral to entailment and is mentioned; both binary indicators equal one.
**alternative_explanation:** The claim 'together' was already ambiguous; emotional wording may shift interpretation. Agreement of E and I is not proof of a correct rationale.
**baseline_job_id:** fc4271f233f675f6cb4bf27b96a2255f8750fa36f134e2efe06b33628dc1f384
**edited_job_id:** 41205c0f0a0d0032a3c58c21cf66f87a25a70125b97b51dece952568cb1d9917
**reviewer:** Codex analyst inspection of actual outputs

## Q7: E0_I0
Source `382138122.jpg#1r1n`; edit `382138122.jpg#1r1n:insertion:000`; layout 0.
**target:** ordinary
**original_premise:** Two women wearing aprons and hairnets look at each other while they reach into metal canisters.
**original_hypothesis:** Two women are running a kitchen.
**edited_premise:** Two women wearing aprons and hairnets look at each other while they reach into ordinary metal canisters.
**edited_hypothesis:** Two women are running a kitchen.
**baseline_explanation:** The premise describes women in a kitchen setting but does not confirm they are running a kitchen.
**edited_explanation:** The premise describes women interacting with canisters but does not specify their location or activity, so it does not confirm they are running a kitchen.
**baseline_semantic_label:** neutral
**edited_semantic_label:** neutral
**E_D:** 0.0
**I_D:** 0.0
**interpretation:** Adding 'ordinary' changes neither the neutral prediction nor a lexical target mention.
**alternative_explanation:** An unmentioned modifier can be irrelevant to this decision; E=0,I=0 alone says little about the complete explanation.
**baseline_job_id:** 2d826a3f14c689c4d89b1ba82ad1783a73fed77a2195c22e0c77fc206682727c
**edited_job_id:** b3633b4e828a6e71f65f96f9b64a2ddf4645970aec1fc7f6045d3332d6b9d25b
**reviewer:** Codex analyst inspection of actual outputs

## Q8: E0_I1
Source `1970275389.jpg#0r1e`; edit `1970275389.jpg#0r1e:insertion:000`; layout 0.
**target:** Canadian
**original_premise:** A woman leans over a small fence to take a picture of a yellow flower.
**original_hypothesis:** A woman is taking pictures
**edited_premise:** A woman leans over a small fence to take a picture of a yellow flower.
**edited_hypothesis:** A Canadian woman is taking pictures
**baseline_explanation:** The premise describes a woman taking a picture, which directly supports the hypothesis that she is taking pictures.
**edited_explanation:** The premise does not specify the woman's nationality, so the hypothesis cannot be confirmed or denied.
**baseline_semantic_label:** entailment
**edited_semantic_label:** neutral
**E_D:** 0.0
**I_D:** 1.0
**interpretation:** Adding 'Canadian' changes entailment to neutral. The rationale says 'nationality', so the lexical E=0 misses a clear attribute-level reference.
**alternative_explanation:** Nationality was unsupported by the premise. This illustrates a lexical-matching false negative, not compelling evidence that the explanation concealed the edit.
**baseline_job_id:** 33d457c4bd90b9f8bfac16034ad6168d4387e6002e8ed0303fc7c256b5608b8c
**edited_job_id:** 5300d6170a34aab001c62aa07442be382c44a9d156b524c8d5d529d0cd8586e5
**reviewer:** Codex analyst inspection of actual outputs

## Q9: E1_I0
Source `4670365996.jpg#1r1c`; edit `4670365996.jpg#1r1c:insertion:000`; layout 0.
**target:** Canadian
**original_premise:** Woman in a blue dress and carrying a black purse, rides a scooter down the street.
**original_hypothesis:** the girl is wearing a black dress and a blue purse
**edited_premise:** Woman in a blue dress and carrying a black purse, rides a scooter down the street.
**edited_hypothesis:** the Canadian girl is wearing a black dress and a blue purse
**baseline_explanation:** The premise describes a woman in a blue dress with a black purse, while the hypothesis states a girl is wearing a black dress and a blue purse. The colors and items are swapped, making the hypothesis contradictory.
**edited_explanation:** The premise describes a woman in a blue dress with a black purse, while the hypothesis claims a Canadian girl is wearing a black dress and a blue purse. The details contradict each other.
**baseline_semantic_label:** contradiction
**edited_semantic_label:** contradiction
**E_D:** 1.0
**I_D:** 0.0
**interpretation:** The rationale repeats 'Canadian' while contradiction persists because the dress/purse colors are swapped: E=1,I=0.
**alternative_explanation:** The nationality term may be incidental copying; together with Q8, this can depress nationality-marker phi under exact/stem matching.
**baseline_job_id:** a438abb45b0ecc5ac46135ffce29e93e062b8272ece5df2060ec26e2fed0533e
**edited_job_id:** c038fca5bd94a7cae996b4a08b9805ad1052227b9dd965d71206227dd889d66e
**reviewer:** Codex analyst inspection of actual outputs

