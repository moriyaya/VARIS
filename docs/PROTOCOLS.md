# Protocol notes

The JSON files in `configs/` specify the archived experiment settings. Commands are in [Training and evaluation](RUNNING.md).

## Numerical toy

Both triple-well variants have three analytic lower optima. The frozen-scale variant evaluates response selection under the original upper solver; the decision-coupled variant differentiates through the selected generated response and runs projected Adam for 200 steps. The reference minimizer is analytic and is used only for evaluation. Use all six contexts and seven preferences for each of the three seeds. Average queries within a seed, then report the mean and standard deviation across the three seeds. Context-shift cases can also be reported separately using the `split` field.

The inference code retains the original validity radius of 0.18. Returned-response metrics and projected-reference regret are separately named in the CSV.

## PDE inverse problems

The solver uses 64 spatial points, 24 time steps, final time 0.20, reaction coefficient 1.5, and four sine/four cosine coefficients. Observations consist of 62 entries. A shared conditional MLP takes the observation, log diffusivity, and branch code. Training uses codes −1 and +1; inference evaluates five codes (−1, −0.5, 0, 0.5, 1) together in one call. The same pool is reused across all four preferences.

The configuration includes a 1e-4 admission tolerance, 1e-2 repair eligibility band, 0.15 trust radius, and at most 50 repair trials. Repaired responses are checked again and compared with the valid incumbent. Unresolved queries invoke the original 500-step, five-start lower-level Adam fallback.

Average task/preference rows within each split and seed, then average seeds. `online_s` measures batched inference on the recorded hardware, including the checks, repair and fallback that actually occurred.

## Robotic configuration optimization

The original frozen IKFlow checkpoint proposes 16 configurations for each shifted target. Admissibility requires ≤1 mm translation error, ≤1 degree orientation error, bounded joints and no self-collision. A query keeps a valid archive throughout upper optimization; the final returned response minimizes its scalarized upper score over that archive.

The formal grid is 32 targets × three preferences × three seeds. Retain the full 128-target generation size even when checking fewer targets. Random streams for initialization and generated candidates are derived from dataset seed, target ID, inference seed and stream name. The outer update uses implicit hypergradients and a preference QP. The IKFlow generator is pretrained and frozen.

## Medical image segmentation

The fixed five-fold split partitions all 48 cases, with one test fold and one validation fold per run. Task02 has six experts except case09 with five; Hungarian matching handles the variable expert count. Training uses nearest-neighbor resizing of masks, bilinear resizing of images, per-image percentile normalization, flips and 90-degree rotations. The six output masks are generated together.

Annotation-coverage Dice averages the best predicted Dice for each expert mask. Prediction-precision Dice reverses this matching direction. Generalized energy distance uses IoU distance. Q-Dice compares the average predicted probability map with mean expert occupancy at the implemented thresholds. Preference Dice averages three independently selected prediction/reference pairs (smallest area, medoid, largest area).

For each seed, pool out-of-fold per-case results over all five test folds so that every case contributes once. Average these per-seed metrics over seeds 0, 1 and 2 and report their standard deviation. The per-fold `result.json` preserves case-level records for this aggregation. This release uses the original validation-coverage checkpoint policy.

## Output metrics

For the numerical toy, `preferred_valid` requires the prescribed validity radius and a preference-optimal nearest exact branch, with ties accepted. `projected_regret` evaluates the nearest exact response; `raw_regret`, where reported, evaluates the returned response. The decision-coupled variant also reports upper-decision error.

For PDE inverse problems, `valid` means observation MSE ≤ 1e-4, `branch_correct` measures preferred-branch agreement, and `joint` requires both. Reference responses are used to compute evaluation metrics.
