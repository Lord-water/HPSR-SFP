# Reproduction scope and version relationship

## Current method versus historical baselines

| Current HPSR-SFP | Earlier Pragio implementation |
| :--- | :--- |
| Twelve candidate substitutions, one candidate-token representation per ending | One masked-context representation and twelve output logits |
| Natural-choice LoRA adaptation, then frozen features | Joint usage/judgment training of shared encoder and heads |
| Stabilized shared vector + candidate residuals + regularized biases | Independent or coupled linear output heads |
| Train-only moments/weights; development-fold calibration/selection | Historical checkpoint/calibrator selection protocol |
| Fixed PoE and Bayes-Gate selection | Earlier acceptability/ranking interfaces |

Do not use the earlier `scripts/train.py --run original_independent_s42` command as the current method. It remains a baseline. Historical configuration names are preserved so existing documentation and launcher IDs remain valid.

## What is included

- Current encoder, exact scientific readout/calibration functions, explicit stage launchers and recorded development folds.
- Same-grid raw/centered/scalar controls and paired statistical routines.
- 498 text-free final annotation mappings, fixed LCCC indices, source reconstruction and historical baseline code/configurations.
- Synthetic engineering checks; their success is not an experimental result.

The mathematical class/function bodies listed in [code provenance](../protocols/code_provenance.json) are extracted from the existing experimental source without mathematical edits. New wrappers provide portable JSON/NPZ artifacts, explicit IDs and encoder checks; they are packaging code, not the original queue launcher. Dependencies on the retired private scheduler are intentionally absent.

## What remains outside the public release

Raw source dialogue; NaturalConv-derived X100/Y100 record-level material; pretrained model weights; learned adapters, heads and calibration parameters; cached hidden states; paper files; experimental result tables and predictions. Model provider names do not certify the exact bytes of the original local pretrained snapshot. Backend, dtype and package differences may affect reproduction.

The reference numerical implementation uses float64 readout fitting, candidate-balanced masked BCE, 400-step L-BFGS-B for Stabilized HPSR, the fixed bias penalty .005 and lambda-scaled usage penalties. The unconstrained drift variant is discussed in the paper, but is not the default public training objective. This release does not claim to expose every exploratory configuration or to be a self-contained full-experiment archive.

## Dataset names

`first_test.json` → Test-Initial (200); `confirmation.json` → Test-Refined (200). The training/dev subsets contain 48/50 contexts. Their existing source IDs and annotation provenance are preserved. The development calibration fold mapping is the one used by the current frozen-feature experiments, provided separately from the historical baseline folds.
