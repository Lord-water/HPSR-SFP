# HPSR-SFP: stage-by-stage usage

Run from the repository root with Python 3.10 or newer. The `hpsr_sfp` package is the current method; `scripts/train.py` and `src/` remain historical baseline interfaces. No stage is launched automatically on import. Commands below create new private outputs, not published study artifacts.

## 1. Install and reconstruct the source inputs

```bash
python -m pip install -r requirements-hpsr.txt
python scripts/restore_lccc.py --source /path/to/LCCC-large.zip
python scripts/validate_release.py
```

Acquire the source through its original provider. See [data access](data-access.md). The public annotations and indices are unchanged. Do not resample or repartition them. Obtain a local `Qwen/Qwen2.5-1.5B-Instruct` or `hfl/chinese-roberta-wwm-ext` model separately, respecting its terms. Model names are provider identifiers, not redistributed weights. The exact original local weight revision is not certified by this release.

## 2. Adapt on natural choices

```bash
python -m hpsr_sfp adapt --backbone /local/backbone --inputs data/local/train_ready.jsonl --output outputs/hpsr/adapter --device cuda
```

One usage epoch; seed 42; 256 tokens; effective batch 16 (one context with twelve candidates per microbatch); AdamW 1e-4; weight decay .01; rank-8 LoRA with alpha 16 and dropout .05 on attention query/value projections; 5% warmup and the recorded floored cosine schedule. This objective uses only observed choices, not human acceptability labels. GPU execution caps this process at 32% of total device memory and checks free-memory margins. Resource checks cannot reserve memory against other processes. `latest.pt` is an intermediate parameter checkpoint; the public launcher does not yet implement optimizer/RNG-exact resume. Only completed `adapter.pt` is valid for feature extraction.

## 3. Extract frozen features

First select the annotated contexts by ID; this avoids extracting features for the full usage corpus a second time.

```bash
python -m hpsr_sfp select-records --inputs data/local/train_ready.jsonl --labels data/labels/train.json --output data/local/human_train.jsonl
python -m hpsr_sfp select-records --inputs data/local/dev_ready.jsonl --labels data/labels/dev.json --output data/local/human_dev.jsonl
python -m hpsr_sfp features --backbone /local/backbone --adapter outputs/hpsr/adapter/adapter.pt --inputs data/local/human_train.jsonl --output outputs/hpsr/train.npz --device cuda
python -m hpsr_sfp features --backbone /local/backbone --adapter outputs/hpsr/adapter/adapter.pt --inputs data/local/human_dev.jsonl --output outputs/hpsr/dev.npz --device cuda
```

For an existing larger feature cache, `python -m hpsr_sfp subset --features outputs/all.npz --labels data/labels/train.json --output outputs/train.npz` selects the requested IDs without changing their order.

Omitting `--adapter` produces the unadapted Base-Raw control. Each cache stores `h[N,12,D]`, `u[N,12]`, record IDs, explicit candidate order, an encoder identity and input hash. Load only trusted local adapter checkpoints. Extraction also supports `--device cpu`; CPU and CUDA dtypes can produce numerical differences, so use the original CUDA precision for a numerical reproduction.

## 4. Select and calibrate the readout

```bash
python -m hpsr_sfp fit --train-features outputs/hpsr/train.npz --dev-features outputs/hpsr/dev.npz --train-labels data/labels/train.json --dev-labels data/labels/dev.json --folds protocols/hpsr_development_folds.json --method hpsr --output outputs/hpsr/readout.json
```

All feature/usage moments and high-dimensional weights are fit on the 48 training contexts. The fixed 50-context development folds choose the readout grid and calibration family by context-fold OOF Macro-Brier. Calibration is then fit on all development judgments for subsequent application. This is **calibrator cross-fitting**: the high-dimensional readout is not re-trained within those development folds, and the selected OOF score is a selection statistic rather than an unbiased final test estimate.

The recorded HPSR grid is 6 lambda values × 5 residual shrinkage ratios × 2 usage-cue settings, each with temperature and shrunk-candidate calibration. `--method raw`, `centered`, `scalar`, or `spline` provides the same nine-L2/four-calibrator controls. For Base-Raw use the raw control with unadapted features. Search traces, optimizer success flags, gradients and selected parameters are written to the private bundle; an iteration cap is not a convergence guarantee. Tie-breaking follows grid enumeration order. No test labels are accepted by `fit`.

## 5. Predict first; score separately

```bash
python -m hpsr_sfp select-records --inputs data/local/heldout_ready.jsonl --labels data/labels/first_test.json --output data/local/initial.jsonl
python -m hpsr_sfp features --backbone /local/backbone --adapter outputs/hpsr/adapter/adapter.pt --inputs data/local/initial.jsonl --output outputs/hpsr/heldout.npz --device cuda
python -m hpsr_sfp predict --features outputs/hpsr/heldout.npz --readout outputs/hpsr/readout.json --output outputs/hpsr/heldout_predictions.json
python -m hpsr_sfp score --predictions outputs/hpsr/heldout_predictions.json --labels data/labels/first_test.json --output outputs/hpsr/initial_metrics.json
```

Repeat with `confirmation.json` and new output paths for Test-Refined. The subset step reads only annotation IDs; feature extraction and prediction do not inspect judgment values.

Scoring selects annotation IDs from a larger prediction file while requiring complete coverage. It reports Macro-Brier, supported-candidate Macro-AUROC, pooled ten-bin ECE, per-particle values and selected A/N/U counts. U is masked in probability metrics and retained in output counts. Usage-only, PoE and Bayes-Gate are all computed without access to test judgments. PoE maximizes log usage + log calibrated acceptability; Gate restricts to p >= .5, falling back to argmax acceptability if empty. Inventory order breaks ties.

## 6. Paired inference

`hpsr_sfp.statistics.paired(p, baseline, y, group_ids, reps=2000)` returns the paired Brier difference, context-cluster bootstrap CI and two-sided sign-randomization p value. Supply matrices in identical record/candidate order and use the original grouping keys. `holm(comparisons)` adjusts the explicitly supplied family **in place**. For the six Base-Raw attribution comparisons, pass all two-backbone × three-track comparisons in one family. Do not choose a correction family after inspecting significance. These functions are reusable analysis code; no empirical result files or predictions are distributed.
