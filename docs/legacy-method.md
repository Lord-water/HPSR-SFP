# Earlier Pragio implementation

This page describes the historical two-head baseline, not the current HPSR-SFP pipeline.

The earlier model encodes dialogue history and a masked target once with a BERT-style encoder. Separate usage and acceptability heads produce a normalized choice distribution and candidate-wise sigmoid scores. Observed choices supervise NLL and resolved human judgments supervise masked BCE. Their weighted sum updates the shared encoder and heads jointly. Calibration is selected using development judgments.

![Earlier shared-encoder, two-head architecture](../assets/method.png)

The dialogue is illustrative; this diagram does not show experimental predictions. The earlier [banner](../assets/pragio-banner.svg) remains a historical asset.

The implementation remains in [method_core.py](../src/method_core.py) and [training_interface.py](../src/training_interface.py). Independent, coupled, AN-LS, EPR and CRISP comparisons retain their original configurations. Follow [legacy commands](legacy-usage.md) for these baselines and [current usage](usage.md) for counterfactual adaptation and Stabilized HPSR.

The transition changes the representation and training pipeline. Existing annotation IDs and labels remain reusable; learned weights from the earlier model are not interchangeable with the new candidate-conditioned encoder.
