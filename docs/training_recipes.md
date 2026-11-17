# Training configuration reference

Each JSON configuration binds a checkpoint to its architecture and scientific training parameters. `effective_optimizer` records the optimizer and schedule used by the corresponding training implementation; `effective_loss` records the active objective. The model and output sections define inference behavior. Other training fields retain method-specific source parameters; effective_loss and effective_optimizer define the active objective and optimizer settings. A null source coefficient does not override a numeric effective coefficient. These metadata sections are not direct keyword arguments to the loss functions.

UREP uses a shared ResNet-18/FPN encoder, 64 × 64 non-overlapping patches (K=16), 32-group GroupNorm, reflect padding, and spatial averaging of each raw head before activation. Learned aggregation uses softmax scores; uniform aggregation uses 1/K. The image objective is Smooth L1 on log GSD, beta=0.1. Local NLL uses exp(-s) times squared log residual plus s; A5 uses ordinary patch log-MSE. The A5 coefficient is 0.04768970142662903 from the reported fixed train-sample dose calibration. Other coefficients remain bound to their original configuration.

RT-CNN uses Adam and the configured MultiStepLR schedule; UREP and ResNet-18 use AdamW and cosine annealing. Training lasts at most the configured epoch count; the published weights are the recorded best validation checkpoint, not necessarily the final epoch. Selection uses source-macro MAPE for Mixed8 and image-micro MAPE for the other benchmarks, with the earliest epoch retained for ties.

Inputs use per-channel 2nd–98th percentile stretch. Spatial augmentation applies a reflect-padded shift crop (padding 4), horizontal/vertical flips, and rotations by multiples of 90 degrees. `preprocessing.py` contains the shared per-sample hash-based augmentation recipe. Dataset-specific epoch sampling differs: Mixed8 draws 10,000 per source using permutation cycles, then shuffles; see the dataset metadata and scientific parameters for the exact experiment. Do not infer a full historical random sequence solely from the public seed.

This package provides architecture, loss, preprocessing, inference, and checkpoint-loading code. Historical multiworker training is not guaranteed to be bitwise reproducible. Predictions and image-level analysis arrays belong with the data repository; `reference_prediction_sha256` identifies the original prediction file, while a null download URL means that result artifact has no published download location in this index.

For UREP, adapt the active loss settings to the flat `loss_fn` interface:

```python
from urep_gsd.patch_model import loss_fn

objective = config["training"]["effective_loss"]
loss_config = dict(config["model"])
loss_config.update(
    local_loss_type=objective["local_loss"],
    local_loss_alpha=objective["local_coefficient"],
    nll_alpha=objective["local_coefficient"],
)
loss, metrics = loss_fn(outputs, targets, loss_config)
```

RT-CNN uses `tree_losses(outputs, targets, model, config["training"]["effective_loss"])`. ResNet-18 uses the recorded image-level log Smooth L1 objective. These functions specify objectives; this package has no complete training command.
