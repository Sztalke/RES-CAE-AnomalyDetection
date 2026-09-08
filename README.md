# RES-CAE-AnomalyDetection

Residual convolutional autoencoder for image anomaly detection, written in PyTorch. The model is trained to reconstruct defect-free (OK) images only; at inference time, regions that reconstruct poorly are anomaly candidates.

The architecture is fully described in a JSON config, so encoder/decoder depth, kernel sizes, strides and attention can be changed without touching the code.

## What is in the repo

| Path | Contents |
|---|---|
| `main.py` | Training entry point: loads the config, builds the model, trains, saves the checkpoint and loss/gradient plots |
| `configs/res_cae_config.json` | Default architecture: 512×512 grayscale input, 8→512 channels, CBAM attention, residual blocks in the bottleneck |
| `models/res_cae.py` | `ResidualAutoencoder`, `ResidualBlock`, `ResidualUpBlock`, `CBAM` (channel + spatial attention) |
| `models/base.py`, `models/utils.py` | Base model with `train_step` / `evaluate`, layer factory from config |
| `training/losses.py` | MSE loss and SSIM+MSE loss (local-window SSIM, configurable kernel size) |
| `training/trainer.py` | Training loop with LR scheduling, gradient clipping, per-epoch validation/test loss, optional asynchronous TensorBoard logging |
| `datasets/CustomDataset.py`, `datasets/transforms.py` | Recursive image folder dataset (`.png`, `.jpeg`, `.bmp`, `.JPG`), grayscale or RGB, torchvision transforms |
| `utils/visualization.py` | Reconstruction previews, loss and gradient-norm plots, live OpenCV grid during training |
| `tests.ipynb` | Scratch notebook for inspecting reconstructions |

## Architecture (default config)

- **Encoder:** 7×7 and 5×5 convolutions with stride-2 downsampling and max pooling, InstanceNorm + LeakyReLU, CBAM attention on every block except the first, then two residual blocks (128→256→512 channels, stride 2).
- **Decoder:** two residual up-blocks (transposed convolutions) followed by transposed convolutions back to 1 channel, sigmoid output.
- **Regularisation:** dropout in residual blocks (0.1), AdamW with weight decay, gradient clipping.
- **Loss:** SSIM + MSE. Pure MSE tends to blur fine texture, which is exactly where small defects live; SSIM keeps local structure sharp so reconstruction error stays informative.

## Data layout

```
data/
  train/set1/        # OK images only
  validation/set1/
  test/set1/
```

Images are loaded as grayscale and resized to 512×512. Folders are scanned recursively, so subfolders per product or camera are fine. The `set{N}` suffix is selected in `main.py`.

## Running

```bash
pip install -r requirements.txt      # PyTorch 2.5 + CUDA 12.4 pinned
python main.py
```

Hyperparameters live at the top of `main.py` (`num_epochs`, `learning_rate`, `batch_size`, `clip_value`, `input_shape`). Outputs:

- `saved_models/ResCAE` – trained weights
- `losses_plot.png`, `gradient_norms_plot.png` – training curves
- TensorBoard logs (scalars, weight and gradient histograms) when logging is enabled in the trainer

## Status and limitations

- This repository contains the model and training pipeline. Anomaly scoring (per-pixel reconstruction error map, image-level score, threshold selection on a validation set) is not included here.
- No checkpointing or early stopping inside the training loop; the model is saved once at the end.
- Research code: the SSIM implementation via `torchmetrics` (`ssim_loss1`) is kept for reference but is not used.

## Requirements

Python 3.10+, PyTorch 2.5.1 (CUDA 12.4 build), torchvision, torchmetrics, OpenCV, scikit-image, matplotlib, TensorBoard. Exact versions are in `requirements.txt`.
