# DPSR Final

This repository is prepared for a reproducible Python 3.10 environment with PyTorch installed through Conda or Mamba and the remaining Python dependencies installed from [requirements.txt](./requirements.txt).

## Recommended Environment

Target platform:

- Ubuntu 22.04
- Python 3.10.6
- NVIDIA GPU
- CUDA 11.8 runtime via `pytorch-cuda`

Create and activate the environment:

```bash
conda create -n dpsr python=3.10.6
conda activate dpsr
```

Install the PyTorch stack first:

```bash
pip install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8
```

Install the remaining Python packages:

```bash
pip install -r requirements.txt
```

## Notes

- The active training code uses CUDA and NCCL distributed training.
- `mamba-ssm` and `causal-conv1d` are pinned for reproducibility because the model imports low-level Mamba operators directly.
- `causal-conv1d` is not necessary for training.
- If one of the CUDA extension builds fails, retry with:

```bash
pip install --no-build-isolation -r requirements.txt
```

