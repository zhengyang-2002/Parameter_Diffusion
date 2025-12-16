# Experiments

This folder contains scripts to generate results for the report.

## 1) Generalization: D2NWG vs GD
Compares:
- **D2NWG**: sample a 10-way ResNet-18 classifier head (5130 params) from the diffusion model, conditioned on a support set.
- **GD baseline**: train a 10-way linear head with SGD on the same support set (ResNet-18 backbone frozen).

Produces `results/generalization.csv`.

Example (fast smoke):
```bash
conda activate param_diffusion
python experiments/eval_generalization.py \
  --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
  --dataset fake --data_root ./data \
  --tasks 3 --support 1 --query 8 \
  --ddim_steps 25 --device mps
```

Suggested *unseen* datasets to test (10-way tasks):
- `cifar10` (10-way, standard)
- `svhn` (10-way digits)
- `stl10` (10-way)
- `cifar100` (sample 10 classes per task)

## 2) Ablation: sampling steps vs performance
Measures how **DDIM sampling steps** affect downstream task loss/accuracy.

Produces `results/ablation_sampling_steps.csv`.

Example:
```bash
conda activate param_diffusion
python experiments/ablation_sampling_steps.py \
  --diffusion_checkpoint ./components/diffusion/checkpoints/last.ckpt \
  --dataset cifar100 --data_root ./data \
  --tasks 20 --support 5 --query 50 \
  --steps 5,10,25,50,100 --device mps
```

Notes:
- These scripts assume the diffusion model generates **only the ResNet-18 head** (512→10 + bias), matching the 5130-parameter format used throughout this repo.
- If your diffusion checkpoint was trained for a different head size, adjust `--num_way` (and ensure the checkpoint matches).
