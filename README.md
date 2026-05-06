# RoadGlyph

Official code for **RoadGlyph**, an LM-free end-to-end autonomous driving model that conditions waypoint prediction on predicted scene-language tokens via an Action Conditioning Module (ACM).

---

## Overview

RoadGlyph uses a frozen **InternViT-300M-448px** vision encoder and predicts:
- **64 route waypoints** (long-horizon path)
- **10 speed waypoints** (short-horizon velocity)
- **Scene tokens**: lateral action, longitudinal action, speed context, route context

Waypoint generation is conditioned on scene tokens through an **ACM (Action Conditioning Module)** using FiLM (Feature-wise Linear Modulation). No language model is used at inference.

**V3 improvements** over the base model:
- 2nd-order finite-difference smoothness loss on speed and route waypoints (jerk suppression)
- Savitzky-Golay filtering on waypoints during inference

---

## Repository Structure

```
.
├── roadglyph/                  # Model, training, and data pipeline
│   ├── train_v3.py             # Training entry point (V3)
│   ├── config.py               # Hydra dataclass configs (V1)
│   ├── config_v3.py            # Hydra dataclass configs (V3)
│   ├── config/                 # Hydra YAML configs
│   │   ├── config_v3.yaml
│   │   └── experiment/
│   │       └── td_v3.yaml      # Main V3 experiment config
│   ├── models/
│   │   ├── road_glyph.py       # Base model (ACM + heads)
│   │   └── road_glyph_v3.py    # V3 model (+ smoothness loss)
│   ├── models/encoder/
│   │   └── internvit.py        # InternViT-300M-448px encoder
│   ├── dataloader/             # DataModule and Dataset (V3)
│   ├── utils/                  # Custom types
│   ├── data_generation/        # Slot and template ID generation
│   └── ablation/               # Ablation study scripts and results
├── team_code/                  # CARLA evaluation agents (Bench2Drive)
│   ├── agent_road_glyph_v3.py  # V3 agent (UKF + Savitzky-Golay)
│   └── agent_road_glyph.py     # Base agent
├── real_vehicle_deployment/    # Real-vehicle inference on NVIDIA DRIVE Pegasus
│   ├── inference_v3.cpp        # Main inference binary (C++)
│   ├── preprocess_cuda.cu/cuh  # GPU image preprocessing
│   ├── CMakeLists.txt
│   ├── MANUAL_inference_v3.md  # Build and run instructions
│   ├── eval_metrics.py         # MTBI / MDBI / Task SR evaluation
│   ├── can_smoothness.py       # CAN-log smoothness evaluation
│   └── results/                # Real-vehicle evaluation CSVs
├── pretrained/                 # Model checkpoint (see below)
├── start_eval_roadglyph_v3.py  # Bench2Drive SLURM evaluation orchestrator
└── environment.yaml            # Conda environment
```

---

## Installation

```bash
conda env create -f environment.yaml
conda activate roadglyph
```

---

## Pretrained Checkpoint

The pretrained V3 checkpoint and ONNX model are hosted on HuggingFace:

**[GlyphPilot/road-glyph](https://huggingface.co/GlyphPilot/road-glyph)**

| File | Description |
|------|-------------|
| `checkpoints/best.ckpt/checkpoint/mp_rank_00_model_states.pt` | PyTorch checkpoint (DeepSpeed ZeRO Stage 2, 589 MB) |
| `road_glyph_fp32_wp64_op15_v3.onnx` | ONNX model for real-vehicle deployment (1.2 GB) |

### Download checkpoint

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='GlyphPilot/road-glyph',
    filename='checkpoints/best.ckpt/checkpoint/mp_rank_00_model_states.pt',
    local_dir='pretrained',
)
"
```

The `pretrained/` directory already contains the required `latest` pointer file for DeepSpeed checkpoint loading.

---

## Training

```bash
cd /path/to/repo
python roadglyph/train_v3.py experiment=td_v3 \
    data_module.data_path=/path/to/simlingo_dataset
```

Key config options (`roadglyph/config/experiment/td_v3.yaml`):
- `data_module.data_path`: path to the SimLingo dataset
- `model.smoothness_loss_weight`: weight for speed waypoint jerk penalty (default: 0.1)
- `model.route_smoothness_loss_weight`: weight for route waypoint jerk penalty (default: 0.1)
- `trainer.gpus`: number of GPUs (trained with 8× A100)

---

## Evaluation (Bench2Drive)

### Prerequisites
- [CARLA 0.9.15](https://carla.org)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)

### Setup

Edit the USER CONFIG block at the top of `start_eval_roadglyph_v3.py`:

```python
REPO_ROOT   = "/path/to/repo"
CARLA_ROOT  = "/path/to/carla0915"
CHECKPOINT  = "/path/to/pretrained/checkpoints/best.ckpt"
OUT_ROOT    = "/path/to/eval_output"
```

### Run

```bash
python start_eval_roadglyph_v3.py
```

Results are written to `OUT_ROOT`. The script submits SLURM jobs and monitors completion automatically.

---

## Real-Vehicle Deployment

See [`real_vehicle_deployment/MANUAL_inference_v3.md`](real_vehicle_deployment/MANUAL_inference_v3.md) for full build and run instructions on NVIDIA DRIVE Pegasus (aarch64, CUDA 10.2, DriveWorks 2.2).

### Quick Start

```bash
cd real_vehicle_deployment
mkdir -p build && cd build
cmake ..
make inference_v3 -j4

# Run (waypoint prediction only, no vehicle control)
./inference_v3 --model /path/to/road_glyph_fp32_wp64_op15_v3.onnx

# Full autonomous driving with Stanley controller + data logging
./inference_v3 --model /path/to/road_glyph_fp32_wp64_op15_v3.onnx --stanley-full --log-can
```

### Evaluation

```bash
# Safety metrics (MTBI, MDBI, Task SR)
python real_vehicle_deployment/eval_metrics.py results/

# Smoothness metrics
python real_vehicle_deployment/can_smoothness.py results/can_log_YYYYMMDD_HHMMSS.csv
```

---

## Ablation Studies

See [`roadglyph/ablation/README.md`](roadglyph/ablation/README.md) for instructions on reproducing all ablation experiments (A1–C2).

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Vision encoder | InternViT-300M-448px (frozen) |
| Feature pooling | Mean pooling over patch tokens |
| ACM | FiLM modulation (γ, β from predicted scene tokens) |
| Speed waypoints | 10 points, 2D (x, y) in ego frame |
| Route waypoints | 64 points, 2D (x, y) in ego frame |
| Scene tokens | Lat action (4) · Lon action (8) · Speed kind (4) · Speed sub (6) · Route kind (3) · Route sub (5) |
| Training | DeepSpeed ZeRO Stage 2, 8× A100, 20 epochs |

---

## License

[To be specified upon paper acceptance]
