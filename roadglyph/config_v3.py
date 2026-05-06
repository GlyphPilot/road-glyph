"""
V3 config dataclasses — extends V1 config with smoothness/consistency params.
Import this module (e.g. in train_v3.py) to register V3 configs into Hydra ConfigStore.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore

# Re-import base configs so all V1 groups remain available
from roadglyph.config import (  # noqa: F401  (side-effect: registers V1 configs)
    RoadGlyphDataModuleConfig,
    RoadGlyphTrainConfig,
    InternViTEncoderConfig,
    LLaVAnextEncoderConfig,
    ResnetEncoderConfig,
    register_configs,
)

register_configs()  # ensure V1 configs are in the store


@dataclass
class RoadGlyphModelV3Config:
    """V1 model config + V3 smoothness / consistency knobs."""

    vision_model: Any = None

    # architecture (same as V1)
    hidden_dim: int = 512
    pool_type: str = "mean"
    decoder_type: str = "mlp"
    decoder_layers: int = 2
    decoder_nhead: int = 8
    cond_emb_dim: int = 64
    film_hidden: int = 256
    predict_phase: bool = False
    predict_route_as_wps: bool = True
    num_route_wps: int = 64
    route_wp_loss_weight: float = 0.5
    route_wp_loss_warmup_epochs: int = 3
    speed_wps_mode: str = "2d"
    use_film: bool = True
    use_prev_action_id: bool = False

    # ablation (same as V1)
    ablation_no_acm: bool = False
    ablation_drop_action_tokens: bool = False
    ablation_single_pass: bool = False
    ablation_lon_from_vpool: bool = False
    ablation_no_gt_conditioning: bool = False
    ablation_no_grad_mask: bool = False

    # teacher forcing (same as V1)
    use_gt_conditioning: bool = True
    gt_ctx_ratio: float = 0.3
    gt_lat_ratio: float = 0.2
    gt_lon_ratio: float = 0.3

    # optimizer (same as V1)
    lr: float = 3e-5
    vision_lr: Optional[float] = 3e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05

    # loss class weights (same as V1)
    lon_action_weights: Optional[List[float]] = None
    lat_action_weights: Optional[List[float]] = None
    new_layer_norm_minmax: bool = False
    speed_as_input: bool = True

    # ── V3 additions ──────────────────────────────────────────────────
    smoothness_loss_weight: float = 0.1       # weight for speed_wps jerk penalty (A)
    route_smoothness_loss_weight: float = 0.1  # weight for route_wps jerk penalty (A)
    consistency_loss_weight: float = 0.5       # weight for temporal consistency loss (B)
    prev_route_dropout: float = 0.1            # prob to zero out prev_route_wps encoding

    _target_: str = "roadglyph.models.road_glyph_v3.RoadGlyphModelV3"


@dataclass
class RoadGlyphDataModuleV3Config(RoadGlyphDataModuleConfig):
    """V1 DataModule config — only _target_ changes."""
    _target_: str = "roadglyph.dataloader.datamodule_v3.RoadGlyphDataModuleV3"


@dataclass
class RoadGlyphTrainV3Config:
    """Top-level train config for V3 — model typed as V3Config for schema validation."""
    model: Any = None
    data_module: Any = None

    seed: int = 42
    gpus: int = 8
    debug: bool = False
    overfit: int = 0
    fp16_loss_scale: float = 32.0

    enable_wandb: bool = False
    wandb_project: Optional[str] = "roadglyph"
    name: Optional[str] = "test"
    wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"

    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2"
    accumulate_grad_batches: int = 1
    devices: Union[str, int] = "auto"
    val_every_n_epochs: int = 1

    resume: bool = False
    resume_path: Optional[str] = None
    checkpoint: Optional[str] = None
    weights: Optional[str] = None


def register_v3_configs():
    cs = ConfigStore.instance()
    cs.store(name="roadglyph_v3_base", node=RoadGlyphTrainV3Config)
    cs.store(group="model", name="roadglyph_v3", node=RoadGlyphModelV3Config)
    cs.store(group="data_module", name="roadglyph_v3", node=RoadGlyphDataModuleV3Config)


register_v3_configs()
