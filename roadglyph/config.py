from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore


@dataclass
class LLaVAnextEncoderConfig:
    variant: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = False
    _target_: str = "simlingo_base_training.models.encoder.llavanext.LLaVAnextEncoderModel"


@dataclass
class ResnetEncoderConfig:
    variant: str = "microsoft/resnet-34"
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = True
    _target_: str = "simlingo_base_training.models.encoder.resnet.ResnetEncoderModel"


@dataclass
class InternViTEncoderConfig:
    variant: str = "OpenGVLab/InternViT-300M-448px"
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = False
    _target_: str = "roadglyph.models.encoder.internvit.InternViTEncoderModel"


@dataclass
class RoadGlyphModelConfig:
    vision_model: Any

    # architecture
    hidden_dim: int = 512
    pool_type: str = "mean"           # mean | cls | attn
    decoder_type: str = "mlp"         # mlp | decoder
    decoder_layers: int = 2
    decoder_nhead: int = 8
    cond_emb_dim: int = 64
    film_hidden: int = 256
    predict_phase: bool = False
    predict_route_as_wps: bool = False
    num_route_wps: int = 64
    route_wp_loss_weight: float = 0.5
    route_wp_loss_warmup_epochs: int = 3
    speed_wps_mode: str = "2d"

    # ablation
    use_film: bool = True             # False = baseline (vision → waypoints only, no S/R/A)

    use_prev_action_id: bool = False

    # ── ablation study flags (roadglyph/ablation/) ──────────────────────────
    # A1: No-ACM – keep heads/losses but force γ=1, β=0 (identity FiLM)
    ablation_no_acm: bool = False
    # A2: Token-agnostic ACM – zero out e_lat / e_lon in ACM conditioner input
    ablation_drop_action_tokens: bool = False
    # B1: Single-pass – skip Pass-1; lon head takes v_pool directly
    ablation_single_pass: bool = False
    # B2: Lon-head shortcut – lon head input changed from e_ctx → v_pool
    ablation_lon_from_vpool: bool = False
    # C1: No GT-conditioning – disable teacher forcing from epoch 0
    ablation_no_gt_conditioning: bool = False
    # C2: No gradient masking – skip detach for samples w/o context labels
    ablation_no_grad_mask: bool = False

    # teacher forcing (ctx/lat/lon separate)
    use_gt_conditioning: bool = True
    gt_ctx_ratio: float = 0.3
    gt_lat_ratio: float = 0.2
    gt_lon_ratio: float = 0.3

    # optimizer
    lr: float = 1e-4
    vision_lr: Optional[float] = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05

    # loss class weights
    lon_action_weights: Optional[List[float]] = None
    lat_action_weights: Optional[List[float]] = None

    new_layer_norm_minmax: bool = False
    speed_as_input: bool = True

    _target_: str = "roadglyph.models.roadglyph.RoadGlyphModel"


@dataclass
class RoadGlyphDataModuleConfig:
    batch_size: int = 16
    num_workers: int = 10
    data_path: str = "/path/to/simlingo_dataset"  # override via experiment yaml or CLI
    bucket_path: str = "database/bucketsv2_simlingo"
    encoder: str = "llavanext"
    train_partitions: Optional[Dict[str, float]] = None
    cut_bottom_quarter: bool = False
    use_global_img: bool = False
    skip_first_n_frames: int = 10
    pred_len: int = 11
    hist_len: int = 3
    image_enhancing: bool = False
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5
    num_route_points: int = 64
    use_town13: bool = True
    use_old_towns: bool = True
    route_as: str = "target_point"
    use_prev_action_id: bool = False
    bucket_name: Optional[str] = "all"

    _target_: str = "roadglyph.dataloader.datamodule.RoadGlyphDataModule"


@dataclass
class RoadGlyphTrainConfig:
    model: RoadGlyphModelConfig
    data_module: Any

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


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="roadglyph_base", node=RoadGlyphTrainConfig)
    cs.store(group="data_module", name="roadglyph", node=RoadGlyphDataModuleConfig)
    cs.store(group="model", name="roadglyph", node=RoadGlyphModelConfig)
    cs.store(group="model/vision_model", name="llavanext", node=LLaVAnextEncoderConfig)
    cs.store(group="model/vision_model", name="resnet", node=ResnetEncoderConfig)
    cs.store(group="model/vision_model", name="internvit", node=InternViTEncoderConfig)


register_configs()
