from typing import List, NamedTuple, Optional
import torch
from torch import Tensor

# Re-export base types
from simlingo_base_training.utils.custom_types import (
    DrivingLabel,
    DrivingOutput,
    ParamGroup,
    TrainingOutput,
)

IGNORE_INDEX = -100


class ContextFactorLabel(NamedTuple):
    ctx_speed_kind: Tensor       # [B] int64, 0-3 or -100
    ctx_speed_sub: Tensor        # [B] int64, 0-5 or -100
    ctx_route_kind: Tensor       # [B] int64, 0-2 or -100
    ctx_route_sub: Tensor        # [B] int64, 0-4 or -100


class ActionTokenLabel(NamedTuple):
    lat_action_id: Tensor   # [B] int64, 0-3 or -100  (lateral / route action)
    lon_action_id: Tensor   # [B] int64, 0-7 or -100  (longitudinal / speed action)
    phase: Tensor           # [B] int64, 0-2 or -100


class RoadGlyphInput(NamedTuple):
    camera_images: torch.Tensor     # [B, T, N, C, H, W] or processed
    image_sizes: torch.Tensor       # from processor
    camera_intrinsics: torch.Tensor # [B, N, 3, 3]
    camera_extrinsics: torch.Tensor # [B, N, 4, 4]
    vehicle_speed: torch.Tensor     # [B, 1]
    map_route: torch.Tensor         # [B, 2, 2] target points
    target_point: torch.Tensor      # [B, 2]
    hlc: torch.Tensor               # [B] int64, 1-6
    prev_lat_action_id: torch.Tensor   # [B] int64
    prev_lon_action_id: torch.Tensor   # [B] int64
    prev_phase: torch.Tensor           # [B] int64


class RoadGlyphExample(NamedTuple):
    driving_input: RoadGlyphInput
    driving_label: DrivingLabel
    ctx_label: ContextFactorLabel
    action_token_label: ActionTokenLabel
    run_id: Tensor
    timestamp: Tensor
