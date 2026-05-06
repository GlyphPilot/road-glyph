from typing import NamedTuple

import torch
from torch import Tensor

from roadglyph.utils.custom_types import (
    IGNORE_INDEX,
    ActionTokenLabel,
    ContextFactorLabel,
    DrivingLabel,
    DrivingOutput,
    ParamGroup,
    TrainingOutput,
)

__all__ = [
    "IGNORE_INDEX",
    "ActionTokenLabel",
    "ContextFactorLabel",
    "DrivingLabel",
    "DrivingOutput",
    "ParamGroup",
    "TrainingOutput",
    "RoadGlyphInputV3",
    "RoadGlyphExampleV3",
]


class RoadGlyphInputV3(NamedTuple):
    camera_images: Tensor       # [B, T, N, C, H, W]
    image_sizes: Tensor
    camera_intrinsics: Tensor   # [B, N, 3, 3]
    camera_extrinsics: Tensor   # [B, N, 4, 4]
    vehicle_speed: Tensor       # [B, 1]
    map_route: Tensor           # [B, 2, 2]
    target_point: Tensor        # [B, 2]
    hlc: Tensor                 # [B] int64
    prev_lat_action_id: Tensor  # [B] int64
    prev_lon_action_id: Tensor  # [B] int64
    prev_phase: Tensor          # [B] int64
    prev_route_wps: Tensor      # [B, num_route_wps, 2] — prev frame route in current ego frame


class RoadGlyphExampleV3(NamedTuple):
    driving_input: RoadGlyphInputV3
    driving_label: DrivingLabel
    ctx_label: ContextFactorLabel
    action_token_label: ActionTokenLabel
    run_id: Tensor
    timestamp: Tensor
