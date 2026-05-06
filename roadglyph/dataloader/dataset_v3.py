"""
RoadGlyphCARLADataV3: extends V1 dataset with prev_route_wps loading.

prev_route_wps: previous frame's route_adjusted transformed into the
current ego coordinate frame (via ego_matrix).
"""
import gzip
import os
import random

import numpy as np
import ujson

from roadglyph.dataloader.dataset import (
    RoadGlyphCARLAData,
    _encode_action_token_label,
    _encode_ctx_label,
    _safe_read_json_gz,
)


def _load_measurement(meas_dir: str, frame_idx: int):
    path = os.path.join(meas_dir, f"{frame_idx:04d}.json.gz")
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return ujson.load(f)
    except Exception:
        return None


def _transform_route_to_current_ego(
    route: np.ndarray,
    prev_ego_matrix: np.ndarray,
    curr_ego_matrix: np.ndarray,
) -> np.ndarray:
    """Transform 2D route points from prev ego frame to current ego frame.

    route: [N, 2]  — points in prev ego frame (x forward, y left, z up)
    Returns: [N, 2] in current ego frame.
    """
    N = len(route)
    # Extend to 3D homogeneous (z=0 in ego frame)
    pts = np.ones((N, 4), dtype=np.float64)
    pts[:, :2] = route
    pts[:, 2] = 0.0

    # prev_ego_matrix: transforms prev-ego coords → world coords
    # curr_ego_matrix: transforms curr-ego coords → world coords
    # We want: prev-ego → world → curr-ego
    T = np.linalg.inv(curr_ego_matrix) @ prev_ego_matrix  # [4, 4]
    pts_curr = (T @ pts.T).T  # [N, 4]
    return pts_curr[:, :2].astype(np.float32)


class RoadGlyphCARLADataV3(RoadGlyphCARLAData):
    """V1 dataset + prev_route_wps in current ego frame."""

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # Load prev frame's route and transform to current ego frame
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        frame_idx = sample_start + self.hist_len - 1
        meas_dir = str(measurements[0], encoding="utf-8")

        prev_route_wps = self._load_prev_route_wps(meas_dir, frame_idx)
        data["prev_route_wps"] = prev_route_wps
        return data

    def _load_prev_route_wps(self, meas_dir: str, frame_idx: int) -> np.ndarray:
        """Load prev frame route_adjusted transformed to current ego frame."""
        zeros = np.zeros((self.num_route_points, 2), dtype=np.float32)

        if frame_idx <= 0:
            return zeros

        curr_meas = _load_measurement(meas_dir, frame_idx)
        prev_meas = _load_measurement(meas_dir, frame_idx - 1)
        if curr_meas is None or prev_meas is None:
            return zeros

        prev_route = prev_meas.get("route")
        if not prev_route:
            return zeros

        prev_ego = np.array(prev_meas.get("ego_matrix", np.eye(4)))
        curr_ego = np.array(curr_meas.get("ego_matrix", np.eye(4)))

        prev_route = np.array(prev_route, dtype=np.float64)

        # Apply same processing pipeline as load_route()
        prev_route = self.augment_route(prev_route, y_augmentation=0.0, yaw_augmentation=0.0)
        prev_route = self.equal_spacing_route(prev_route)
        if self.num_route_points > 20:
            prev_route = self.upsample_route(prev_route, self.num_route_points)

        # Ensure correct length
        N = self.num_route_points
        if len(prev_route) < N:
            pad = np.tile(prev_route[-1:], (N - len(prev_route), 1))
            prev_route = np.vstack([prev_route, pad])
        else:
            prev_route = prev_route[:N]

        # Transform from prev ego frame → current ego frame
        prev_route_curr = _transform_route_to_current_ego(prev_route, prev_ego, curr_ego)
        return prev_route_curr.astype(np.float32)
