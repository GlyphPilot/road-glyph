"""
Dataset that extends BaseDataset with context-factor + action-token loading.
"""
import gzip
import os
import random

import cv2
import numpy as np
import ujson

from simlingo_base_training.dataloader.dataset_base import BaseDataset

# ──────────────────────────────────────────────
# Label encoding maps
# ──────────────────────────────────────────────

IGNORE = -100

# ctx_speed_kind
SPEED_KIND_MAP = {"none": 0, "dynamic": 1, "static": 2, "policy": 3}
# ctx_speed_sub
SPEED_SUBTYPE_MAP = {
    "NA": 0, "vehicle": 1, "pedestrian": 2,
    "stop_sign": 3, "traffic_light": 4, "construction_site": 5,
}
# ctx_route_kind
ROUTE_KIND_MAP = {"none": 0, "static": 1, "policy": 2}
# ctx_route_sub
ROUTE_SUBTYPE_MAP = {
    "NA": 0, "construction_site": 1, "lane_change": 2, "turn": 3, "route_adjustment": 4,
}

# lat_action_id (from template_id.before_action)
ROUTE_ACTION_MAP = {"other": 0, "go_around": 1, "overtake": 2, "give_way": 3}
# lon_action_id (from template_id.speed_action)
SPEED_ACTION_MAP = {
    "other": 0, "remain_stopped": 1, "come_to_a_stop_now": 2,
    "slow_down": 3, "maintain_current_speed": 4, "maintain_reduced_speed": 5,
    "increase_speed": 6, "wait_gap": 7,
}
# Phase
PHASE_MAP = {"Before": 0, "During": 1, "After": 2}

# Conservative priority for speed_action conflict detection
_SPEED_PRIORITY = [
    "remain_stopped", "come_to_a_stop_now", "slow_down",
    "maintain_reduced_speed", "maintain_current_speed", "increase_speed",
]


def _safe_read_json_gz(path: str):
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return ujson.load(f)
    except Exception:
        return None


def _encode_ctx_label(slot_data) -> dict:
    """Parse slots JSON → label ints (ContextFactorLabel keys)."""
    out = {
        "ctx_speed_kind": IGNORE,
        "ctx_speed_sub": IGNORE,
        "ctx_route_kind": IGNORE,
        "ctx_route_sub": IGNORE,
    }
    if slot_data is None:
        return out

    sr = slot_data.get("speed_reason")
    if isinstance(sr, dict):
        kind = sr.get("kind")
        if kind in SPEED_KIND_MAP:
            out["ctx_speed_kind"] = SPEED_KIND_MAP[kind]

            if kind == "dynamic":
                dyn = sr.get("dynamic_object")
                if isinstance(dyn, dict):
                    sk = dyn.get("kind", "NA")
                    out["ctx_speed_sub"] = SPEED_SUBTYPE_MAP.get(sk, 0)
                else:
                    out["ctx_speed_sub"] = 0  # NA
            elif kind == "static":
                sta = sr.get("static_object")
                if isinstance(sta, dict):
                    sk = sta.get("kind", "NA")
                    out["ctx_speed_sub"] = SPEED_SUBTYPE_MAP.get(sk, 0)
                else:
                    out["ctx_speed_sub"] = 0
            else:
                out["ctx_speed_sub"] = 0  # NA for none/policy

    rr = slot_data.get("route_reason")
    if isinstance(rr, dict):
        kind = rr.get("kind")
        if kind in ROUTE_KIND_MAP:
            out["ctx_route_kind"] = ROUTE_KIND_MAP[kind]
            obj = rr.get("object")
            if isinstance(obj, dict):
                ok = obj.get("kind", "NA")
                out["ctx_route_sub"] = ROUTE_SUBTYPE_MAP.get(ok, 0)
            else:
                out["ctx_route_sub"] = 0
    return out


def _encode_action_token_label(tid_data) -> dict:
    """Parse template_id JSON → label ints (ActionTokenLabel keys)."""
    out = {"lat_action_id": IGNORE, "lon_action_id": IGNORE, "phase": IGNORE}
    if tid_data is None:
        return out

    tid = tid_data.get("template_id") if isinstance(tid_data, dict) else None
    if not isinstance(tid, dict):
        return out

    # Lateral action id
    ba = tid.get("before_action")
    if isinstance(ba, str) and ba != "NA":
        out["lat_action_id"] = ROUTE_ACTION_MAP.get(ba, 0)
    else:
        out["lat_action_id"] = 0  # other

    # Longitudinal action id
    sa = tid.get("speed_action")
    if isinstance(sa, str) and sa != "NA":
        out["lon_action_id"] = SPEED_ACTION_MAP.get(sa, 0)
    else:
        out["lon_action_id"] = 0  # other

    # Phase
    phase = tid.get("phase")
    if isinstance(phase, str) and phase in PHASE_MAP:
        out["phase"] = PHASE_MAP[phase]

    return out


class RoadGlyphCARLAData(BaseDataset):
    """Extends BaseDataset with context-factor + action-token loading."""

    def __init__(self, use_prev_action_id: bool = False, **cfg):
        super().__init__(**cfg, base=True)
        self.use_prev_action_id = use_prev_action_id

    def __getitem__(self, index):
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        # ── measurements ──
        loaded_measurements, current_measurement, measurement_file_current = \
            self.load_current_and_future_measurements(measurements, sample_start)
        data["measurement_path"] = measurement_file_current

        # augmentation
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement["augmentation_rotation"]
            aug_translation = current_measurement["augmentation_translation"]
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0

        # ── waypoints ──
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
        data["speed"] = current_measurement["speed"]
        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement["target_point"])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement["target_point_next"])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        data["target_point"] = target_point
        data["map_route"] = np.array([target_point, next_target_point])

        # ── images ──
        data = self.load_images(data, images, augment_sample=augment_sample)

        # ── HLC ──
        data["hlc"] = int(current_measurement.get("command", 4))

        # ── context factor labels (slots/) ──
        meas_dir = str(measurements[0], encoding="utf-8")
        route_dir = os.path.dirname(meas_dir)
        frame_idx = sample_start + self.hist_len - 1
        slots_path = os.path.join(route_dir, "slots", f"{frame_idx:04d}.json.gz")
        ctx_factor_data = _safe_read_json_gz(slots_path)
        ctx_labels = _encode_ctx_label(ctx_factor_data)
        data.update(ctx_labels)

        # ── action token labels (template_id/) ──
        tid_path = os.path.join(route_dir, "template_id", f"{frame_idx:04d}.json.gz")
        action_token_data = _safe_read_json_gz(tid_path)
        action_labels = _encode_action_token_label(action_token_data)
        data.update(action_labels)

        # ── prev action token ──
        if self.use_prev_action_id and frame_idx > 0:
            prev_tid_path = os.path.join(route_dir, "template_id", f"{frame_idx - 1:04d}.json.gz")
            prev_action_data = _safe_read_json_gz(prev_tid_path)
            prev_action = _encode_action_token_label(prev_action_data)
            data["prev_lat_action_id"] = max(prev_action["lat_action_id"], 0)
            data["prev_lon_action_id"] = max(prev_action["lon_action_id"], 0)
            data["prev_phase"] = max(prev_action["phase"], 0)
        else:
            data["prev_lat_action_id"] = 0
            data["prev_lon_action_id"] = 0
            data["prev_phase"] = 0

        return data
