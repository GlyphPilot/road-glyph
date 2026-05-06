import os
import glob
import gzip
import json
from pathlib import Path
from collections import Counter
import sys

# add project root to PYTHONPATH
PROJECT_ROOT = "/path/to/repo"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    import tqdm
except ImportError:
    tqdm = None

from dataset_generation.language_labels.utils import (
    build_projection_matrix,
    is_vehicle_visible_in_image,
    get_vehicle_appearance_string,
)

# ============================================================
# CONFIG (match your training config / original generator args)
# ============================================================
SIMLINGO_DATASET_ROOT = "/path/to/dataset"
DATA_ROOT = f"{SIMLINGO_DATASET_ROOT}/data/simlingo"
COMMENTARY_ROOT = f"{SIMLINGO_DATASET_ROOT}/commentary/simlingo"

# These should match the args used in COMsGenerator
ORIGINAL_IMAGE_SIZE = (800, 600)   # <-- set to the real one you used
ORIGINAL_FOV = 90                 # <-- set to the real one you used

# ROI used by is_vehicle_visible_in_image()
MIN_X, MAX_X = 0, ORIGINAL_IMAGE_SIZE[0]
MIN_Y, MAX_Y = 0, ORIGINAL_IMAGE_SIZE[1]

# commentary logic constants
JUNCTION_LIGHT_HAZARD_MAX_DIST = 40.0
DIST_OBJ_MAX = 40.0

# lidar visibility threshold (same as original)
MIN_NUM_POINTS_VISIBLE = 3
MIN_POS_X = -1.5

MIN_WALKER_NUM_POINTS_FOR_HAZARD = 3

# ============================================================
# IO
# ============================================================
def read_json_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def write_json_gz(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        # treat sentinel "infinite" as None
        if v >= 1e6:
            return None
        return v
    except Exception:
        return None


# ============================================================
# PATH MAP
# ============================================================
def commentary_to_measurements_path(commentary_path: str) -> str:
    return commentary_path.replace("/commentary/simlingo/", "/data/simlingo/").replace("/commentary/", "/measurements/")

def commentary_to_boxes_path(commentary_path: str) -> str:
    return commentary_path.replace("/commentary/simlingo/", "/data/simlingo/").replace("/commentary/", "/boxes/")

def commentary_to_slots_path(commentary_path: str) -> str:
    return commentary_path.replace("/commentary/simlingo/", "/data/simlingo/").replace("/commentary/", "/slots/")




# ============================================================
# CORE: follow original commentary's "speed reason object" logic
# ============================================================
def is_visible_like_original(obj_box: dict, camera_matrix) -> bool:
    if not isinstance(obj_box, dict):
        return False
    # original rule (used for vehicle/walker)
    if obj_box.get("num_points", 0) is None:
        return False
    if obj_box.get("num_points", 0) <= MIN_NUM_POINTS_VISIBLE:
        return False
    pos = obj_box.get("position", None)
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 1):
        return False
    if pos[0] <= MIN_POS_X:
        return False
    return bool(is_vehicle_visible_in_image(obj_box, MIN_X, MAX_X, MIN_Y, MAX_Y, camera_matrix))

def classify_kind_from_type(speed_reduced_by_obj_type: str):
    if not isinstance(speed_reduced_by_obj_type, str) or not speed_reduced_by_obj_type:
        return None, None

    t = speed_reduced_by_obj_type.lower()

    # hazard overrides
    if "walker.hazard" in t:
        return "dynamic", "pedestrian"
    if "vehicle.hazard" in t:
        return "dynamic", "vehicle"

    # statics
    if (t == "traffic.stop") or ("stop" in t and "sign" in t):
        return "static", "stop_sign"
    if "traffic_light" in t:
        return "static", "traffic_light"
    if "trafficwarning" in t:
        return "static", "construction_site"

    # dynamics
    if ("walker" in t) or ("pedestrian" in t):
        return "dynamic", "pedestrian"
    if ("vehicle" in t) or ("car" in t) or ("truck" in t) or ("bus" in t):
        return "dynamic", "vehicle"

    return None, None


def extract_speed_reason_object(current_boxes: list, current_measurement: dict, camera_matrix):
    """
    Returns:
      reason_kind: "dynamic"|"static"|"policy"|"none"
      dynamic_object: dict|None
      static_object: dict|None
      meta/debug info
    """

    boxes_by_id = {}
    ego_info_box = None
    walker_boxes = []

    traffic_light_affecting = None  # box
    for box in current_boxes:
        if box.get("class") == "ego_info":
            ego_info_box = box
        if "id" in box:
            try:
                bid = int(box["id"])
            except Exception:
                continue
            boxes_by_id[bid] = box

        if box.get("class") == "walker":
            walker_boxes.append(box)

        # original: only traffic_light where affects_ego True
        if box.get("class") == "traffic_light" and bool(box.get("affects_ego", False)):
            # keep nearest / or just any; original keeps tuple but later manual check scans anyway
            traffic_light_affecting = box

    # measurement fields
    speed_reduced_by_obj_type = current_measurement.get("speed_reduced_by_obj_type", None)
    speed_reduced_by_obj_id = current_measurement.get("speed_reduced_by_obj_id", None)
    speed_reduced_by_obj_distance = safe_float(current_measurement.get("speed_reduced_by_obj_distance", None))

    decreased_box = None
    if speed_reduced_by_obj_id is not None:
        try:
            decreased_box = boxes_by_id.get(int(speed_reduced_by_obj_id), None)
        except Exception:
            decreased_box = None

    # apply original hazard overrides
    stop_sign_hazard = bool(current_measurement.get("stop_sign_hazard", False))
    light_hazard = bool(current_measurement.get("light_hazard", False))

    # ego_info distance_to_junction used by original override
    dist_to_junction = None
    if isinstance(ego_info_box, dict):
        dist_to_junction = safe_float(ego_info_box.get("distance_to_junction", None))

    if stop_sign_hazard:
        speed_reduced_by_obj_type = "traffic.stop"
        speed_reduced_by_obj_id = -1
        decreased_box = None
        if dist_to_junction is not None:
            speed_reduced_by_obj_distance = dist_to_junction - 5.0
        else:
            speed_reduced_by_obj_distance = None

    elif light_hazard and (dist_to_junction is not None and dist_to_junction < JUNCTION_LIGHT_HAZARD_MAX_DIST):
        speed_reduced_by_obj_type = "traffic_light"
        speed_reduced_by_obj_id = -1
        decreased_box = None
        if dist_to_junction is not None:
            speed_reduced_by_obj_distance = dist_to_junction - 5.0
        else:
            speed_reduced_by_obj_distance = None

    # original: if distance exists and >40 => clear object (policy/none)
    if speed_reduced_by_obj_distance is not None:
        speed_reduced_by_obj_distance = round(speed_reduced_by_obj_distance, 1)
        if speed_reduced_by_obj_distance > DIST_OBJ_MAX:
            speed_reduced_by_obj_type = None
            speed_reduced_by_obj_id = None
            decreased_box = None
            speed_reduced_by_obj_distance = None

    else:
        speed_reduced_by_obj_type = None
        speed_reduced_by_obj_id = None
        decreased_box = None

    # original: manual check for missed traffic light (affects_ego, dist<40, state Red)
    cause_object_at_traffic_light = False
    for box in current_boxes:
        if box.get("class") == "traffic_light" and bool(box.get("affects_ego", False)):
            d = safe_float(box.get("distance", None))
            if d is not None and d < 40 and box.get("state") == "Red":
                if decreased_box is None:
                    speed_reduced_by_obj_type = "traffic_light"
                    speed_reduced_by_obj_id = box.get("id", None)
                    decreased_box = box
                    speed_reduced_by_obj_distance = round(d, 1)
                # traffic_light_state check (optional)
                if decreased_box.get("traffic_light_state", None) == "Red":
                    cause_object_at_traffic_light = True
                break
    # ============================================================
    # ORIGINAL-LIKE HAZARD OVERRIDE (vehicle_hazard / walker_hazard)
    # ============================================================

    # 1) compute walker_hazard with lidar-hit filter like original
    walker_hazard_raw = bool(current_measurement.get("walker_hazard", False))
    walker_hazard = False
    if walker_hazard_raw:
        # original: only consider walker hazard if any walker has num_points > 3
        for w in walker_boxes:
            if w.get("num_points", 0) is not None and w.get("num_points", 0) > MIN_WALKER_NUM_POINTS_FOR_HAZARD:
                walker_hazard = True
                break

    # 2) vehicle hazard (original had an additional distance-based condition)
    vehicle_hazard = bool(current_measurement.get("vehicle_hazard", False))
    vehicle_hazard_id = current_measurement.get("vehicle_affecting_id", None)
    vehicle_hazard_box = None
    if vehicle_hazard and vehicle_hazard_id is not None:
        try:
            vehicle_hazard_box = boxes_by_id.get(int(vehicle_hazard_id), None)
        except Exception:
            vehicle_hazard_box = None

    # original condition:
    # elif vehicle_hazard and (vehicle_hazard_box exists and decreased_box exists and
    #      vehicle_hazard_box.distance <= decreased_box.distance - 5):
    vehicle_hazard_can_override = False
    if vehicle_hazard_box is not None and decreased_box is not None:
        vh_d = safe_float(vehicle_hazard_box.get("distance", None))
        dec_d = safe_float(decreased_box.get("distance", None))
        if vh_d is not None and dec_d is not None and vh_d <= (dec_d - 5.0):
            vehicle_hazard_can_override = True

    # 3) Apply hazard priority:
    # In the user-proposed priority, walker > vehicle, but in the original commentary,
    # vehicle_hazard branch comes earlier than walker_hazard branch (in the shown snippet),
    # BUT walker_hazard also has special reasoning.
    # Keep original order: vehicle_hazard override first, then walker_hazard.
    #
    # If you want "walker always wins", swap the two blocks below.

    # --- vehicle hazard override ---
    if vehicle_hazard_can_override:
        # treat as dynamic vehicle reason (cause_object becomes hazard vehicle)
        speed_reduced_by_obj_type = "vehicle.hazard"
        speed_reduced_by_obj_id = vehicle_hazard_box.get("id", None)
        decreased_box = vehicle_hazard_box
        # prefer hazard box distance; keep prior distance if missing
        speed_reduced_by_obj_distance = safe_float(vehicle_hazard_box.get("distance", None)) or speed_reduced_by_obj_distance

    # --- walker hazard override ---
    if walker_hazard:
        # prefer walker id: walker_affecting_id if exists, else speed_reduced_by_obj_id
        walker_id = current_measurement.get("walker_affecting_id", None)
        walker_box = None
        if walker_id is not None:
            try:
                walker_box = boxes_by_id.get(int(walker_id), None)
            except Exception:
                walker_box = None

        # fallback: use speed_reduced_by_obj_id if it is a walker
        if walker_box is None and decreased_box is not None and decreased_box.get("class") == "walker":
            walker_box = decreased_box

        if walker_box is not None:
            speed_reduced_by_obj_type = "walker.hazard"
            speed_reduced_by_obj_id = walker_box.get("id", None)
            decreased_box = walker_box
            speed_reduced_by_obj_distance = safe_float(walker_box.get("distance", None)) or speed_reduced_by_obj_distance

    # now finalize object + visibility
    dynamic_object = None
    static_object = None
    reason_kind = "none"
    used_rule = "no_reason"

    if speed_reduced_by_obj_type is None:
        # original would generate "policy-ish" reasons here; for slots we keep none/policy
        # pick policy if target_speed != speed_limit? (too much). Keep "none".
        return reason_kind, dynamic_object, static_object, {
            "used_rule": used_rule,
            "speed_reduced_by_obj_type": None,
            "speed_reduced_by_obj_id": None,
            "distance": None,
            "cause_object_at_traffic_light": cause_object_at_traffic_light,
        }

    # classify kind
    rk, kind = classify_kind_from_type(speed_reduced_by_obj_type)
    if rk is None:
        return "none", None, None, {
            "used_rule": "unknown_type",
            "speed_reduced_by_obj_type": speed_reduced_by_obj_type,
        }

    # stop_sign / construction_site without id: original doesn't do visibility check; but you want visible-only
    if rk == "static" and kind in ["stop_sign", "construction_site"] and decreased_box is None:
        static_object = {
            "kind": kind,
            "object_id": None,
            "state": None,
            "distance": speed_reduced_by_obj_distance,
            "description": "stop sign" if kind == "stop_sign" else "construction site",
            "visible_in_image": None,  # unknown
        }
        return "static", None, static_object, {"used_rule": f"{kind}_no_box_static_unknown_visibility", "distance": speed_reduced_by_obj_distance}

    # traffic_light: can be decreased_box (manual) or none
    if rk == "static" and kind == "traffic_light":
        # if we have a box, we can keep it; visibility for TL in original is NOT checked via is_vehicle_visible_in_image.
        # But you want visible-only. We don't have 2D bbox here, so use the same projection method by reusing is_vehicle_visible_in_image if it works for TL boxes.
        if decreased_box is None:
            return "policy", None, None, {
                "used_rule": "traffic_light_missing_box_policy",
                "distance": speed_reduced_by_obj_distance,
            }

        tl_visible = is_vehicle_visible_in_image(decreased_box, MIN_X, MAX_X, MIN_Y, MAX_Y, camera_matrix)
        if not tl_visible:
            return "policy", None, None, {
                "used_rule": "traffic_light_not_visible_policy",
                "distance": speed_reduced_by_obj_distance,
            }

        static_object = {
            "kind": "traffic_light",
            "object_id": decreased_box.get("id", None),
            "state": (decreased_box.get("state", None) or "unknown").lower() if isinstance(decreased_box.get("state", None), str) else "unknown",
            "distance": speed_reduced_by_obj_distance,
            "description": f"{(decreased_box.get('state','unknown') or 'unknown').lower()} traffic light",
        }
        return "static", None, static_object, {
            "used_rule": "traffic_light",
            "distance": speed_reduced_by_obj_distance,
        }

    # dynamic: vehicle/pedestrian require visibility (original does that)
    if rk == "dynamic":
        if decreased_box is None:
            return "policy", None, None, {
                "used_rule": f"{kind}_missing_box_policy",
                "distance": speed_reduced_by_obj_distance,
            }

        visible = is_visible_like_original(decreased_box, camera_matrix)
        if not visible:
            return "policy", None, None, {
                "used_rule": f"{kind}_not_visible_policy",
                "distance": speed_reduced_by_obj_distance,
            }

        # simplified: no age/type/color
        dynamic_object = {
            "kind": kind,  # "vehicle" | "pedestrian"
            "object_id": decreased_box.get("id", None),
            "distance": speed_reduced_by_obj_distance,
            "description": kind,
        }
        return "dynamic", dynamic_object, None, {
            "used_rule": f"{kind}_visible",
            "distance": speed_reduced_by_obj_distance,
        }

    # fallback
    return "none", None, None, {"used_rule": "fallthrough"}

def extract_route_reason(meas: dict):
    """Extract minimal route_reason from measurement only.
    Returns one of: construction_site / lane_change / turn / none / policy
    """
    if not isinstance(meas, dict):
        return {"kind": "none", "object": None, "rule": "invalid_meas"}

    scenario = meas.get("scenario_name", None)
    cmd = meas.get("command", None)  # 1 left, 2 right, 5/6 lane change
    changed_route = bool(meas.get("changed_route", False))

    scen_str = (str(scenario).lower() if scenario is not None else “”)

    if ("invadingturn" in scen_str) or ("hazardatsidelane" in scen_str) or ("trafficcone" in scen_str) or ("construction" in scen_str):
        return {
            "kind": "static",
            "object": {"kind": "construction_site", "object_id": None, "distance": None, "visible": None},
            "rule": "scenario_construction",
            "raw": {"scenario_name": scenario, "changed_route": changed_route, "command": cmd},
        }

    # command-based: lane change / turn
    if cmd in (5, 6):
        return {
            "kind": "policy",
            "object": {"kind": "lane_change", "object_id": None, "distance": None, "visible": None},
            "rule": "command_lane_change",
            "raw": {"scenario_name": scenario, "changed_route": changed_route, "command": cmd},
        }

    if cmd in (1, 2):
        return {
            "kind": "policy",
            "object": {"kind": "turn", "object_id": None, "distance": None, "visible": None},
            "rule": "command_turn",
            "raw": {"scenario_name": scenario, "changed_route": changed_route, "command": cmd},
        }

    if changed_route:
        return {
            "kind": "policy",
            "object": {"kind": "route_adjustment", "object_id": None, "distance": None, "visible": None},
            "rule": "changed_route_generic",
            "raw": {"scenario_name": scenario, "changed_route": changed_route, "command": cmd},
        }

    return {"kind": "none", "object": None, "rule": "no_route_reason", "raw": {"scenario_name": scenario, "changed_route": changed_route, "command": cmd}}


# ============================================================
# MAIN
# ============================================================
def main():
    camera_matrix = build_projection_matrix(ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1], ORIGINAL_FOV)

    com_paths = sorted(glob.glob(os.path.join(COMMENTARY_ROOT, "**/commentary/*.json.gz"), recursive=True))
    if not com_paths:
        print(f"[ERROR] No commentary found under: {COMMENTARY_ROOT}")
        return

    pbar = tqdm.tqdm(com_paths, desc="slots from original commentary speed-reason logic") if tqdm else com_paths

    stats_route_rule = Counter()
    stats_route_kind = Counter()

    failed = 0
    missing_boxes = 0
    missing_meas = 0

    for com_path in pbar:
        try:
            meas_path = commentary_to_measurements_path(com_path)
            boxes_path = commentary_to_boxes_path(com_path)

            if not os.path.exists(meas_path):
                missing_meas += 1
                continue
            if not os.path.exists(boxes_path):
                missing_boxes += 1
                continue

            meas = read_json_gz(meas_path)
            boxes = read_json_gz(boxes_path)

            reason_kind, dyn_obj, stat_obj, dbg = extract_speed_reason_object(boxes, meas, camera_matrix)
            route_reason = extract_route_reason(meas)


            out = {
                "source_commentary": com_path,
                "reason_source": "measurement+projection_like_original",

                "speed_reason": {
                    "kind": reason_kind,
                    "dynamic_object": dyn_obj,
                    "static_object": stat_obj,
                    "debug_rule": dbg.get("used_rule", None),
                    "distance": dbg.get("distance", None),
                },
                "route_reason": route_reason,

                "debug": {
                    "measurements_path": meas_path,
                    "boxes_path": boxes_path,
                },
            }


            out_path = commentary_to_slots_path(com_path)
            write_json_gz(out_path, out)

            stats_route_kind[route_reason.get("kind","unknown")] += 1
            stats_route_rule[route_reason.get("rule","unknown")] += 1


        except Exception as e:
            failed += 1
            log_path = os.path.join(DATA_ROOT, "slots_failed_paths_original_logic.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{com_path}\t{repr(e)}\n")

    summary = {
        "total": len(com_paths),
        "missing_measurements": missing_meas,
        "missing_boxes": missing_boxes,
        "failed": failed,
        "route_reason_kind_counts": dict(stats_route_kind),
        "route_reason_top_rules": dict(stats_route_rule.most_common(50)),
        "params": {
            "ORIGINAL_IMAGE_SIZE": ORIGINAL_IMAGE_SIZE,
            "ORIGINAL_FOV": ORIGINAL_FOV,
            "MIN_X": MIN_X, "MAX_X": MAX_X, "MIN_Y": MIN_Y, "MAX_Y": MAX_Y,
            "DIST_OBJ_MAX": DIST_OBJ_MAX,
            "MIN_NUM_POINTS_VISIBLE": MIN_NUM_POINTS_VISIBLE,
            "MIN_POS_X": MIN_POS_X,
        },
        "notes": [
            "This follows the original commentary generator's speed-reason object selection and visibility checks.",
            "stop_sign/construction_site without a concrete box are mapped to policy (visibility cannot be proven).",
        ],
    }

    summary_path = os.path.join(SIMLINGO_DATASET_ROOT, "slots_summary_follow_original_commentary_logic.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done!")
    print(f"- Summary: {summary_path}")
    print(f"- Missing measurements: {missing_meas}")
    print(f"- Missing boxes: {missing_boxes}")
    print(f"- Failed: {failed}")
    print("- Route reason kind counts:")
    for k, v in stats_route_kind.items():
        print(f"  {k}: {v}")



if __name__ == "__main__":
    main()
