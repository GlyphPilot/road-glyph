import os
import glob
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = "/path/to/repo"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import tqdm
except ImportError:
    tqdm = None

# ============================================================
# CONFIG
# ============================================================
SIMLINGO_DATASET_ROOT = "/path/to/dataset"
DATA_ROOT = f"{SIMLINGO_DATASET_ROOT}/data/simlingo"
COMMENTARY_ROOT = f"{SIMLINGO_DATASET_ROOT}/commentary/simlingo"

# If True: merge template_id into existing slots files.
# If False: write dedicated template_id files under /template_id/.
merge_into_existing_slots = False

# ============================================================
# IO
# ============================================================
def read_json_gz(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json_gz(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

# ============================================================
# PATH MAP
# ============================================================
def commentary_to_measurements_path(commentary_path: str) -> str:
    return (
        commentary_path
        .replace("/commentary/simlingo/", "/data/simlingo/")
        .replace("/commentary/", "/measurements/")
    )

def commentary_to_slots_path(commentary_path: str) -> str:
    return (
        commentary_path
        .replace("/commentary/simlingo/", "/data/simlingo/")
        .replace("/commentary/", "/slots/")
    )

def commentary_to_templateid_path(commentary_path: str) -> str:
    return (
        commentary_path
        .replace("/commentary/simlingo/", "/data/simlingo/")
        .replace("/commentary/", "/template_id/")
    )

# ============================================================
# Commentary text extraction
# ============================================================
COMMENT_KEYS = [
    "commentary_template",
    "commentary",
    "commentary_text",
    "instruction",
    "text",
    "caption",
    "utterance",
    "message",
]

def extract_commentary_text(obj: Dict) -> str:
    if not isinstance(obj, dict):
        return ""
    for k in COMMENT_KEYS:
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ["meta", "data", "annotation", "annotations"]:
        v = obj.get(k, None)
        if isinstance(v, dict):
            for kk in COMMENT_KEYS:
                vv = v.get(kk, None)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return ""

# ============================================================
# template_id logic
# ============================================================
DURING_PREFIX = r"^\s*stay on your current\b"
AFTER_PREFIX  = r"^\s*return\b"
PHASE_TO_INT = {"Before": 1, "During": 2, "After": 3}

def classify_phase(text: str) -> str:
    if re.search(DURING_PREFIX, text, flags=re.IGNORECASE):
        return "During"
    if re.search(AFTER_PREFIX, text, flags=re.IGNORECASE):
        return "After"
    return "Before"

BEFORE_ACTION_PATTERNS = {
    "give_way": [
        r"\bgive way\b",
        r"\byield\b",
        r"\blet .* pass\b",
        r"\bright of way\b",
        r"\bshift\b",
    ],
    "go_around": [
        r"^\s*go around\b",
        r"^\s*prepare to go around\b",
        r"^\s*avoid\b",
        r"^\s*prepare to avoid\b",
    ],
    "overtake": [
        r"\bovertake\b",
        r"\bpass (?:the )?(?:vehicle|car|truck|bus)\b",
        r"\bpass it\b",
        r"\bget ahead of\b",
    ],
}

BEFORE_ACTION_TO_INT = {
    "other": 0,
    "go_around": 1,
    "overtake": 2,
    "give_way": 3,
}

def _first_match_pos(text: str, patterns: List[str]) -> Optional[int]:
    best = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            pos = m.start()
            if best is None or pos < best:
                best = pos
    return best

def classify_before_action(text: str) -> str:
    best_action = None
    best_pos = None
    priority = ["go_around", "overtake", "give_way"]
    for action in priority:
        pos = _first_match_pos(text, BEFORE_ACTION_PATTERNS.get(action, []))
        if pos is None:
            continue
        if best_pos is None or pos < best_pos:
            best_action, best_pos = action, pos
    return best_action if best_action else "other"

WAIT_GAP_PATTERNS = [
    r"\bwait for a gap\b",
    r"\bwait for a gap in the traffic\b",
    r"\bwait for a safe gap\b",
    r"\bwait for an opening\b",
]

SPEED_PATTERNS = {
    "remain_stopped": [
        r"\bremain stopped\b",
        r"\bstay stopped\b",
        r"\bkeep (?:the vehicle )?stopped\b",
    ],
    "come_to_a_stop_now": [
        r"\bcome to a stop\b",
        r"\bstop now\b",
        r"\bbrake to a stop\b",
        r"\bbring (?:the vehicle|the car) to a stop\b",
    ],
    "maintain_reduced_speed": [
        r"\bmaintain the reduced speed\b",
        r"\bmaintain a reduced speed\b",
        r"\bkeep the reduced speed\b",
    ],
    "maintain_current_speed": [
        r"\bmaintain (?:your )?current speed\b",
        r"\bkeep (?:your )?current speed\b",
    ],
    "increase_speed": [
        r"\bincrease your speed\b",
        r"\bspeed up\b",
        r"\baccelerate\b",
    ],
    "slow_down": [
        r"\bslow down\b",
        r"\breduce your speed\b",
        r"\bdecelerate\b",
        r"\bbrake\b",
    ],
}

SPEED_ACTION_TO_INT = {
    "other": 0,
    "remain_stopped": 1,
    "come_to_a_stop_now": 2,
    "slow_down": 3,
    "maintain_current_speed": 4,
    "maintain_reduced_speed": 5,
    "increase_speed": 6,
    "wait_gap": 7,
}

def classify_speed_action(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "other"
    if _first_match_pos(t, WAIT_GAP_PATTERNS) is not None:
        return "wait_gap"

    priority = [
        "remain_stopped",
        "come_to_a_stop_now",
        "maintain_reduced_speed",
        "maintain_current_speed",
        "increase_speed",
        "slow_down",
    ]
    best_action = None
    best_pos = None
    for action in priority:
        pos = _first_match_pos(t, SPEED_PATTERNS.get(action, []))
        if pos is None:
            continue
        if best_pos is None or pos < best_pos:
            best_action, best_pos = action, pos
    return best_action if best_action else "other"

def hlc_int_from_command(cmd: Optional[int]) -> int:
    if cmd is None:
        return 4
    try:
        c = int(cmd)
    except Exception:
        return 4
    if c in (5, 6):
        return 5
    if c in (1, 2, 3, 4):
        return c
    return 4

def compute_template_id_fixed4(text: str, cmd: Optional[int]) -> Dict:
    """
    template_id (4-digit) = [HLC][PHASE][ROUTE_SUB][SPEED_SUB]
    During/After: route_sub=0, speed_sub=0 (fixed)
    """
    hlc = hlc_int_from_command(cmd)
    phase_str = classify_phase(text)
    phase_i = PHASE_TO_INT[phase_str]

    route_sub = 0
    speed_sub = 0
    before_action = "NA"
    speed_action = "NA"

    if phase_str == "Before":
        before_action = classify_before_action(text)
        route_sub = BEFORE_ACTION_TO_INT.get(before_action, 0)
        speed_action = classify_speed_action(text)
        speed_sub = SPEED_ACTION_TO_INT.get(speed_action, 0)
    else:
        route_sub = 0
        speed_sub = 0

    tid = hlc * 1000 + phase_i * 100 + route_sub * 10 + speed_sub

    return {
        "template_id": tid,
        "hlc": hlc,
        "phase": phase_str,
        "route_sub": route_sub,
        "speed_sub": speed_sub,
        "before_action": before_action,
        "speed_action": speed_action,
        "command": cmd if cmd is not None else None,
        "commentary_text": text,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    com_paths = sorted(glob.glob(os.path.join(COMMENTARY_ROOT, "**/commentary/*.json.gz"), recursive=True))
    if not com_paths:
        print(f"[ERROR] No commentary found under: {COMMENTARY_ROOT}")
        return

    pbar = tqdm.tqdm(com_paths, desc="template_id for all frames") if tqdm else com_paths

    missing_meas = 0
    failed = 0
    merged = 0
    written = 0

    for com_path in pbar:
        try:
            meas_path = commentary_to_measurements_path(com_path)
            if not os.path.exists(meas_path):
                missing_meas += 1
                continue

            comm_obj = read_json_gz(com_path) or {}
            text = extract_commentary_text(comm_obj)

            meas_obj = read_json_gz(meas_path) or {}
            cmd = meas_obj.get("command", None)
            try:
                cmd_int = int(cmd) if cmd is not None else None
            except Exception:
                cmd_int = None

            tid_obj = compute_template_id_fixed4(text, cmd_int)

            if merge_into_existing_slots:
                slots_path = commentary_to_slots_path(com_path)
                slots_obj = read_json_gz(slots_path) or {}
                slots_obj["template_id"] = {
                    "id": tid_obj["template_id"],
                    "hlc": tid_obj["hlc"],
                    "phase": tid_obj["phase"],
                    "route_sub": tid_obj["route_sub"],
                    "speed_sub": tid_obj["speed_sub"],
                    "before_action": tid_obj["before_action"],
                    "speed_action": tid_obj["speed_action"],
                    "command": tid_obj["command"],
                }
                # Uncomment to also store the source commentary text:
                # slots_obj["template_id"]["commentary_text"] = tid_obj["commentary_text"]

                write_json_gz(slots_path, slots_obj)
                merged += 1
            else:
                out_path = commentary_to_templateid_path(com_path)
                write_json_gz(out_path, {
                    "source_commentary": com_path,
                    "measurements_path": meas_path,
                    "template_id": tid_obj,
                })
                written += 1

        except Exception as e:
            failed += 1
            log_path = os.path.join(DATA_ROOT, "template_id_failed_paths.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{com_path}\t{repr(e)}\n")

    print("Done!")
    print(f"- total commentary files: {len(com_paths)}")
    print(f"- missing measurements: {missing_meas}")
    print(f"- merged into slots: {merged}")
    print(f"- written template_id files: {written}")
    print(f"- failed: {failed}")

if __name__ == "__main__":
    main()
