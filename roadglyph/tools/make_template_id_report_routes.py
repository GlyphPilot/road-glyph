import argparse
import gzip
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------
# Phase (commentary-based)
# ----------------------------
DURING_PREFIX = r"^\s*stay on your current\b"
AFTER_PREFIX = r"^\s*return\b"
PHASE_TO_INT = {"Before": 1, "During": 2, "After": 3}

def classify_phase(text: str) -> str:
    if re.search(DURING_PREFIX, text, flags=re.IGNORECASE):
        return "During"
    if re.search(AFTER_PREFIX, text, flags=re.IGNORECASE):
        return "After"
    return "Before"

# ----------------------------
# Route sub (Before sub-classification)
# ----------------------------
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

# ----------------------------
# Speed sub (used only in Before phase; fixed to 0 for During/After)
# ----------------------------
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

# ----------------------------
# HLC from command
# ----------------------------
def hlc_int_from_command(cmd: Optional[int]) -> int:
    # 1-4 pass through; 5/6 collapse to 5; unreadable defaults to 4
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

def read_json_gz(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Candidate keys for commentary text (fallback when exact field name is unknown)
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
    for k in COMMENT_KEYS:
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Also check common nested structures
    for k in ["meta", "data", "annotation", "annotations"]:
        v = obj.get(k, None)
        if isinstance(v, dict):
            for kk in COMMENT_KEYS:
                vv = v.get(kk, None)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return ""

def compute_template_id_fixed4(text: str, cmd: Optional[int]) -> Tuple[int, int, str, int, int, str, str]:
    """
    template_id (4-digit fixed) = [HLC][PHASE][ROUTE_SUB][SPEED_SUB]
    - During/After: route_sub=0, speed_sub=0 fixed
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
        # During/After: all sub-classes fixed to 0
        route_sub = 0
        speed_sub = 0

    tid = hlc * 1000 + phase_i * 100 + route_sub * 10 + speed_sub
    return tid, hlc, phase_str, route_sub, speed_sub, before_action, speed_action

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_xlsx", default="template_id_report.xlsx")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files; otherwise limits the number of commentary files processed")
    args = ap.parse_args()

    rows = []
    nfiles = 0

    # Walk ROOT_DIR/**/commentary/*.json.gz
    for route_name in sorted(os.listdir(args.root_dir)):
        route_dir = os.path.join(args.root_dir, route_name)
        if not os.path.isdir(route_dir):
            continue

        comm_dir = os.path.join(route_dir, "commentary")
        if not os.path.isdir(comm_dir):
            continue

        for fn in sorted(os.listdir(comm_dir)):
            if not fn.endswith(".json.gz"):
                continue

            if args.max_files and nfiles >= args.max_files:
                break

            commentary_path = os.path.join(comm_dir, fn)
            frame = fn.replace(".json.gz", "")  # "0010"
            meas_path = os.path.join(args.data_root, route_name, "measurements", f"{frame}.json.gz")

            comm_obj = read_json_gz(commentary_path) or {}
            text = extract_commentary_text(comm_obj)

            meas_obj = read_json_gz(meas_path) or {}
            cmd = meas_obj.get("command", None)
            try:
                cmd_int = int(cmd) if cmd is not None else None
            except Exception:
                cmd_int = None

            tid, hlc, phase_str, route_sub, speed_sub, before_action, speed_action = compute_template_id_fixed4(text, cmd_int)

            rows.append({
                "template_id": tid,
                "hlc": hlc,
                "phase": phase_str,
                "route_sub": route_sub,
                "speed_sub": speed_sub,
                "before_action": before_action,
                "speed_action": speed_action,
                "command": cmd_int if cmd_int is not None else "",
                "route_name": route_name,
                "frame": frame,
                "commentary_path": commentary_path,
                "measurement_path": meas_path,
                "commentary_text": text,
            })
            nfiles += 1

        if args.max_files and nfiles >= args.max_files:
            break

    if not rows:
        raise RuntimeError(
            "No commentary/*.json.gz files found. "
            "Ensure that route folders exist directly under ROOT_DIR and each contains a commentary/ subdirectory."
        )

    df = pd.DataFrame(rows)

    # summary: template_id distribution
    summary = (
        df.groupby(["template_id", "hlc", "phase", "route_sub", "speed_sub"], dropna=False)
          .size()
          .reset_index(name="num_records")
          .sort_values(["num_records", "template_id"], ascending=[False, True])
    )

    # by_route: template_id distribution per route
    by_route = (
        df.groupby(["route_name", "template_id"], dropna=False)
          .size()
          .reset_index(name="num_records")
          .sort_values(["route_name", "num_records"], ascending=[True, False])
    )

    # samples: up to 10 examples per template_id
    samples = (
        df.sort_values(["template_id", "route_name", "frame"])
          .groupby("template_id", as_index=False)
          .head(10)
    )

    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="template_id_summary", index=False)
        by_route.to_excel(writer, sheet_name="by_route", index=False)
        samples.to_excel(writer, sheet_name="samples", index=False)

    print("Done!")
    print(f"- scanned files: {len(df)}")
    print(f"- unique template_id: {df['template_id'].nunique()}")
    print(f"- saved: {args.out_xlsx}")

if __name__ == "__main__":
    main()
