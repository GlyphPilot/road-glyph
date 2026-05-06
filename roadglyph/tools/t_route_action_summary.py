import csv
import gzip
import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Tuple

IN_CSV = "unique_commentary_templates.csv"
OUT_DIR = "template_id_groups"
SUMMARY_CSV = "template_id_summary.csv"

# ----------------------------
# 1) HLC is read from measurements "command" field
# ----------------------------
# command values:
# 1: left, 2: right, 3: straight, 4: follow, 5: lane change left, 6: lane change right
# Mapping rules:
# 01 02 03 = hlc 1,2,3
# 04 = lane follow
# 05 = lane change (left/right merged)
def hlc_int_from_command(command: Optional[int]) -> int:
    if command is None:
        return 4  # fallback: lane follow
    try:
        c = int(command)
    except Exception:
        return 4

    if c in (5, 6):
        return 5
    if c in (1, 2, 3, 4):
        return c
    return 4


# ----------------------------
# 2) Deviation phase classification (commentary-based)
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
# 3) Before route sub classification (3-way, commentary-based)
# ----------------------------
BEFORE_ACTION_PATTERNS = {
    "give_way": [
        r"\bgive way\b",
        r"\byield\b",
        r"\blet .* pass\b",
        r"\bright of way\b",
        r"\bshift\b",  # "shift" maps to give_way
    ],
    "go_around": [
        # only matches when the sentence starts with one of these forms
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

# Single-digit encoding
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
    # When multiple actions appear, choose the one that occurs earliest in the text
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
# 4) Speed action classification (commentary-based)
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
        r"\bbrake\b",  # stop-related patterns are caught earlier in the priority list
    ],
}

# Single-digit encoding (0-7)
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


def classify_speed_action(commentary_template: str) -> str:
    text = (commentary_template or "").strip()
    if not text:
        return "other"

    if _first_match_pos(text, WAIT_GAP_PATTERNS) is not None:
        return "wait_gap"

    # Prioritized matching for remaining categories
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
        pos = _first_match_pos(text, SPEED_PATTERNS.get(action, []))
        if pos is None:
            continue
        if best_pos is None or pos < best_pos:
            best_action, best_pos = action, pos

    return best_action if best_action else "other"


# ----------------------------
# 5) Misc helpers
# ----------------------------
def safe_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\- ]+", "_", name)
    name = name.replace(" ", "_")
    return name or "other"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_command_from_measurement(meas_path: Optional[str]) -> Optional[int]:
    if not meas_path:
        return None
    if not os.path.exists(meas_path):
        return None

    try:
        with gzip.open(meas_path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        cmd = obj.get("command", None)
        return int(cmd) if cmd is not None else None
    except Exception:
        return None


def build_measurement_path(row: dict) -> Optional[str]:
    # (A)
    p = (row.get("measurement_path") or row.get("measurements_path") or "").strip()
    if p:
        return p

    # (B)
    route_dir = (row.get("route_dir") or row.get("route_path") or "").strip()
    frame = (row.get("frame") or row.get("measurement_id") or row.get("meas_id") or "").strip()
    if route_dir and frame:
        try:
            frame_int = int(frame)
            frame_str = f"{frame_int:04d}"
        except ValueError:
            frame_str = frame
        return os.path.join(route_dir, "measurements", f"{frame_str}.json.gz")

    return None


def compute_template_id(commentary: str, command: Optional[int]) -> Tuple[int, int, int, int, str, str]:
    """
    template_id: 4-digit fixed = [HLC][PHASE][ROUTE_SUB][SPEED_SUB]
      - HLC: 1-5
      - PHASE: 1=Before, 2=During, 3=After
      - ROUTE_SUB: classified only in Before (0-3); During/After fixed to 0
      - SPEED_SUB: speed_action class (0-7); During/After fixed to 0

    return:
      (template_id, hlc_i, phase_i, route_sub_i, phase_str, speed_action_str)
    """
    text = (commentary or "").strip()

    hlc_i = hlc_int_from_command(command)
    phase_str = classify_phase(text)
    phase_i = PHASE_TO_INT[phase_str]

    # defaults
    route_sub_i = 0
    speed_sub_i = 0
    speed_action_str = "other"

    if phase_str == "Before":
        before_action = classify_before_action(text)
        route_sub_i = BEFORE_ACTION_TO_INT.get(before_action, 0)

        speed_action_str = classify_speed_action(text)
        speed_sub_i = SPEED_ACTION_TO_INT.get(speed_action_str, 0)
    else:
        # During/After: all sub-classes fixed to 0
        route_sub_i = 0
        speed_sub_i = 0
        speed_action_str = "NA"

    template_id = hlc_i * 1000 + phase_i * 100 + route_sub_i * 10 + speed_sub_i
    return template_id, hlc_i, phase_i, route_sub_i, phase_str, speed_action_str


def main():
    ensure_dir(OUT_DIR)

    groups = defaultdict(list)          # template_id -> rows
    total_count_sum = defaultdict(int)  # template_id -> total_count

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tpl = row.get("commentary_template", "") or ""

            meas_path = build_measurement_path(row)
            cmd = read_command_from_measurement(meas_path)

            template_id, hlc_i, phase_i, route_sub_i, phase_str, speed_action_str = compute_template_id(tpl, cmd)

            try:
                cnt = int(row.get("count", "0"))
            except ValueError:
                cnt = 0

            row["_count_int"] = cnt
            row["_template_id"] = template_id
            row["_hlc_int"] = hlc_i
            row["_phase"] = phase_str
            row["_route_sub_int"] = route_sub_i
            row["_speed_action_new"] = speed_action_str
            row["_measurement_path"] = meas_path or ""
            row["_command"] = "" if cmd is None else cmd

            groups[template_id].append(row)
            total_count_sum[template_id] += cnt

    # Save one file per template_id
    for tid, rows in groups.items():
        rows_sorted = sorted(rows, key=lambda r: r["_count_int"], reverse=True)

        leaf_dir = os.path.join(OUT_DIR, str(tid))
        ensure_dir(leaf_dir)

        file_stem = safe_name(f"template_id_{tid}")

        out_csv = os.path.join(leaf_dir, f"{file_stem}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "count",
                "template_id",
                "hlc_int",
                "phase",
                "route_sub_int",
                "speed_action_new",
                "command",
                "measurement_path",
                "route_action_orig",
                "speed_action_orig",
                "speed_reason",
                "commentary_template",
            ])
            for r in rows_sorted:
                w.writerow([
                    r.get("count", ""),
                    r.get("_template_id", ""),
                    r.get("_hlc_int", ""),
                    r.get("_phase", ""),
                    r.get("_route_sub_int", ""),
                    r.get("_speed_action_new", ""),
                    r.get("_command", ""),
                    r.get("_measurement_path", ""),
                    (r.get("route_action") or "").strip(),
                    (r.get("speed_action") or "").strip(),
                    r.get("speed_reason", ""),
                    r.get("commentary_template", ""),
                ])

        out_txt = os.path.join(leaf_dir, f"{file_stem}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for r in rows_sorted:
                f.write(
                    f"[{r.get('count','')}] "
                    f"(tid={r.get('_template_id','')}, hlc={r.get('_hlc_int','')}, phase={r.get('_phase','')}, "
                    f"route_sub={r.get('_route_sub_int','')}, speed_new={r.get('_speed_action_new','')}, cmd={r.get('_command','')}) "
                    f"{r.get('commentary_template','')}\n"
                )

    # Save summary (per template_id)
    summary_rows = []
    for tid, rows in groups.items():
        summary_rows.append({
            "template_id": tid,
            "unique_templates": len(rows),
            "total_count": total_count_sum[tid],
        })

    summary_rows.sort(key=lambda x: (x["unique_templates"], x["total_count"]), reverse=True)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["template_id", "unique_templates", "total_count"])
        for r in summary_rows:
            w.writerow([r["template_id"], r["unique_templates"], r["total_count"]])

    print("Done!")
    print(f"- Summary: {SUMMARY_CSV}")
    print(f"- Groups saved under: {OUT_DIR}/")
    print("\nTop template_ids by unique templates:")
    for r in summary_rows[:20]:
        print(f"{str(r['template_id']):8s} | unique={r['unique_templates']:4d} | total_count={r['total_count']}")


if __name__ == "__main__":
    main()
