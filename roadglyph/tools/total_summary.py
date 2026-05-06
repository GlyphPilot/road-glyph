# total_template_summary.py
import csv
import gzip
import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Tuple

IN_CSV = "unique_commentary_templates.csv"
OUT_DIR = "total_template_groups"
SUMMARY_CSV = "total_template_summary.csv"

# ----------------------------
# 1) HLC is read from the measurements "command" field
# ----------------------------
# command values:
# 1: left, 2: right, 3: straight, 4: follow, 5: lane change left, 6: lane change right
COMMAND_TO_HLC = {
    1: "turn_left",
    2: "turn_right",
    3: "follow_the_route",
    4: "follow_the_route",
    5: "lane_change",
    6: "lane_change",
}

# ----------------------------
# 2) Deviation phase classification (commentary-based)
# ----------------------------
DURING_PREFIX = r"^\s*stay on your current\b"
AFTER_PREFIX = r"^\s*return\b"

# ----------------------------
# 3) Before sub-classification (commentary-based)
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

# ----------------------------
# 4) Speed action classification (commentary-based)
# ----------------------------
SPEED_ACTIONS = [
    "remain_stopped",
    "come_to_a_stop_now",
    "maintain_current_speed",
    "maintain_reduced_speed",
    "increase_speed",
    "slow_down",
    "wait_gap",
    "other",
]

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


def _first_match_pos(text: str, patterns: List[str]) -> Optional[int]:
    best = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            pos = m.start()
            if best is None or pos < best:
                best = pos
    return best


def safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\- ]+", "_", name)
    name = name.replace(" ", "_")
    return name or "other"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Route-side classifiers
# ----------------------------
def classify_phase(text: str) -> str:
    if re.search(DURING_PREFIX, text, flags=re.IGNORECASE):
        return "During"
    if re.search(AFTER_PREFIX, text, flags=re.IGNORECASE):
        return "After"
    return "Before"


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


def classify_hlc_from_command(command: Optional[int]) -> str:
    if command is None:
        return "follow_the_route"
    return COMMAND_TO_HLC.get(command, "follow_the_route")


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


# ----------------------------
# Speed-side classifier
# ----------------------------
def classify_speed_action(commentary_template: str) -> str:
    text = (commentary_template or "").strip()
    if not text:
        return "other"

    # 0) wait_gap is its own label (highest priority)
    if _first_match_pos(text, WAIT_GAP_PATTERNS) is not None:
        return "wait_gap"

    # 1) classify remaining actions by priority order
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
# Bucket (route + speed)
# ----------------------------
def bucket_key_and_path(commentary: str, hlc: str, speed_action_new: str) -> Tuple[str, str]:
    """
    Folder structure:
      - Before: OUT_DIR/hlc/Before/before_action/speed_action_new/
      - During/After: OUT_DIR/hlc/During(or After)/speed_action_new/
    """
    text = (commentary or "").strip()
    phase = classify_phase(text)

    if phase == "Before":
        before_action = classify_before_action(text)
        leaf_dir = os.path.join(OUT_DIR, hlc, phase, before_action, speed_action_new)
        bucket_key = f"{hlc}/{phase}/{before_action}/{speed_action_new}"
    else:
        leaf_dir = os.path.join(OUT_DIR, hlc, phase, speed_action_new)
        bucket_key = f"{hlc}/{phase}/{speed_action_new}"

    return bucket_key, leaf_dir


def main():
    ensure_dir(OUT_DIR)

    buckets = defaultdict(list)         # bucket_key -> rows
    total_count_sum = defaultdict(int)  # bucket_key -> total_count

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tpl = row.get("commentary_template", "") or ""

            # 1) route side
            meas_path = build_measurement_path(row)
            cmd = read_command_from_measurement(meas_path)
            hlc = classify_hlc_from_command(cmd)

            # 2) speed side
            speed_action_new = classify_speed_action(tpl)

            # 3) combined bucket
            bucket_key, leaf_dir = bucket_key_and_path(tpl, hlc, speed_action_new)

            try:
                cnt = int(row.get("count", "0"))
            except ValueError:
                cnt = 0

            row["_count_int"] = cnt
            row["_bucket_key"] = bucket_key
            row["_leaf_dir"] = leaf_dir
            row["_measurement_path"] = meas_path or ""
            row["_command"] = "" if cmd is None else cmd
            row["_hlc"] = hlc
            row["_speed_action_new"] = speed_action_new

            buckets[bucket_key].append(row)
            total_count_sum[bucket_key] += cnt

    # Save one file per bucket
    for bucket_key, rows in buckets.items():
        leaf_dir = rows[0]["_leaf_dir"]
        ensure_dir(leaf_dir)

        rows_sorted = sorted(rows, key=lambda r: r["_count_int"], reverse=True)
        file_stem = safe_name(bucket_key.replace("/", "__"))

        out_csv = os.path.join(leaf_dir, f"{file_stem}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "count",
                "command", "hlc",
                "speed_action_new",
                "bucket",
                "measurement_path",
                "route_action_orig", "speed_action_orig", "speed_reason",
                "commentary_template",
            ])
            for r in rows_sorted:
                w.writerow([
                    r.get("count", ""),
                    r.get("_command", ""),
                    r.get("_hlc", ""),
                    r.get("_speed_action_new", ""),
                    bucket_key,
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
                    f"(cmd={r.get('_command','')}, hlc={r.get('_hlc','')}, speed_new={r.get('_speed_action_new','')}) "
                    f"{r.get('commentary_template','')}\n"
                )

    # Save summary (per bucket)
    summary_rows = []
    for bucket_key, rows in buckets.items():
        summary_rows.append({
            "bucket": bucket_key,
            "unique_templates": len(rows),
            "total_count": total_count_sum[bucket_key],
        })

    summary_rows.sort(key=lambda x: (x["unique_templates"], x["total_count"]), reverse=True)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "unique_templates", "total_count"])
        for r in summary_rows:
            w.writerow([r["bucket"], r["unique_templates"], r["total_count"]])

    print("Done!")
    print(f"- Summary: {SUMMARY_CSV}")
    print(f"- Buckets saved under: {OUT_DIR}/")
    print("\nTop buckets by unique templates:")
    for r in summary_rows[:30]:
        print(f"{r['bucket'][:70]:70s} | unique={r['unique_templates']:4d} | total_count={r['total_count']}")


if __name__ == "__main__":
    main()
