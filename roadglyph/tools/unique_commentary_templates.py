# Commentary template types
import os
import json
import gzip
import re
import csv
from collections import Counter

ROOT_DIR = "/path/to/dataset/commentary/simlingo"
OUT_CSV = "unique_commentary_templates.csv"

def read_json_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Rule-based template classification
# -----------------------------
def classify_route_action(t: str) -> str:
    if not t:
        return ""
    s = t.lower()

    # (3) end of deviation
    if "return to your original route" in s or "return to your original lane" in s:
        return "DEVIATION_END"

    # (2) during deviation: prefix is the key indicator
    if "stay on your current lane to" in s:
        return "DEVIATION_DURING"

    # (special) waiting for a lane change gap before deviation
    if "wait for a gap in the traffic" in s:
        return "DEVIATION_BEFORE_WAIT_GAP"

    # (1) before deviation: scenario-specific action
    if "overtake" in s:
        return "DEVIATION_BEFORE_OVERTAKE"
    if "give way" in s:
        return "DEVIATION_BEFORE_GIVE_WAY"
    if "go around" in s or "circumvent" in s or "avoid" in s:
        return "DEVIATION_BEFORE_GO_AROUND"

    # default
    if s.startswith("follow the route"):
        return "FOLLOW_ROUTE"
    return "OTHER"

def classify_speed_action(t: str) -> str:
    if not t:
        return ""
    s = t.lower()

    if "remain stopped" in s:
        return "REMAIN_STOPPED"
    if "come to a stop now" in s:
        return "STOP_NOW"
    if "maintain your current speed" in s:
        return "MAINTAIN_SPEED"
    if "maintain the reduced speed" in s:
        return "MAINTAIN_REDUCED_SPEED"
    if "increase your speed" in s or "accelerate" in s:
        return "INCREASE_SPEED"
    if "slow down" in s or "decelerate" in s:
        return "SLOW_DOWN"
    if re.search(r"\bstop\b", s):
        return "STOP"
    return "UNKNOWN"

def classify_speed_reason(t: str) -> str:
    if not t:
        return ""
    s = t.lower()

    if "pedestrian" in s or "child" in s:
        return "PEDESTRIAN"
    if "traffic light" in s:
        if "red traffic light" in s:
            return "TRAFFIC_LIGHT_RED"
        if "green" in s:
            return "TRAFFIC_LIGHT_GREEN"
        return "TRAFFIC_LIGHT"
    if "stop sign" in s:
        return "STOP_SIGN"
    if "junction" in s:
        return "JUNCTION_NOTICE"
    if "construction" in s or "cones" in s:
        return "CONSTRUCTION"
    if "accident" in s:
        return "ACCIDENT"
    if "emergency vehicle" in s:
        return "EMERGENCY_VEHICLE"
    if "vehicle" in s or "car" in s or "<object>" in s:
        if "stay behind" in s or "remain behind" in s or "follow" in s:
            return "LEAD_VEHICLE_FOLLOW"
        if "avoid a collision" in s or "to avoid a collision" in s:
            return "COLLISION_AVOIDANCE"
        return "VEHICLE"
    if "to reach the target speed" in s:
        return "REACH_TARGET_SPEED"
    return "NONE_OR_UNKNOWN"

def main():
    template_counts = Counter()
    template_to_labels = {}

    n_total = 0
    n_error = 0

    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for fn in filenames:
            if not (fn.endswith(".json.gz") or fn.endswith(".json.jz")):
                continue
            n_total += 1
            fpath = os.path.join(dirpath, fn)

            try:
                d = read_json_gz(fpath)
                if not isinstance(d, dict):
                    continue

                t = d.get("commentary_template", "")
                if not t:
                    continue

                template_counts[t] += 1

                # compute labels only once per unique template
                if t not in template_to_labels:
                    ra = classify_route_action(t)
                    sa = classify_speed_action(t)
                    sr = classify_speed_reason(t)
                    template_to_labels[t] = (ra, sa, sr)

            except Exception as e:
                n_error += 1
                if n_error <= 10:
                    print(f"[error] {fpath} -> {e}")
                elif n_error == 11:
                    print("[error] too many errors... (suppressing further messages)")

            if n_total % 50000 == 0:
                print(f"[progress] scanned={n_total:,} unique_templates={len(template_to_labels):,} errors={n_error:,}")

    # Save CSV (most frequent first)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["count", "route_action", "speed_action", "speed_reason", "commentary_template"])
        for t, cnt in template_counts.most_common():
            ra, sa, sr = template_to_labels[t]
            w.writerow([cnt, ra, sa, sr, t])

    print("\nDone!")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"Unique templates: {len(template_to_labels):,}")
    print(f"Scanned files: {n_total:,}, Errors: {n_error:,}")
    print(f"Saved: {OUT_CSV}")

    # Print top 30 samples to console
    print("\nTop 30 templates:")
    for t, cnt in template_counts.most_common(30):
        ra, sa, sr = template_to_labels[t]
        print(f"{cnt:7d} | {ra:16s} | {sa:20s} | {sr:22s} | {t}")

if __name__ == "__main__":
    main()