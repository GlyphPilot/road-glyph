import csv
import os
import re
from collections import defaultdict
from typing import List, Optional

IN_CSV = "unique_commentary_templates.csv"
OUT_DIR = "speed_action_groups"
SUMMARY_CSV = "speed_action_summary.csv"

SPEED_ACTIONS = [
    "remain_stopped",
    "come_to_a_stop_now",
    "maintain_current_speed",
    "maintain_reduced_speed",
    "increase_speed",
    "slow_down",
    "wait_gap",          # separated as its own label
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

def safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\- ]+", "_", name)
    name = name.replace(" ", "_")
    return name or "other"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    groups = defaultdict(list)
    total_count_sum = defaultdict(int)

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tpl = row.get("commentary_template", "") or ""
            sa = classify_speed_action(tpl)

            try:
                cnt = int(row.get("count", "0"))
            except ValueError:
                cnt = 0

            row["_count_int"] = cnt
            row["_speed_action_new"] = sa
            groups[sa].append(row)
            total_count_sum[sa] += cnt

    # 1) save one file per speed_action
    for sa, rows in groups.items():
        rows_sorted = sorted(rows, key=lambda r: r["_count_int"], reverse=True)

        out_csv = os.path.join(OUT_DIR, f"{safe_filename(sa)}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["count", "speed_action_new", "speed_action_orig", "speed_reason", "route_action", "commentary_template"])
            for r in rows_sorted:
                w.writerow([
                    r.get("count",""),
                    sa,
                    (r.get("speed_action") or "").strip(),
                    r.get("speed_reason",""),
                    (r.get("route_action") or "").strip(),
                    r.get("commentary_template",""),
                ])

        out_txt = os.path.join(OUT_DIR, f"{safe_filename(sa)}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for r in rows_sorted:
                f.write(f"[{r.get('count','')}] {r.get('commentary_template','')}\n")

    # 2) save summary
    summary_rows = []
    for sa in SPEED_ACTIONS:
        rows = groups.get(sa, [])
        summary_rows.append({
            "speed_action": sa,
            "unique_templates": len(rows),
            "total_count": total_count_sum.get(sa, 0),
        })

    summary_rows.sort(key=lambda x: (x["unique_templates"], x["total_count"]), reverse=True)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speed_action", "unique_templates", "total_count"])
        for r in summary_rows:
            w.writerow([r["speed_action"], r["unique_templates"], r["total_count"]])

    print("Done!")
    print(f"- Summary: {SUMMARY_CSV}")
    print(f"- Groups : {OUT_DIR}/ (csv + txt per speed_action)")
    print("\nTop speed_actions by unique templates:")
    for r in summary_rows[:20]:
        print(f"{r['speed_action']:22s} | unique={r['unique_templates']:4d} | total_count={r['total_count']}")

if __name__ == "__main__":
    main()
