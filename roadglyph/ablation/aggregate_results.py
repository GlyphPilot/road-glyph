"""
aggregate_results.py — Ablation study result aggregation and table generation

Combines Bench2Drive closed-loop metrics (results_*.json) with
Δ_wp metrics (delta_wp_results.json) to produce
results_ablation.csv and a Markdown table.

Usage:
    python roadglyph/ablation/aggregate_results.py \\
        --b2d_dir    eval_results/ablations/ \\
        --delta_wp   roadglyph/ablation/delta_wp_results.json \\
        --output_csv roadglyph/ablation/results_ablation.csv \\
        --output_md  roadglyph/ablation/results_ablation.md

Expected Bench2Drive JSON format (output of merge_route_json.py):
    {
      "driving_score": 0.74,
      "success_rate":  0.68,
      "efficiency":    0.91,
      "comfortness":   0.87,
      ...
    }
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional


# ── ablation metadata ───────────────────────────────────────────────────────

ABLATION_META = {
    # ── inference-only (output format of run_inference_ablation.py) ──────
    "baseline_seed42": {
        "tag": "Full",
        "desc": "RoadGlyph full model (two-pass ACM, GT cond., grad mask) — inference Δ_wp",
    },
    "A1_no_acm_seed42": {
        "tag": "A1",
        "desc": "A1: No-ACM (inference override) — γ=1, β=0 forced",
    },
    "A2_token_agnostic_seed42": {
        "tag": "A2",
        "desc": "A2: Token-agnostic ACM (inference override) — e_lat=e_lon=0",
    },
    # ── re-training ablation (trained with ablate_* YAML configs) ────────
    "ablate_A1_no_acm_seed42": {
        "tag": "A1*",
        "desc": "A1: No-ACM (re-trained) — γ=1, β=0 (heads/losses retained)",
    },
    "ablate_A2_token_agnostic_seed42": {
        "tag": "A2*",
        "desc": "A2: Token-agnostic ACM (re-trained) — e_lat, e_lon=0",
    },
    "ablate_B1_single_pass_seed42": {
        "tag": "B1",
        "desc": "B1: Single-pass — Pass-1 removed, lon head uses v_pool directly",
    },
    "ablate_B2_lon_shortcut_seed42": {
        "tag": "B2",
        "desc": "B2: Lon-head shortcut — lon head input e_ctx → v_pool",
    },
    "ablate_C1_no_gt_seed42": {
        "tag": "C1",
        "desc": "C1: No GT-conditioning — teacher forcing fully disabled",
    },
    "ablate_C2_no_grad_mask_seed42": {
        "tag": "C2",
        "desc": "C2: No grad mask — detach removed for samples without labels",
    },
}

B2D_METRICS = ["driving_score", "success_rate", "efficiency", "comfortness"]
DELTA_METRICS = ["delta_wp_spd", "delta_wp_rt", "delta_wp_sum"]
ALL_METRICS = B2D_METRICS + DELTA_METRICS


def load_b2d_result(b2d_dir: Path, run_name: str) -> Optional[Dict]:
    """Load Bench2Drive result JSON."""
    # try multiple filename patterns
    candidates = [
        b2d_dir / f"{run_name}.json",
        b2d_dir / f"{run_name}" / "results.json",
        b2d_dir / f"{run_name}" / "merged.json",
        b2d_dir / run_name / "eval_results.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def load_delta_wp(delta_wp_path: Path) -> Dict:
    """Load Δ_wp JSON."""
    if not delta_wp_path.exists():
        return {}
    with open(delta_wp_path) as f:
        return json.load(f)


def collect_rows(b2d_dir: Path, delta_wp_all: Dict) -> list:
    """Collect all metrics for each run_name."""
    rows = []
    for run_name, meta in ABLATION_META.items():
        row = {
            "run_name": run_name,
            "tag":      meta["tag"],
            "desc":     meta["desc"],
        }

        if b2d_dir is not None:
            b2d = load_b2d_result(b2d_dir, run_name)
            if b2d is not None:
                for k in B2D_METRICS:
                    row[k] = b2d.get(k, None)
            else:
                for k in B2D_METRICS:
                    row[k] = None

        dw = delta_wp_all.get(run_name, {})
        for k in DELTA_METRICS:
            row[k] = dw.get(k, None)

        rows.append(row)
    return rows


def fmt(v, fmt_str=".3f") -> str:
    if v is None:
        return "—"
    return format(float(v), fmt_str)


def write_csv(rows: list, out_path: Path):
    fieldnames = ["tag", "run_name"] + ALL_METRICS + ["desc"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"CSV saved: {out_path}")


def write_markdown(rows: list, out_path: Path):
    lines = []
    lines.append("## RoadGlyph Ablation Study Results\n")
    lines.append("| Tag | Description "
                 "| DS | SR | Eff | Cmf "
                 "| Δ_spd | Δ_rt | Δ_sum |")
    lines.append("|-----|-------------|"
                 "----|----|-----|-----"
                 "|-------|------|-------|")
    for r in rows:
        tag  = r["tag"]
        desc = r["desc"].split(" — ", 1)[-1] if " — " in r["desc"] else r["desc"]
        ds   = fmt(r.get("driving_score"))
        sr   = fmt(r.get("success_rate"))
        eff  = fmt(r.get("efficiency"))
        cmf  = fmt(r.get("comfortness"))
        dspd = fmt(r.get("delta_wp_spd"))
        drt  = fmt(r.get("delta_wp_rt"))
        dsum = fmt(r.get("delta_wp_sum"))
        lines.append(f"| **{tag}** | {desc} | {ds} | {sr} | {eff} | {cmf} | {dspd} | {drt} | {dsum} |")

    lines.append("")
    lines.append("**DS**: Driving Score, **SR**: Success Rate, **Eff**: Efficiency, **Cmf**: Comfortness")
    lines.append("**Δ_spd/rt/sum**: Token-intervention consistency (larger = more sensitive to token change)")
    lines.append("")
    lines.append("*1 seed (seed=42). Δ_wp: N=256 val frames, all (lat×lon)=(4×8) intervention pairs.*")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown saved: {out_path}")


def print_table(rows: list):
    """Print a simple table to the console."""
    header = f"{'Tag':<6} {'DS':>6} {'SR':>6} {'Eff':>6} {'Cmf':>6} {'Δ_spd':>7} {'Δ_rt':>7} {'Δ_sum':>7}"
    print("\n" + "="*60)
    print(header)
    print("-"*60)
    for r in rows:
        tag  = r.get("tag", "?")
        ds   = fmt(r.get("driving_score"))
        sr   = fmt(r.get("success_rate"))
        eff  = fmt(r.get("efficiency"))
        cmf  = fmt(r.get("comfortness"))
        dspd = fmt(r.get("delta_wp_spd"))
        drt  = fmt(r.get("delta_wp_rt"))
        dsum = fmt(r.get("delta_wp_sum"))
        print(f"{tag:<6} {ds:>6} {sr:>6} {eff:>6} {cmf:>6} {dspd:>7} {drt:>7} {dsum:>7}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results → CSV + Markdown"
    )
    parser.add_argument(
        "--b2d_dir", type=str, default=None,
        help="Bench2Drive eval results directory (per-run_name subdirectories or JSON files)"
    )
    parser.add_argument(
        "--delta_wp", type=str,
        default="roadglyph/ablation/delta_wp_results.json",
        help="output JSON from compute_delta_wp.py"
    )
    parser.add_argument(
        "--output_csv", type=str,
        default="roadglyph/ablation/results_ablation.csv"
    )
    parser.add_argument(
        "--output_md", type=str,
        default="roadglyph/ablation/results_ablation.md"
    )
    # manual input mode (read B2D metrics from existing CSV when eval results are unavailable)
    parser.add_argument(
        "--input_csv", type=str, default=None,
        help="read B2D metrics from existing CSV and merge with Δ_wp"
    )
    args = parser.parse_args()

    b2d_dir    = Path(args.b2d_dir) if args.b2d_dir else None
    delta_wp_p = Path(args.delta_wp)
    out_csv    = Path(args.output_csv)
    out_md     = Path(args.output_md)

    delta_wp_all = load_delta_wp(delta_wp_p)

    if args.input_csv and Path(args.input_csv).exists():
        # read B2D metrics from CSV (manual input workflow)
        import csv as _csv
        manual_b2d = {}
        with open(args.input_csv) as f:
            for row in _csv.DictReader(f):
                manual_b2d[row["run_name"]] = row

        rows = []
        for run_name, meta in ABLATION_META.items():
            row = {"run_name": run_name, "tag": meta["tag"], "desc": meta["desc"]}
            b2d_row = manual_b2d.get(run_name, {})
            for k in B2D_METRICS:
                v = b2d_row.get(k)
                row[k] = float(v) if v not in (None, "", "—") else None
            dw = delta_wp_all.get(run_name, {})
            for k in DELTA_METRICS:
                row[k] = dw.get(k, None)
            rows.append(row)
    else:
        rows = collect_rows(b2d_dir, delta_wp_all)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    print_table(rows)


if __name__ == "__main__":
    main()
