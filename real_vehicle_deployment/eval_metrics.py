"""
On-vehicle autonomous driving evaluation — MTBI / MDBI / Task SR

Analyzes the CSV produced by inference_v3 --log-can.

CSV columns:
  timestamp_us, speed_kmh, steering_angle_deg,
  override, hlc, rtk_lat, rtk_lon, rtk_valid

Metrics:
  MTBI  : Mean Time Between Interventions (seconds)
  MDBI  : Mean Distance Between Interventions (meters, RTK-based)
  SR    : Task Success Rate (%) per HLC task type

Intervention types:
  MANUAL_TAKEOVER  : override 1→0 (human takes control)
  TASK_ABORT       : intervention during an active task

Task definitions:
  HLC 1  : intersection left turn
  HLC 2  : intersection right turn
  HLC 3  : intersection straight
  HLC 5  : lane change left
  HLC 6  : lane change right
  HLC 4  : lane_follow gentle curve (auto-detected by yaw rate threshold)

Task success criterion:
  - No intervention between task HLC start and FOLLOW_LANE return
  - lane_follow curve: no intervention throughout the curve segment

Usage:
  python eval_metrics.py can_log_20260428_103000.csv
  python eval_metrics.py can_log_20260428_103000.csv --plot
  python eval_metrics.py can_log_20260428_103000.csv --min-task-dist 10
"""

import argparse
import math
import numpy as np
from pathlib import Path


HLC_LEFT         = 1
HLC_RIGHT        = 2
HLC_STRAIGHT     = 3
HLC_FOLLOW_LANE  = 4
HLC_CHANGE_LEFT  = 5
HLC_CHANGE_RIGHT = 6

HLC_NAME = {
    HLC_LEFT:        "intersection_left",
    HLC_RIGHT:       "intersection_right",
    HLC_STRAIGHT:    "intersection_straight",
    HLC_FOLLOW_LANE: "lane_follow",
    HLC_CHANGE_LEFT: "lane_change_left",
    HLC_CHANGE_RIGHT:"lane_change_right",
}

TASK_HLCS = [HLC_LEFT, HLC_RIGHT, HLC_STRAIGHT, HLC_CHANGE_LEFT, HLC_CHANGE_RIGHT]

CURVE_YAW_RATE_THRESH = 0.08   # rad/s — above this is treated as a curve
CURVE_MIN_DURATION_S  = 2.0    # minimum curve duration to count as a task (s)
MIN_TASK_DIST_M       = 5.0    # tasks shorter than this are ignored (m)
GROUP_INTERVENTION_GAP_S = 10.0  # consecutive interventions within this gap are merged

WHEELBASE   = 2.765
STEER_RATIO = 12.0


def haversine(lat1, lon1, lat2, lon2):
    """Distance between two WGS84 coordinates (m)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cum_distance(lats, lons, rtk_valid):
    """Cumulative travel distance (m). Holds previous value when RTK is invalid."""
    dist = np.zeros(len(lats))
    for i in range(1, len(lats)):
        if rtk_valid[i] > 0 and rtk_valid[i-1] > 0:
            dist[i] = dist[i-1] + haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        else:
            dist[i] = dist[i-1]
    return dist


def yaw_rate_from_can(speed_kmh, steer_deg):
    """Estimate yaw rate from speed and steering via bicycle model (rad/s)."""
    v = speed_kmh / 3.6
    wheel_rad = math.radians(steer_deg / STEER_RATIO)
    return (v / WHEELBASE) * math.tan(wheel_rad)


def load_csv(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1,
                         filling_values=0.0, invalid_raise=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    d = {}
    d['timestamp_us']       = data[:, 0]
    d['speed_kmh']          = data[:, 1]
    d['steering_angle_deg'] = data[:, 2]
    d['hlc']                = data[:, 4].astype(int)
    d['rtk_lat']            = data[:, 5]
    d['rtk_lon']            = data[:, 6]

    if ncols >= 10:
        # inference_v3 format (10 columns)
        # override_feedback (col9): CAN 0x52 byte0; value >= 1 means autonomous
        d['override']  = (data[:, 9] >= 1).astype(int)
        d['rtk_valid'] = (data[:, 7] > 0).astype(int)  # rtk_quality
    else:
        # legacy 8-column format
        d['override']  = data[:, 3].astype(int)
        d['rtk_valid'] = data[:, 7].astype(int)

    d['timestamp_s'] = (d['timestamp_us'] - d['timestamp_us'][0]) * 1e-6
    d['yaw_rate'] = np.array([
        yaw_rate_from_can(s, st)
        for s, st in zip(d['speed_kmh'], d['steering_angle_deg'])
    ])
    d['cum_dist'] = cum_distance(d['rtk_lat'], d['rtk_lon'], d['rtk_valid'])
    return d, len(data)


def detect_interventions(d):
    """Detect override 1→0 transitions as interventions."""
    interventions = []
    ovr = d['override']
    for i in range(1, len(ovr)):
        if ovr[i-1] == 1 and ovr[i] == 0:
            interventions.append({
                'idx':         i,
                'timestamp_s': d['timestamp_s'][i],
                'cum_dist_m':  d['cum_dist'][i],
                'type':        'MANUAL_TAKEOVER',
                'hlc':         d['hlc'][i],
                'hlc_name':    HLC_NAME.get(d['hlc'][i], f"hlc_{d['hlc'][i]}"),
                'speed_kmh':   d['speed_kmh'][i],
            })
    return interventions


def group_interventions(interventions, gap_s=GROUP_INTERVENTION_GAP_S):
    """Merge consecutive interventions within gap_s into a single event."""
    if not interventions:
        return []

    groups = []
    current = [interventions[0]]
    for iv in interventions[1:]:
        if iv['timestamp_s'] - current[-1]['timestamp_s'] <= gap_s:
            current.append(iv)
        else:
            groups.append(current)
            current = [iv]
    groups.append(current)

    grouped = []
    for g in groups:
        rep = g[0].copy()
        rep['n_raw']       = len(g)
        rep['group_end_s'] = g[-1]['timestamp_s']
        grouped.append(rep)
    return grouped


def compute_mtbi_mdbi(d, interventions_raw, interventions_grouped):
    """Compute MTBI and MDBI from autonomous segments."""
    ts  = d['timestamp_s']
    cd  = d['cum_dist']

    ovr = d['override'].copy()
    for iv in interventions_grouped:
        mask = (ts >= iv['timestamp_s']) & (ts <= iv['group_end_s'])
        ovr[mask] = 0

    auto_segments = []
    in_auto = False
    seg_start = 0
    for i in range(len(ovr)):
        if ovr[i] == 1 and not in_auto:
            in_auto   = True
            seg_start = i
        elif ovr[i] == 0 and in_auto:
            in_auto = False
            auto_segments.append((seg_start, i - 1))
    if in_auto:
        auto_segments.append((seg_start, len(ovr) - 1))

    durations = [ts[e] - ts[s] for s, e in auto_segments]
    distances = [cd[e] - cd[s] for s, e in auto_segments]

    total_auto_time = sum(durations)
    total_auto_dist = sum(distances)
    n_interventions = len(interventions_grouped)

    mtbi = total_auto_time / n_interventions if n_interventions > 0 else float('inf')
    mdbi = total_auto_dist / n_interventions if n_interventions > 0 else float('inf')

    return {
        'mtbi_s':              mtbi,
        'mdbi_m':              mdbi,
        'n_interventions':     n_interventions,
        'n_interventions_raw': len(interventions_raw),
        'total_auto_time_s':   total_auto_time,
        'total_auto_dist_m':   total_auto_dist,
        'n_auto_segments':     len(auto_segments),
    }


def detect_tasks(d, min_dist_m=MIN_TASK_DIST_M):
    """Detect HLC-based task segments and lane_follow curves."""
    tasks = []
    hlc   = d['hlc']
    ovr   = d['override']
    ts    = d['timestamp_s']
    cd    = d['cum_dist']
    yr    = d['yaw_rate']
    n     = len(hlc)

    # 1. HLC tasks (non-follow_lane)
    i = 0
    while i < n:
        if ovr[i] == 1 and hlc[i] in TASK_HLCS:
            task_hlc = hlc[i]
            start_i  = i
            while i < n and hlc[i] == task_hlc and ovr[i] == 1:
                i += 1
            end_i = i - 1

            dist = cd[end_i] - cd[start_i]
            if dist < min_dist_m:
                continue

            intervened = bool(np.any(ovr[start_i:end_i+1] == 0))
            completed  = (end_i + 1 < n and hlc[end_i+1] == HLC_FOLLOW_LANE)

            tasks.append({
                'type':         HLC_NAME[task_hlc],
                'hlc':          task_hlc,
                'start_s':      ts[start_i],
                'end_s':        ts[end_i],
                'duration_s':   ts[end_i] - ts[start_i],
                'dist_m':       dist,
                'intervened':   intervened,
                'completed':    completed and not intervened,
                'start_dist_m': cd[start_i],
            })
        else:
            i += 1

    # 2. lane_follow curves
    in_curve    = False
    curve_start = 0
    for i in range(n):
        is_curve = (abs(yr[i]) >= CURVE_YAW_RATE_THRESH and
                    hlc[i] == HLC_FOLLOW_LANE and ovr[i] == 1)
        if is_curve and not in_curve:
            in_curve    = True
            curve_start = i
        elif not is_curve and in_curve:
            in_curve = False
            duration = ts[i-1] - ts[curve_start]
            dist     = cd[i-1] - cd[curve_start]
            if duration >= CURVE_MIN_DURATION_S and dist >= min_dist_m:
                intervened = bool(np.any(ovr[curve_start:i] == 0))
                tasks.append({
                    'type':         'lane_follow_curve',
                    'hlc':          HLC_FOLLOW_LANE,
                    'start_s':      ts[curve_start],
                    'end_s':        ts[i-1],
                    'duration_s':   duration,
                    'dist_m':       dist,
                    'intervened':   intervened,
                    'completed':    not intervened,
                    'start_dist_m': cd[curve_start],
                })

    tasks.sort(key=lambda x: x['start_s'])
    return tasks


def compute_sr(tasks):
    """Task Success Rate per type."""
    from collections import defaultdict
    counts  = defaultdict(int)
    success = defaultdict(int)
    for t in tasks:
        counts[t['type']] += 1
        if t['completed']:
            success[t['type']] += 1
    sr = {}
    for k in counts:
        sr[k] = success[k] / counts[k] * 100 if counts[k] > 0 else 0.0
    return sr, counts, success


def print_report(mtbi_mdbi, sr, counts, success, interventions, tasks, n_samples, duration_s):
    W = 60
    print("\n" + "=" * W)
    print("  On-Vehicle Autonomous Driving Evaluation")
    print("=" * W)
    print(f"  Total samples   : {n_samples}")
    print(f"  Total duration  : {duration_s:.1f} s  ({duration_s/60:.1f} min)")
    print(f"  Autonomous time : {mtbi_mdbi['total_auto_time_s']:.1f} s")
    print(f"  Autonomous dist : {mtbi_mdbi['total_auto_dist_m']:.1f} m")
    print(f"  Autonomous segs : {mtbi_mdbi['n_auto_segments']}")
    print()

    print(f"  ── MTBI / MDBI {'─'*35}")
    n     = mtbi_mdbi['n_interventions']
    n_raw = mtbi_mdbi['n_interventions_raw']
    if n == 0:
        print(f"  No interventions")
    else:
        mtbi = mtbi_mdbi['mtbi_s']
        mdbi = mtbi_mdbi['mdbi_m']
        print(f"  Interventions  : {n}  (raw {n_raw} → grouped at 10s gap)")
        print(f"  MTBI           : {mtbi:.1f} s  ({mtbi/60:.2f} min)")
        print(f"  MDBI           : {mdbi:.1f} m  ({mdbi/1000:.3f} km)")

    print()
    print(f"  ── Intervention details {'─'*28}")
    if not interventions:
        print("  None")
    for iv in interventions:
        print(f"  [{iv['timestamp_s']:7.1f}s | {iv['cum_dist_m']:6.0f}m]  "
              f"{iv['type']:<20}  HLC:{iv['hlc_name']}  "
              f"speed:{iv['speed_kmh']:.1f}km/h")

    print()
    print(f"  ── Task Success Rate {'─'*31}")
    task_order = [
        'lane_change_left', 'lane_change_right',
        'intersection_straight', 'intersection_left', 'intersection_right',
        'lane_follow_curve',
    ]
    all_types = task_order + [k for k in counts if k not in task_order]
    for t in all_types:
        if t not in counts:
            continue
        n_t  = counts[t]
        s_t  = success[t]
        rate = sr[t]
        bar  = '█' * int(rate / 5) + '░' * (20 - int(rate / 5))
        print(f"  {t:<24} {bar}  {s_t:2d}/{n_t:2d} ({rate:5.1f}%)")

    print()
    print(f"  ── Task details {'─'*36}")
    print(f"  {'#':<3} {'type':<24} {'dur':>5}s {'dist':>6}m  result")
    print(f"  {'-'*58}")
    for i, t in enumerate(tasks):
        result = "PASS" if t['completed'] else "FAIL (intervention)"
        print(f"  {i+1:<3} {t['type']:<24} {t['duration_s']:5.1f}  "
              f"{t['dist_m']:6.0f}m  {result}")

    print("=" * W + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv",             help="path to can_log_*.csv")
    parser.add_argument("--plot",          action="store_true")
    parser.add_argument("--min-task-dist", type=float, default=MIN_TASK_DIST_M,
                        help=f"minimum task distance in meters (default: {MIN_TASK_DIST_M})")
    args = parser.parse_args()

    print(f"\n[input] {args.csv}")
    d, n_samples = load_csv(args.csv)
    duration_s   = d['timestamp_s'][-1] - d['timestamp_s'][0]

    interventions_raw = detect_interventions(d)
    interventions     = group_interventions(interventions_raw)
    mtbi_mdbi         = compute_mtbi_mdbi(d, interventions_raw, interventions)
    tasks             = detect_tasks(d, min_dist_m=args.min_task_dist)
    sr, counts, success = compute_sr(tasks)

    print_report(mtbi_mdbi, sr, counts, success, interventions, tasks, n_samples, duration_s)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            ts = d['timestamp_s']

            ax = axes[0]
            ax.fill_between(ts, d['override'], alpha=0.3, color='green', label='override (auto)')
            ax2 = ax.twinx()
            ax2.plot(ts, d['hlc'], color='navy', lw=0.8, label='HLC')
            ax2.set_yticks(list(HLC_NAME.keys()))
            ax2.set_yticklabels([HLC_NAME[k][:12] for k in HLC_NAME], fontsize=7)
            ax.set_ylabel('Override'); ax.legend(loc='upper left')
            ax.set_title('Override & HLC'); ax.grid(True, alpha=0.3)
            for iv in interventions:
                ax.axvline(iv['timestamp_s'], color='red', lw=1.5, alpha=0.8)

            ax = axes[1]
            ax.plot(ts, d['speed_kmh'], color='steelblue', lw=0.8, label='speed (km/h)')
            ax.axhline(15, color='red', ls='--', lw=0.8, label='15 km/h limit')
            for iv in interventions:
                ax.axvline(iv['timestamp_s'], color='red', lw=1.5, alpha=0.8)
            colors = {'lane_change_left':'#FFD700','lane_change_right':'#FFA500',
                      'intersection_left':'#90EE90','intersection_right':'#32CD32',
                      'intersection_straight':'#00CED1','lane_follow_curve':'#DDA0DD'}
            for t in tasks:
                c = colors.get(t['type'], '#CCCCCC')
                ax.axvspan(t['start_s'], t['end_s'], alpha=0.25, color=c)
            ax.set_ylabel('Speed (km/h)'); ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[2]
            valid = d['rtk_valid'] > 0
            if np.any(valid):
                sc = ax.scatter(d['rtk_lon'][valid], d['rtk_lat'][valid],
                                c=ts[valid], cmap='viridis', s=2)
                plt.colorbar(sc, ax=ax, label='time (s)')
                for iv in interventions:
                    idx = iv['idx']
                    if d['rtk_valid'][idx] > 0:
                        ax.plot(d['rtk_lon'][idx], d['rtk_lat'][idx], 'rv', ms=8, zorder=5)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.set_title('RTK Trajectory (▼ = intervention location)'); ax.grid(True, alpha=0.3)

            fig2, ax3 = plt.subplots(figsize=(10, 4))
            labels = list(sr.keys())
            values = [sr[k] for k in labels]
            bars   = ax3.bar(labels, values, color=['#4CAF50' if v >= 80 else '#FF7043' for v in values])
            ax3.set_ylim(0, 110)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Task Success Rate')
            ax3.axhline(100, color='gray', ls='--', lw=0.8)
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
                         f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
            plt.xticks(rotation=20, ha='right')
            plt.tight_layout()

            out1 = str(Path(args.csv).with_suffix('')) + '_eval.png'
            out2 = str(Path(args.csv).with_suffix('')) + '_sr.png'
            fig.savefig(out1, dpi=120, bbox_inches='tight')
            fig2.savefig(out2, dpi=120, bbox_inches='tight')
            print(f"[plot saved] {out1}")
            print(f"[plot saved] {out2}")
            plt.show()
        except ImportError:
            print("[warning] matplotlib not found, skipping --plot")


if __name__ == "__main__":
    main()
