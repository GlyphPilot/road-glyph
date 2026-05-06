#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import re
from glob import glob
from typing import Any, Dict, Optional


def read_gz_json(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


_num_re = re.compile(r"(\d+)\.json\.gz$")


def sort_key(p: str):
    base = os.path.basename(p)
    m = _num_re.search(base)
    return int(m.group(1)) if m else base


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_speed_reason(sr: Optional[Dict[str, Any]]) -> str:
    if not isinstance(sr, dict):
        return "speed_reason: -"

    kind = get(sr, "kind", default="-")
    distance = get(sr, "distance", default=get(sr, "static_object", "distance", default="-"))

    static_obj = get(sr, "static_object", default=None)
    dynamic_obj = get(sr, "dynamic_object", default=None)

    parts = [f"kind={fmt(kind)}", f"dist={fmt(distance)}"]

    if isinstance(static_obj, dict):
        parts.append("static_object=" + json.dumps(
            {
                "kind": get(static_obj, "kind", default=None),
                "object_id": get(static_obj, "object_id", default=None),
                "state": get(static_obj, "state", default=None),
                "distance": get(static_obj, "distance", default=None),
                "description": get(static_obj, "description", default=None),
                "visible_in_image": get(static_obj, "visible_in_image", default=None),
            },
            ensure_ascii=False
        ))
    else:
        parts.append(f"static_object={fmt(static_obj)}")

    if isinstance(dynamic_obj, dict):
        parts.append("dynamic_object=" + json.dumps(dynamic_obj, ensure_ascii=False))
    else:
        parts.append(f"dynamic_object={fmt(dynamic_obj)}")

    debug_rule = get(sr, "debug_rule", default=None)
    if debug_rule is not None:
        parts.append(f"debug_rule={fmt(debug_rule)}")

    return "speed_reason: " + " | ".join(parts)


def summarize_route_reason(rr: Optional[Dict[str, Any]]) -> str:
    if not isinstance(rr, dict):
        return "route_reason: -"

    kind = get(rr, "kind", default="-")
    rule = get(rr, "rule", default="-")
    raw = get(rr, "raw", default={}) if isinstance(get(rr, "raw", default={}), dict) else {}
    command = get(raw, "command", default=None)
    changed_route = get(raw, "changed_route", default=None)
    scenario_name = get(raw, "scenario_name", default=None)

    obj = get(rr, "object", default=get(rr, "obj", default=None))  # fallback for alternate field name

    parts = [
        f"kind={fmt(kind)}",
        f"rule={fmt(rule)}",
        f"command={fmt(command)}",
        f"changed_route={fmt(changed_route)}",
        f"scenario_name={fmt(scenario_name)}",
        f"object={fmt(obj)}",
    ]
    return "route_reason: " + " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slots_dir", help=".../slots (containing *.json.gz)")
    ap.add_argument("--pattern", default="*.json.gz", help="glob pattern (default: *.json.gz)")
    ap.add_argument("--output", default="tsv", choices=["text", "tsv"], help="output format")
    args = ap.parse_args()

    paths = sorted(glob(os.path.join(args.slots_dir, args.pattern)), key=sort_key)
    if not paths:
        raise SystemExit(f"No files matched in: {args.slots_dir}")

    if args.output == "tsv":
        # TSV header
        print("\t".join([
            "file",
            "speed_kind",
            "speed_distance",
            "static_kind",
            "static_distance",
            "static_description",
            "static_visible_in_image",
            "route_kind",
            "route_rule",
            "route_command",
            "route_changed_route",
            "route_scenario_name",
        ]))

    for p in paths:
        data = read_gz_json(p)
        sr = data.get("speed_reason")
        rr = data.get("route_reason")

        if args.output == "text":
            print(f"[{os.path.basename(p)}]")
            print(summarize_speed_reason(sr))
            print(summarize_route_reason(rr))
            print()
        else:
            speed_kind = get(sr, "kind", default=None) if isinstance(sr, dict) else None
            speed_distance = get(sr, "distance", default=get(sr, "static_object", "distance", default=None)) if isinstance(sr, dict) else None

            so = get(sr, "static_object", default=None) if isinstance(sr, dict) else None
            static_kind = get(so, "kind", default=None) if isinstance(so, dict) else None
            static_distance = get(so, "distance", default=None) if isinstance(so, dict) else None
            static_description = get(so, "description", default=None) if isinstance(so, dict) else None
            static_visible = get(so, "visible_in_image", default=None) if isinstance(so, dict) else None

            route_kind = get(rr, "kind", default=None) if isinstance(rr, dict) else None
            route_rule = get(rr, "rule", default=None) if isinstance(rr, dict) else None
            raw = get(rr, "raw", default={}) if isinstance(rr, dict) else {}
            route_command = get(raw, "command", default=None) if isinstance(raw, dict) else None
            route_changed = get(raw, "changed_route", default=None) if isinstance(raw, dict) else None
            route_scenario = get(raw, "scenario_name", default=None) if isinstance(raw, dict) else None

            print("\t".join(map(fmt, [
                os.path.basename(p),
                speed_kind,
                speed_distance,
                static_kind,
                static_distance,
                static_description,
                static_visible,
                route_kind,
                route_rule,
                route_command,
                route_changed,
                route_scenario,
            ])))


if __name__ == "__main__":
    main()
