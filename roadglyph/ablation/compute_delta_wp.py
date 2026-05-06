"""
compute_delta_wp.py — Token Intervention Consistency Metric (Δ_wp)

For N=256 validation frames:
  1. Compute baseline waypoints Y_base using predicted tokens
  2. Force each (lat_k, lon_m) combination (do-intervention) and compute Y_do(k,m)
  3. Δ_wp = mean_{k,m} mean_t ||Y_do(k,m)[t] - Y_base[t]||_2

Usage:
    python roadglyph/ablation/compute_delta_wp.py \\
        --checkpoint <path_to_checkpoint.ckpt> \\
        --experiment <ablation_experiment_name> \\  # e.g. ablate_A1_no_acm
        --data_path /path/to/dataset \\
        --n_samples 256 \\
        --seed 42 \\
        --output_json roadglyph/ablation/delta_wp_results.json

    # or process multiple checkpoints at once:
    python roadglyph/ablation/compute_delta_wp.py \\
        --checkpoint_dir results/ablations/ \\
        --output_json roadglyph/ablation/delta_wp_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_REPO))

# Hydra / OmegaConf
import hydra
from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader, Subset
from roadglyph.dataloader.dataset import RoadGlyphCARLAData
from roadglyph.dataloader.datamodule import RoadGlyphDataModule
from roadglyph.models.road_glyph import RoadGlyphModel
from roadglyph.utils.custom_types import RoadGlyphInput

# ── constants ───────────────────────────────────────────────────────────────
LAT_CLASSES = 4   # 0: other, 1: go_around, 2: overtake, 3: give_way
LON_CLASSES = 8   # 0..7


# ── main computation logic ──────────────────────────────────────────────────

@torch.no_grad()
def compute_delta_wp(
    model: RoadGlyphModel,
    dataloader: DataLoader,
    n_samples: int,
    device: torch.device,
) -> dict:
    """
    Returns:
        {
          "delta_wp_spd": float,    # mean L2 for speed waypoints
          "delta_wp_rt":  float,    # mean L2 for route waypoints (0.0 if unavailable)
          "delta_wp_sum": float,
          "per_pair": {             # per-(lat,lon) mean L2 for debugging
            "spd": [[4,8] list],
            "rt":  [[4,8] list],
          }
        }
    """
    model.eval()

    all_delta_spd = np.zeros((LAT_CLASSES, LON_CLASSES))
    all_delta_rt  = np.zeros((LAT_CLASSES, LON_CLASSES))
    n_collected   = 0

    for batch in dataloader:
        if n_collected >= n_samples:
            break

        # batch: RoadGlyphExample
        di: RoadGlyphInput = batch.driving_input

        # move to GPU
        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, (list, tuple)):
                cls = type(x)
                return cls(_to(v) for v in x)
            return x

        di = type(di)(*(_to(f) for f in di))

        # Encode vision features (once, cached)
        with torch.cuda.amp.autocast():
            V = model._encode_vision(di)

        # Baseline waypoints (using predicted tokens)
        with torch.cuda.amp.autocast():
            spd_base, rt_base = model._forward_from_features(V, di)

        spd_base = spd_base.float().cpu().numpy()   # [B, T_spd, 2]
        rt_base  = rt_base.float().cpu().numpy() if rt_base is not None else None

        B = spd_base.shape[0]
        n_use = min(B, n_samples - n_collected)

        # do-intervention for each (lat_k, lon_m) pair
        for k in range(LAT_CLASSES):
            for m in range(LON_CLASSES):
                with torch.cuda.amp.autocast():
                    spd_do, rt_do = model._forward_from_features(
                        V, di,
                        force_lat=k,
                        force_lon=m,
                    )
                spd_do = spd_do.float().cpu().numpy()  # [B, T, 2]
                rt_do  = rt_do.float().cpu().numpy() if rt_do is not None else None

                # L2 per waypoint, mean over time, mean over batch
                diff_spd = np.linalg.norm(spd_do[:n_use] - spd_base[:n_use], axis=-1)  # [n_use, T]
                all_delta_spd[k, m] += diff_spd.mean()  # mean over t and b

                if rt_base is not None and rt_do is not None:
                    diff_rt = np.linalg.norm(rt_do[:n_use] - rt_base[:n_use], axis=-1)
                    all_delta_rt[k, m] += diff_rt.mean()

        n_collected += n_use
        print(f"  collected {n_collected}/{n_samples} frames", end="\r", flush=True)

    print()
    n_batches = max(1, -(-n_samples // dataloader.batch_size))  # ceil
    # divide by number of batches to average the accumulated values
    n_b_actual = n_collected / dataloader.batch_size
    all_delta_spd /= max(1, n_b_actual)
    all_delta_rt  /= max(1, n_b_actual)

    delta_wp_spd = float(all_delta_spd.mean())
    delta_wp_rt  = float(all_delta_rt.mean())
    delta_wp_sum = delta_wp_spd + delta_wp_rt

    return {
        "n_samples": n_collected,
        "delta_wp_spd": delta_wp_spd,
        "delta_wp_rt":  delta_wp_rt,
        "delta_wp_sum": delta_wp_sum,
        "per_pair": {
            "spd": all_delta_spd.tolist(),
            "rt":  all_delta_rt.tolist(),
        },
    }


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> RoadGlyphModel:
    """Load RoadGlyphModel from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # restore Hydra hyperparameters
    hparams = ckpt.get("hyper_parameters", {})
    if not hparams:
        raise ValueError(
            f"No hyper_parameters found in checkpoint: {ckpt_path}\n"
            "RoadGlyphModel must use save_hyperparameters()."
        )

    # instantiate vision_model
    vision_model_cfg = hparams.pop("vision_model")
    if isinstance(vision_model_cfg, DictConfig):
        vision_model = hydra.utils.instantiate(vision_model_cfg)
    else:
        raise ValueError("Cannot read vision_model config as DictConfig.")

    model = RoadGlyphModel(vision_model=vision_model, **hparams)
    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    return model


def build_val_loader(data_path: str, n_samples: int, seed: int, batch_size: int = 4) -> DataLoader:
    """Sample n_samples frames from the validation set using a fixed seed."""
    from roadglyph.dataloader.dataset import RoadGlyphCARLAData

    dataset = RoadGlyphCARLAData(
        data_path=data_path,
        split="val",
        pred_len=11,
        hist_len=1,
        num_route_points=20,
        route_as="target_point",
        use_prev_action_id=False,
    )

    # sample indices with a fixed seed
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    indices = sorted(indices.tolist())

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    return loader


def run_single(args, ckpt_path: str, run_name: str, existing: dict) -> dict:
    """Compute Δ_wp for a single checkpoint and return the result dict."""
    if run_name in existing:
        print(f"[SKIP] {run_name} (already computed)")
        return existing[run_name]

    print(f"\n{'='*60}")
    print(f"[Δ_wp] {run_name}")
    print(f"  ckpt : {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = load_model_from_checkpoint(ckpt_path, device)

    loader = build_val_loader(
        data_path=args.data_path,
        n_samples=args.n_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"  val samples: {args.n_samples}, seed: {args.seed}")

    result = compute_delta_wp(model, loader, args.n_samples, device)
    result["checkpoint"] = ckpt_path
    result["run_name"]   = run_name

    print(f"  Δ_wp_spd = {result['delta_wp_spd']:.4f}")
    print(f"  Δ_wp_rt  = {result['delta_wp_rt']:.4f}")
    print(f"  Δ_wp_sum = {result['delta_wp_sum']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute Δ_wp token intervention consistency metric")
    # single checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a single .ckpt file")
    parser.add_argument("--run_name", type=str, default=None,
                        help="run name to use in results (when using --checkpoint)")
    # multi-checkpoint directory
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="directory containing ablate_* subdirectories")

    parser.add_argument("--data_path", type=str,
                        default="/path/to/dataset")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_json", type=str,
                        default="roadglyph/ablation/delta_wp_results.json")
    args = parser.parse_args()

    # load existing results if present (resume from prior run)
    out_path = Path(args.output_json)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    results = dict(existing)

    if args.checkpoint is not None:
        run_name = args.run_name or Path(args.checkpoint).stem
        r = run_single(args, args.checkpoint, run_name, results)
        results[run_name] = r

    elif args.checkpoint_dir is not None:
        # find the latest ckpt under each ablate_* subdirectory
        ckpt_dir = Path(args.checkpoint_dir)
        for run_dir in sorted(ckpt_dir.glob("ablate_*")):
            if not run_dir.is_dir():
                continue
            ckpts = sorted(run_dir.glob("**/*.ckpt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpts:
                print(f"[WARN] {run_dir}: no .ckpt found, skipping")
                continue
            ckpt_path = str(ckpts[0])
            run_name  = run_dir.name
            r = run_single(args, ckpt_path, run_name, results)
            results[run_name] = r

    else:
        parser.error("Specify either --checkpoint or --checkpoint_dir.")

    # save results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # summary
    print("\n" + "="*60)
    print(f"{'Run':<40} {'Δ_spd':>8} {'Δ_rt':>8} {'Δ_sum':>8}")
    print("-"*60)
    for name, r in results.items():
        if isinstance(r, dict) and "delta_wp_spd" in r:
            print(f"{name:<40} {r['delta_wp_spd']:>8.4f} {r['delta_wp_rt']:>8.4f} {r['delta_wp_sum']:>8.4f}")


if __name__ == "__main__":
    main()
