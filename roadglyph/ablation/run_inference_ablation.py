"""
run_inference_ablation.py — Run A1/A2 inference ablation using an existing checkpoint without re-training

A1 (No-ACM): uses existing model weights, forces γ=1, β=0 only at forward time
A2 (Token-agnostic ACM): uses existing model weights, forces e_lat/e_lon=0

Weight reuse is possible because the model structure is unchanged.
(B1/B2 require re-training due to a different lon_head input dim → excluded here)

Usage (CPU):
    python roadglyph/ablation/run_inference_ablation.py \\
        --checkpoint roadglyph/outputs/2026-02-11/22-09-01/checkpoints/best-epoch=015-val_loss=0.0000.ckpt \\
        --data_path /path/to/dataset \\
        --n_samples 256 --seed 42 \\
        --output_json roadglyph/ablation/delta_wp_results.json

Usage (GPU):
    CUDA_VISIBLE_DEVICES=0 python roadglyph/ablation/run_inference_ablation.py ...
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_REPO))

# When running standalone without Hydra,
# patch get_original_cwd() to avoid errors inside dataset_base.py
import hydra.utils as _hutils
_hutils.get_original_cwd = lambda: str(_REPO)

from roadglyph.models.road_glyph import RoadGlyphModel, _remap_state_dict
from roadglyph.utils.custom_types import RoadGlyphInput

# ── legacy hparam name → current name mapping ────────────────────────────
_HPARAM_COMPAT = {
    "use_prev_template_id":  "use_prev_action_id",
    "gt_S_ratio":            "gt_ctx_ratio",
    "gt_R_ratio":            "gt_lat_ratio",
    "gt_A_ratio":            "gt_lon_ratio",
    "speed_action_weights":  "lon_action_weights",
    "route_sub_weights":     "lat_action_weights",
}

_MODEL_PARAMS = set(inspect.signature(RoadGlyphModel.__init__).parameters.keys()) - {"self"}


def _translate_hparams(hp: dict) -> dict:
    """Translate legacy hparams to current RoadGlyphModel parameter names."""
    out = {}
    for k, v in hp.items():
        new_k = _HPARAM_COMPAT.get(k, k)
        if new_k in _MODEL_PARAMS:
            out[new_k] = v
    return out


def load_deepspeed_ckpt(ckpt_path: str) -> dict:
    """Load a DeepSpeed ZeRO checkpoint.
    If the path is a directory, reads mp_rank_00_model_states.pt.
    Otherwise loads the .ckpt file directly.
    """
    p = Path(ckpt_path)
    if p.is_dir():
        candidate = p / "checkpoint" / "mp_rank_00_model_states.pt"
        if not candidate.exists():
            # must convert with zero_to_fp32.py first
            raise FileNotFoundError(
                f"{candidate} not found.\n"
                f"Run zero_to_fp32.py first:\n"
                f"  python {p}/zero_to_fp32.py {p} {p}/fp32.ckpt"
            )
        return torch.load(str(candidate), map_location="cpu")
    else:
        return torch.load(ckpt_path, map_location="cpu")


def build_model_from_ckpt(ckpt_path: str) -> RoadGlyphModel:
    """Restore RoadGlyphModel from checkpoint (including vision model)."""
    print(f"  Loading checkpoint: {ckpt_path}")
    raw = load_deepspeed_ckpt(ckpt_path)

    hp = raw.get("hyper_parameters", {})
    model_hp = _translate_hparams(hp)

    sd_raw = raw.get("module", raw.get("state_dict", {}))
    sd_remapped = _remap_state_dict(sd_raw)

    # instantiate vision model (InternViT assumed; adjust below for other encoders)
    vision_sd_keys = [k for k in sd_remapped if k.startswith("vision_model.")]
    if any("image_encoder" in k for k in vision_sd_keys):
        # InternViT
        from roadglyph.models.encoder.internvit import InternViTEncoderModel
        from omegaconf import OmegaConf
        vm_cfg = OmegaConf.create({
            "_target_": "roadglyph.models.encoder.internvit.InternViTEncoderModel",
            "variant": "OpenGVLab/InternViT-300M-448px",
            "embed_dim": hp.get("hidden_dim", 512),
            "freeze": False,
            "downsample_feature_grid_factor": 2,
            "use_global_img": False,
        })
        vision_model = InternViTEncoderModel(
            variant=vm_cfg.variant,
            embed_dim=vm_cfg.embed_dim,
            downsample_feature_grid_factor=vm_cfg.downsample_feature_grid_factor,
            use_global_img=vm_cfg.use_global_img,
        )
    elif any("resnet" in k.lower() for k in vision_sd_keys):
        from simlingo_base_training.models.encoder.resnet import ResnetEncoderModel
        vision_model = ResnetEncoderModel(variant="microsoft/resnet-34", embed_dim=512)
    else:
        raise ValueError("Cannot determine vision model type from state dict.")

    model_hp.pop("vision_model", None)
    model_hp.pop("ablation_no_acm", None)
    model_hp.pop("ablation_drop_action_tokens", None)

    model = RoadGlyphModel(vision_model=vision_model, **model_hp)

    missing, unexpected = model.load_state_dict(sd_remapped, strict=False)
    if missing:
        print(f"  [WARN] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [WARN] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    print(f"  state dict loaded (epoch={raw.get('epoch','?')})")

    # flash_attn requires fp16/bf16; disable for float32 inference
    _disable_flash_attn(model)
    print("  flash_attn disabled (float32 inference)")

    return model


def _disable_flash_attn(model: torch.nn.Module):
    count = 0
    for module in model.modules():
        if hasattr(module, "use_flash_attn"):
            module.use_flash_attn = False
            count += 1
    if count:
        print(f"  (flash_attn disabled: {count} modules)")


def build_val_loader(data_path: str, n_samples: int, seed: int, batch_size: int):
    from torch.utils.data import DataLoader, Subset
    from roadglyph.dataloader.datamodule import RoadGlyphDataModule

    dm = RoadGlyphDataModule(
        batch_size=batch_size,
        num_workers=0,
        data_path=data_path,
        bucket_path=data_path,
        encoder="internvit",
        encoder_variant="OpenGVLab/InternViT-300M-448px",
        route_as="target_point",
        hist_len=1,
        pred_len=11,
        num_route_points=64,
        cut_bottom_quarter=False,
        use_global_img=False,
        img_augmentation=False,
        img_augmentation_prob=0.0,
        img_shift_augmentation=False,
        img_shift_augmentation_prob=0.0,
        image_enhancing=False,
        use_prev_action_id=False,
        use_town13=True,
        use_old_towns=True,
        skip_first_n_frames=10,
        bucket_name="all",
        train_partitions=None,
        predict=False,
    )
    dm.setup()
    dataset = dm.val_dataset

    rng = np.random.default_rng(seed)
    n_use = min(n_samples, len(dataset))
    indices = sorted(rng.choice(len(dataset), size=n_use, replace=False).tolist())
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dm.dl_collate_fn,
    )
    print(f"  val set: {n_use}/{len(dataset)} frames (seed={seed})")
    return loader

LAT_CLASSES = 4
LON_CLASSES = 8


@torch.no_grad()
def compute_delta_wp(model: RoadGlyphModel, loader, n_samples: int, device: torch.device) -> dict:
    model.eval()
    model.to(device)

    delta_spd = np.zeros((LAT_CLASSES, LON_CLASSES))
    delta_rt  = np.zeros((LAT_CLASSES, LON_CLASSES))
    n_total = 0

    for batch_idx, batch in enumerate(loader):
        if n_total >= n_samples:
            break

        di: RoadGlyphInput = batch.driving_input

        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, (list, tuple)):
                return type(x)(_to(v) for v in x)
            return x

        di = type(di)(*(_to(f) for f in di))
        B = di.vehicle_speed.size(0)
        n_use = min(B, n_samples - n_total)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            V = model._encode_vision(di)  # [B, N, D]

        spd_base, rt_base = model._forward_from_features(V, di)
        spd_base = spd_base.float().cpu().numpy()[:n_use]
        rt_base  = (rt_base.float().cpu().numpy()[:n_use]) if rt_base is not None else None

        for k in range(LAT_CLASSES):
            for m in range(LON_CLASSES):
                spd_do, rt_do = model._forward_from_features(V, di, force_lat=k, force_lon=m)
                spd_do = spd_do.float().cpu().numpy()[:n_use]

                diff_spd = np.linalg.norm(spd_do - spd_base, axis=-1)  # [n_use, T]
                delta_spd[k, m] += diff_spd.mean()

                if rt_base is not None and rt_do is not None:
                    rt_do = rt_do.float().cpu().numpy()[:n_use]
                    diff_rt = np.linalg.norm(rt_do - rt_base, axis=-1)
                    delta_rt[k, m] += diff_rt.mean()

        n_total += n_use
        print(f"  [{batch_idx+1}] {n_total}/{n_samples} frames", end="\r", flush=True)

    print()
    n_batches = max(1, len(loader))
    delta_spd /= n_batches
    delta_rt  /= n_batches

    return {
        "n_samples":    n_total,
        "delta_wp_spd": float(delta_spd.mean()),
        "delta_wp_rt":  float(delta_rt.mean()),
        "delta_wp_sum": float(delta_spd.mean() + delta_rt.mean()),
        "per_pair": {
            "spd": delta_spd.tolist(),
            "rt":  delta_rt.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute A1/A2 Δ_wp using an existing checkpoint without re-training"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="DeepSpeed checkpoint directory or .ckpt file")
    parser.add_argument("--data_path", type=str,
                        default="/path/to/dataset")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="number of validation frames (use 16-32 for CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="use 1 for CPU to reduce memory")
    parser.add_argument("--output_json", type=str,
                        default="roadglyph/ablation/delta_wp_results.json")
    parser.add_argument("--variants", nargs="+",
                        default=["baseline", "A1_no_acm", "A2_token_agnostic"],
                        choices=["baseline", "A1_no_acm", "A2_token_agnostic"],
                        help="variants to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n device: {device}")
    if device.type == "cpu":
        print(f"  WARNING: running on CPU — n_samples={args.n_samples} may be slow.")
        print(f"  For a quick test: --n_samples 16 --batch_size 1\n")

    out_path = Path(args.output_json)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    results = dict(existing)

    print("=== Loading checkpoint ===")
    model_base = build_model_from_ckpt(args.checkpoint)

    loader = build_val_loader(args.data_path, args.n_samples, args.seed, args.batch_size)

    variant_flags = {
        "baseline":          {},
        "A1_no_acm":         {"abl_no_acm": True},
        "A2_token_agnostic": {"abl_drop_action_tokens": True},
    }

    for variant in args.variants:
        run_name = f"{variant}_seed{args.seed}"
        if run_name in results:
            print(f"\n[SKIP] {run_name} (already computed)")
            continue

        print(f"\n=== {variant} ===")
        flags = variant_flags[variant]

        for attr, val in flags.items():
            setattr(model_base, attr, val)

        result = compute_delta_wp(model_base, loader, args.n_samples, device)

        for attr in flags:
            setattr(model_base, attr, False)

        result["checkpoint"] = args.checkpoint
        result["variant"]    = variant
        results[run_name]    = result

        print(f"  Δ_wp_spd = {result['delta_wp_spd']:.4f}")
        print(f"  Δ_wp_rt  = {result['delta_wp_rt']:.4f}")
        print(f"  Δ_wp_sum = {result['delta_wp_sum']:.4f}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n" + "="*60)
    print(f"{'Variant':<35} {'Δ_spd':>7} {'Δ_rt':>7} {'Δ_sum':>7}")
    print("-"*60)
    for name, r in results.items():
        if isinstance(r, dict) and "delta_wp_spd" in r:
            print(f"{name:<35} {r['delta_wp_spd']:>7.4f} {r['delta_wp_rt']:>7.4f} {r['delta_wp_sum']:>7.4f}")
    print("="*60)

    print("\n### Markdown table")
    print("| Variant | Δ_wp_spd | Δ_wp_rt | Δ_wp_sum |")
    print("|---------|----------|---------|----------|")
    for name, r in results.items():
        if isinstance(r, dict) and "delta_wp_spd" in r:
            print(f"| {name} | {r['delta_wp_spd']:.4f} | {r['delta_wp_rt']:.4f} | {r['delta_wp_sum']:.4f} |")


if __name__ == "__main__":
    main()
