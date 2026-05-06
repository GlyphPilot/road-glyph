"""
RoadGlyphModel: LM-free vision encoder + ACM conditioning + Context/Lat/Lon heads.
"""
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
import pytorch_lightning as pl
from torchvision.models import ResNet18_Weights, resnet18

from simlingo_base_training.models.adaptors.adaptors import NormZeroOne, VectorInputAdaptor
from simlingo_base_training.models.utils import configure_params_groups, summarise_losses
from simlingo_base_training.utils.custom_types import DrivingLabel, ParamGroup, TrainingOutput

from roadglyph.utils.custom_types import (
    IGNORE_INDEX,
    RoadGlyphExample,
    RoadGlyphInput,
    ContextFactorLabel,
    ActionTokenLabel,
)

# ──────────────────────────────────────────────
# Pooling
# ──────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, D]
        w = self.attn(x)                       # [B, N, 1]
        w = torch.softmax(w, dim=1)
        return (w * x).sum(dim=1)              # [B, D]


def build_pooler(pool_type: str, dim: int):
    if pool_type == "attn":
        return AttentionPool(dim)
    return None  # mean / cls handled inline


# ──────────────────────────────────────────────
# Context Head  (scene-reason classification)
# ──────────────────────────────────────────────

class ContextHead(nn.Module):
    """4 classification heads from pooled vision features."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.speed_kind   = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        self.speed_sub    = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 6))
        self.route_kind   = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.route_sub    = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 5))

    def forward(self, v_pool: Tensor) -> Dict[str, Tensor]:
        return {
            "speed_kind":   self.speed_kind(v_pool),
            "speed_sub":    self.speed_sub(v_pool),
            "route_kind":   self.route_kind(v_pool),
            "route_sub":    self.route_sub(v_pool),
        }



# ──────────────────────────────────────────────
# Lateral Action Head  (route-action)
# ──────────────────────────────────────────────

class LateralActionHead(nn.Module):
    def __init__(self, dim: int, hlc_dim: int, hidden: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim + hlc_dim, hidden), nn.SiLU(), nn.Linear(hidden, 4)
        )

    def forward(self, v_pool: Tensor, e_hlc: Tensor) -> Tensor:
        return self.head(torch.cat([v_pool, e_hlc], dim=-1))



# ──────────────────────────────────────────────
# Longitudinal Action Head  (speed-action)
# ──────────────────────────────────────────────

class LongitudinalActionHead(nn.Module):
    def __init__(self, s_emb_dim: int, spd_dim: int, hidden: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(s_emb_dim + spd_dim, hidden), nn.SiLU(), nn.Linear(hidden, 8)
        )

    def forward(self, e_ctx: Tensor, e_spd: Tensor) -> Tensor:
        return self.head(torch.cat([e_ctx, e_spd], dim=-1))



# ──────────────────────────────────────────────
# ACM Conditioner (Action-Conditioned Modulation)
# ──────────────────────────────────────────────

class ACMConditioner(nn.Module):
    """Soft-embedding ACM: predicted (or GT) class → embedding → gamma/beta."""

    def __init__(
        self,
        out_dim: int,
        cond_emb_dim: int = 64,
        hlc_classes: int = 7,
        spd_hidden: int = 64,
        film_hidden: int = 256,
        use_prev_action_id: bool = False,
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.use_prev_action_id = use_prev_action_id

        # Soft-embedding tables for Context
        self.emb_speed_kind  = nn.Embedding(4, cond_emb_dim)
        self.emb_speed_sub   = nn.Embedding(6, cond_emb_dim)
        self.emb_route_kind  = nn.Embedding(3, cond_emb_dim)
        self.emb_route_sub   = nn.Embedding(5, cond_emb_dim)
        # Soft-embedding tables for Lat / Lon actions
        self.emb_lat = nn.Embedding(4, cond_emb_dim)
        self.emb_lon = nn.Embedding(8, cond_emb_dim)

        # HLC / speed encoders
        self.hlc_emb = nn.Embedding(hlc_classes, cond_emb_dim)
        self.spd_mlp = nn.Sequential(
            nn.Linear(1, spd_hidden), nn.SiLU(), nn.Linear(spd_hidden, cond_emb_dim)
        )

        # Optional prev action id
        if self.use_prev_action_id:
            self.prev_lat_emb = nn.Embedding(5, cond_emb_dim)    # 0-3 + pad(4)
            self.prev_lon_emb = nn.Embedding(9, cond_emb_dim)    # 0-7 + pad(8)
            self.prev_phase_emb = nn.Embedding(4, cond_emb_dim)  # 0-2 + pad(3)

        # Ctx(4) + Lat + Lon + hlc + spd
        n_slots = 4 + 1 + 1 + 1 + 1
        if self.use_prev_action_id:
            n_slots += 3  # prev R/A/phase
        in_dim = n_slots * cond_emb_dim

        self.film_mlp = nn.Sequential(
            nn.Linear(in_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * out_dim),
        )

    @property
    def s_emb_dim(self) -> int:
        return 4 * self.cond_emb_dim

    def _soft_embed(self, logits: Tensor, emb: nn.Embedding) -> Tensor:
        p = F.softmax(logits.float(), dim=-1)  # [B, C] float32 for numerical stability
        w = emb.weight                         # [C, cond_emb_dim]
        return (p.to(w.dtype) @ w)             # [B, cond_emb_dim]

    def _gt_embed(self, labels: Tensor, emb: nn.Embedding) -> Tensor:
        safe = labels.clamp(min=0)
        return emb(safe)                       # [B, cond_emb_dim]

    def forward(
        self,
        ctx_logits: Dict[str, Tensor],
        lat_logits: Tensor,
        lon_logits: Tensor,
        hlc: Tensor,
        speed: Tensor,
        # teacher-forcing
        use_gt_ctx: bool = False,
        use_gt_lat: bool = False,
        use_gt_lon: bool = False,
        gt_ctx: Optional[ContextFactorLabel] = None,
        gt_action: Optional[ActionTokenLabel] = None,
        # prev action id
        prev_lat_action_id: Optional[Tensor] = None,
        prev_lon_action_id: Optional[Tensor] = None,
        prev_phase: Optional[Tensor] = None,
        # ablation
        drop_action_tokens: bool = False,  # A2: zero e_lat / e_lon
        drop_all_tokens: bool = False,     # A3: zero e_ctx / e_lat / e_lon
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns e_ctx_flat [B, 4*C], gamma [B, D], beta [B, D]."""

        # --- Context embeddings ---
        if use_gt_ctx and gt_ctx is not None:
            e_sk = self._gt_embed(gt_ctx.ctx_speed_kind, self.emb_speed_kind)
            e_ss = self._gt_embed(gt_ctx.ctx_speed_sub, self.emb_speed_sub)
            e_rk = self._gt_embed(gt_ctx.ctx_route_kind, self.emb_route_kind)
            e_rs = self._gt_embed(gt_ctx.ctx_route_sub, self.emb_route_sub)
        else:
            e_sk = self._soft_embed(ctx_logits["speed_kind"], self.emb_speed_kind)
            e_ss = self._soft_embed(ctx_logits["speed_sub"], self.emb_speed_sub)
            e_rk = self._soft_embed(ctx_logits["route_kind"], self.emb_route_kind)
            e_rs = self._soft_embed(ctx_logits["route_sub"], self.emb_route_sub)
        e_ctx = torch.cat([e_sk, e_ss, e_rk, e_rs], dim=-1)  # [B, 4*C]
        if drop_all_tokens:  # A3: remove all token conditioning
            e_ctx = torch.zeros_like(e_ctx)

        # --- Lateral action embedding ---
        if use_gt_lat and gt_action is not None:
            e_lat = self._gt_embed(gt_action.lat_action_id, self.emb_lat)
        else:
            e_lat = self._soft_embed(lat_logits, self.emb_lat)
        if drop_action_tokens or drop_all_tokens:  # A2/A3: remove action-token conditioning
            e_lat = torch.zeros_like(e_lat)

        # --- Longitudinal action embedding ---
        if use_gt_lon and gt_action is not None:
            e_lon = self._gt_embed(gt_action.lon_action_id, self.emb_lon)
        else:
            e_lon = self._soft_embed(lon_logits, self.emb_lon)
        if drop_action_tokens or drop_all_tokens:  # A2/A3: remove action-token conditioning
            e_lon = torch.zeros_like(e_lon)

        # --- HLC / speed ---
        e_hlc = self.hlc_emb(hlc)
        e_spd = self.spd_mlp(speed)

        parts = [e_ctx, e_lat, e_lon, e_hlc, e_spd]

        # --- optional prev action id ---
        if self.use_prev_action_id and prev_lat_action_id is not None:
            parts.append(self.prev_lat_emb(prev_lat_action_id.clamp(min=0)))
            parts.append(self.prev_lon_emb(prev_lon_action_id.clamp(min=0)))
            parts.append(self.prev_phase_emb(prev_phase.clamp(min=0)))

        cond = torch.cat(parts, dim=-1)       # [B, in_dim]
        gamma_beta = self.film_mlp(cond)       # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        return e_ctx, gamma, beta



# ──────────────────────────────────────────────
# Waypoint heads
# ──────────────────────────────────────────────

class WaypointMLPHead(nn.Module):
    """Simple MLP: pool(V_mod) → waypoints."""

    def __init__(self, dim: int, num_wps: int = 10, out_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.num_wps = num_wps
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_wps * out_dim),
        )
        self.out_dim = out_dim

    def forward(self, v_mod: Tensor) -> Tensor:
        # v_mod: [B, N, D] → pool → [B, D]
        v_pool = v_mod.mean(dim=1)
        return self.mlp(v_pool).view(-1, self.num_wps, self.out_dim).cumsum(1)


class WaypointDecoderHead(nn.Module):
    """Lightweight TransformerDecoder: queries cross-attend to V_mod."""

    def __init__(self, dim: int, num_wps: int = 10, out_dim: int = 2,
                 num_layers: int = 2, nhead: int = 8, mlp_dim: int = 256):
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(num_wps, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=mlp_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(), nn.Linear(mlp_dim, out_dim))

    def forward(self, v_mod: Tensor) -> Tensor:
        B = v_mod.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(q, v_mod)  # [B, num_wps, D]
        return self.head(out).cumsum(1)


# ──────────────────────────────────────────────
# Route encoder (reuse pattern)
# ──────────────────────────────────────────────

class RouteEncode(nn.Module):
    def __init__(self, out_channels: int, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_channels)

    def forward(self, route):
        x = route.to(self.backbone.fc.weight.dtype) / 128.0 - 1.0
        return self.backbone(x)


# ──────────────────────────────────────────────
# Checkpoint key mapping (old → new)
# ──────────────────────────────────────────────
#
# Generation 1 (SlotDriving era):
#   slot_head.*           → ctx_head.*
#   route_action_head.*   → lat_action_head.*
#   speed_action_head.*   → lon_action_head.*
#   film.*                → acm.*
#
# Generation 2 (first-pass RoadGlyph rename):
#   No state_dict key changes (class rename only, internal attributes unchanged).
#
# TODO (second-pass): if ACM embedding tables are renamed (emb_R → emb_lat,
#   emb_A → emb_lon), add those mappings here.

_CKPT_KEY_MAP = [
    ("slot_head.", "ctx_head."),
    ("route_action_head.", "lat_action_head."),
    ("speed_action_head.", "lon_action_head."),
    ("film.", "acm."),
    ("acm.emb_R.", "acm.emb_lat."),
    ("acm.emb_A.", "acm.emb_lon."),
    ("acm.prev_R_emb.", "acm.prev_lat_emb."),
    ("acm.prev_A_emb.", "acm.prev_lon_emb."),
]


def _remap_state_dict(state_dict: dict) -> dict:
    """Remap old checkpoint keys to current names.

    No break — allows chained remapping:
    e.g. film.emb_R.weight → acm.emb_R.weight → acm.emb_lat.weight
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in _CKPT_KEY_MAP:
            if old_prefix in new_key:
                new_key = new_key.replace(old_prefix, new_prefix, 1)
        new_sd[new_key] = v
    return new_sd


# ──────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────

class RoadGlyphModel(pl.LightningModule):
    def __init__(
        self,
        vision_model,
        # architecture
        hidden_dim: int = 512,
        pool_type: str = "mean",          # mean | cls | attn
        decoder_type: str = "mlp",        # mlp | decoder
        decoder_layers: int = 2,
        decoder_nhead: int = 8,
        cond_emb_dim: int = 64,
        film_hidden: int = 256,
        predict_phase: bool = False,
        predict_route_as_wps: bool = False,
        num_route_wps: int = 20,
        route_wp_loss_weight: float = 0.5,
        route_wp_loss_warmup_epochs: int = 3,
        speed_wps_mode: str = "2d",
        route_as: str = "target_point",
        # ablation
        use_film: bool = True,
        use_prev_action_id: bool = False,
        # ablation study flags
        ablation_no_acm: bool = False,
        ablation_drop_action_tokens: bool = False,
        ablation_drop_all_tokens: bool = False,
        ablation_single_pass: bool = False,
        ablation_lon_from_vpool: bool = False,
        ablation_no_gt_conditioning: bool = False,
        ablation_no_grad_mask: bool = False,
        # teacher forcing
        use_gt_conditioning: bool = True,
        gt_ctx_ratio: float = 0.3,
        gt_lat_ratio: float = 0.2,
        gt_lon_ratio: float = 0.3,
        # optimizer
        lr: float = 1e-4,
        vision_lr: Optional[float] = None,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        pct_start: float = 0.05,
        # loss weights
        lon_action_weights: Optional[List[float]] = None,
        lat_action_weights: Optional[List[float]] = None,
        new_layer_norm_minmax: bool = False,
        speed_as_input: bool = True,
    ):
        super().__init__()
        if isinstance(vision_model, DictConfig):
            vision_model = hydra.utils.instantiate(vision_model)

        self.save_hyperparameters()

        self.vision_model = vision_model
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        self.decoder_type = decoder_type
        self.use_acm = use_film
        self.predict_phase = predict_phase
        self.predict_route_as_wps = predict_route_as_wps
        self.route_wp_loss_weight = route_wp_loss_weight
        self.route_wp_loss_warmup_epochs = route_wp_loss_warmup_epochs
        self.speed_wps_mode = speed_wps_mode
        self.use_prev_action_id = use_prev_action_id
        self.use_gt_conditioning = use_gt_conditioning
        self.gt_ctx_ratio = gt_ctx_ratio
        self.gt_lat_ratio = gt_lat_ratio
        self.gt_lon_ratio = gt_lon_ratio
        self.lr = lr
        self.vision_lr = vision_lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.pct_start = pct_start

        # ── ablation flags ─────────────────────────────────────────────────────
        self.abl_no_acm              = ablation_no_acm
        self.abl_drop_action_tokens  = ablation_drop_action_tokens
        self.abl_drop_all_tokens     = ablation_drop_all_tokens
        self.abl_single_pass         = ablation_single_pass
        self.abl_lon_from_vpool      = ablation_lon_from_vpool
        self.abl_no_gt_cond          = ablation_no_gt_conditioning
        self.abl_no_grad_mask        = ablation_no_grad_mask
        # Validate: invalid flag combinations
        assert not (ablation_no_acm and ablation_drop_action_tokens), \
            "A1 (ablation_no_acm) and A2 (ablation_drop_action_tokens) are mutually exclusive"
        assert not (ablation_no_acm and ablation_drop_all_tokens), \
            "A1 (ablation_no_acm) and A3 (ablation_drop_all_tokens) are mutually exclusive"
        assert not (ablation_drop_action_tokens and ablation_drop_all_tokens), \
            "A2 (ablation_drop_action_tokens) and A3 (ablation_drop_all_tokens) are mutually exclusive"
        assert not (ablation_single_pass and ablation_lon_from_vpool), \
            "B1 (ablation_single_pass) already uses v_pool for lon; don't combine with B2"
        assert not (ablation_no_acm and not use_film), \
            "ablation_no_acm requires use_film=True (heads must exist)"

        # --- vision projection ---
        vis_dim = self.vision_model.token_size
        self.language_projection = nn.Identity()
        if vis_dim != hidden_dim:
            self.language_projection = nn.Linear(vis_dim, hidden_dim, bias=False)

        # --- pooler ---
        self.pooler = build_pooler(pool_type, hidden_dim)

        # --- Context / Lat / Lon heads + ACM ---
        self.ctx_head = None
        self.lat_action_head = None
        self.lon_action_head = None
        self.acm = None
        self.speed_encoder = None
        self.phase_head = None

        if self.use_acm:
            self.ctx_head = ContextHead(hidden_dim)
            self.lat_action_head = LateralActionHead(hidden_dim, cond_emb_dim)
            ctx_emb_dim = 4 * cond_emb_dim
            # B1/B2: lon head takes v_pool (hidden_dim) instead of e_ctx (ctx_emb_dim)
            lon_input_dim = hidden_dim if (ablation_single_pass or ablation_lon_from_vpool) else ctx_emb_dim
            self.lon_action_head = LongitudinalActionHead(lon_input_dim, cond_emb_dim)

            if predict_phase:
                self.phase_head = nn.Sequential(
                    nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 3)
                )

            self.acm = ACMConditioner(
                out_dim=hidden_dim,
                cond_emb_dim=cond_emb_dim,
                film_hidden=film_hidden,
                use_prev_action_id=use_prev_action_id,
            )
            min_max = (0.0, 110.0 / 3.6) if new_layer_norm_minmax else (0.0, 64.0 / 3.6)
            self.speed_encoder = VectorInputAdaptor(
                input_size=1, token_size=cond_emb_dim, hidden_size=64,
                norm_layer=NormZeroOne(min_max=min_max),
            )

        # --- Speed Waypoint head ---
        num_wps = 10
        wp_out_dim = 2 if speed_wps_mode == "2d" else 1
        if decoder_type == "mlp":
            self.wp_head = WaypointMLPHead(hidden_dim, num_wps, wp_out_dim)
        else:
            self.wp_head = WaypointDecoderHead(
                hidden_dim, num_wps, wp_out_dim,
                num_layers=decoder_layers, nhead=decoder_nhead,
            )

        # --- Route Waypoint head (optional, always 2D) ---
        self.route_wp_head = None
        if predict_route_as_wps:
            if decoder_type == "mlp":
                self.route_wp_head = WaypointMLPHead(hidden_dim, num_route_wps, 2)
            else:
                self.route_wp_head = WaypointDecoderHead(
                    hidden_dim, num_route_wps, 2,
                    num_layers=decoder_layers, nhead=decoder_nhead,
                )

        # --- CE loss weights ---
        if lon_action_weights is not None:
            self.register_buffer("sa_weights", torch.tensor(lon_action_weights, dtype=torch.float))
        else:
            self.sa_weights = None
        if lat_action_weights is not None:
            self.register_buffer("rs_weights", torch.tensor(lat_action_weights, dtype=torch.float))
        else:
            self.rs_weights = None

    # ──────── checkpoint compat ────────

    def on_load_checkpoint(self, checkpoint):
        sd = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = _remap_state_dict(sd)

    # ──────── helpers ────────

    def _pool(self, V: Tensor) -> Tensor:
        if self.pool_type == "cls":
            return V[:, 0]
        elif self.pool_type == "attn":
            return self.pooler(V)
        else:  # mean
            return V.mean(dim=1)

    def _teacher_progress(self) -> float:
        try:
            trainer = self.trainer
        except RuntimeError:
            return 1.0
        if trainer is None or trainer.max_epochs is None:
            return 1.0
        return trainer.current_epoch / max(trainer.max_epochs, 1)

    def _route_wp_current_weight(self) -> float:
        """Linear warmup: 0 → route_wp_loss_weight over warmup epochs."""
        if self.route_wp_loss_warmup_epochs <= 0:
            return self.route_wp_loss_weight
        try:
            trainer = self.trainer
        except RuntimeError:
            return self.route_wp_loss_weight
        if trainer is None:
            return self.route_wp_loss_weight
        progress = min(trainer.current_epoch / self.route_wp_loss_warmup_epochs, 1.0)
        return self.route_wp_loss_weight * progress

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        """Return loss weights dict for summarise_losses (None = all equal)."""
        if self.route_wp_head is None:
            return None
        return {"route_wp_loss": self._route_wp_current_weight()}

    # ──────── forward ────────

    def forward(self, driving_input: RoadGlyphInput):
        """Inference: returns (speed_wps, route_wps) tuple."""
        V = self._encode_vision(driving_input)
        speed_wps, route_wps = self._forward_from_features(V, driving_input)
        return speed_wps, route_wps

    def _forward_from_features(
        self,
        V: Tensor,
        driving_input: RoadGlyphInput,
        force_lat: Optional[int] = None,
        force_lon: Optional[int] = None,
    ):
        """Core forward pass given pre-computed vision features V [B, N, D].

        force_lat / force_lon: override predicted action class with a fixed index
        (used in Δ_wp intervention metric computation).
        """
        if not self.use_acm:
            speed_wps = self.wp_head(V)
            route_wps = self.route_wp_head(V) if self.route_wp_head is not None else None
            return speed_wps, route_wps

        v_pool = self._pool(V)
        ctx_logits = self.ctx_head(v_pool)
        e_hlc_cond = self.acm.hlc_emb(driving_input.hlc)
        lat_logits = self.lat_action_head(v_pool, e_hlc_cond)

        # Token intervention: force lat action (Δ_wp metric)
        if force_lat is not None:
            lat_logits = torch.full_like(lat_logits, -1e4)
            lat_logits[:, force_lat] = 1e4

        e_spd = self.speed_encoder(driving_input.vehicle_speed).squeeze(1)
        B = v_pool.size(0)

        # B1: single-pass — lon head uses v_pool directly, ACM runs once
        if self.abl_single_pass:
            lon_logits = self.lon_action_head(v_pool, e_spd)
        else:
            # Pass-1: get e_ctx for lon head (dummy lon_logits)
            e_ctx_for_lon, _, _ = self.acm(
                ctx_logits, lat_logits,
                torch.zeros(B, 8, device=v_pool.device),
                driving_input.hlc, driving_input.vehicle_speed,
            )
            # B2: lon head uses v_pool; default: e_ctx
            if self.abl_lon_from_vpool:
                lon_logits = self.lon_action_head(v_pool, e_spd)
            else:
                lon_logits = self.lon_action_head(e_ctx_for_lon, e_spd)

        # Token intervention: force lon action (Δ_wp metric)
        if force_lon is not None:
            lon_logits = torch.full_like(lon_logits, -1e4)
            lon_logits[:, force_lon] = 1e4

        # Pass-2 (or single pass for B1): full ACM → γ, β
        _, gamma, beta = self.acm(
            ctx_logits, lat_logits, lon_logits,
            driving_input.hlc, driving_input.vehicle_speed,
            prev_lat_action_id=driving_input.prev_lat_action_id if self.use_prev_action_id else None,
            prev_lon_action_id=driving_input.prev_lon_action_id if self.use_prev_action_id else None,
            prev_phase=driving_input.prev_phase if self.use_prev_action_id else None,
            drop_action_tokens=self.abl_drop_action_tokens,  # A2
            drop_all_tokens=self.abl_drop_all_tokens,        # A3
        )

        # A1: identity FiLM — γ=1, β=0 (keep heads/losses, remove modulation)
        if self.abl_no_acm:
            gamma = torch.ones_like(gamma)
            beta  = torch.zeros_like(beta)

        V_mod = (gamma.unsqueeze(1) * V + beta.unsqueeze(1)).to(V.dtype)
        speed_wps = self.wp_head(V_mod)
        route_wps = self.route_wp_head(V_mod) if self.route_wp_head is not None else None
        return speed_wps, route_wps

    def forward_forced(
        self,
        driving_input: RoadGlyphInput,
        V: Optional[Tensor] = None,
        force_lat: Optional[int] = None,
        force_lon: Optional[int] = None,
    ):
        """Inference with forced action tokens (for Δ_wp intervention metric).

        If V is provided (pre-computed vision features), skips vision encoding.
        """
        if V is None:
            V = self._encode_vision(driving_input)
        return self._forward_from_features(V, driving_input,
                                           force_lat=force_lat, force_lon=force_lon)

    def _encode_vision(self, driving_input: RoadGlyphInput) -> Tensor:
        img = driving_input.camera_images
        embeds, _ = self.vision_model.forward(img, image_sizes=driving_input.image_sizes)
        return self.language_projection(embeds)

    # ──────── training ────────

    def forward_loss(self, example: RoadGlyphExample) -> TrainingOutput:
        di = example.driving_input
        cl = example.ctx_label
        tl = example.action_token_label
        dl = example.driving_label

        V = self._encode_vision(di)
        B = V.size(0)
        ones = torch.ones(B, device=V.device, dtype=torch.long)
        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}

        # --- Baseline path: no ACM ---
        if not self.use_acm:
            speed_wps = self.wp_head(V)
            if self.speed_wps_mode == "2d":
                label_wps = dl.waypoints[:, :speed_wps.size(1)]
            else:
                label_wps = dl.waypoints_1d[:, :speed_wps.size(1)]
            wp_loss = F.mse_loss(speed_wps, label_wps, reduction="none").sum(-1).mean(-1)
            loss_dict["speed_wps_loss"] = (wp_loss, ones)
            if self.route_wp_head is not None:
                route_wps = self.route_wp_head(V)
                route_label = dl.route_adjusted[:, :route_wps.size(1)]
                route_wp_loss = F.mse_loss(route_wps, route_label, reduction="none").sum(-1).mean(-1)
                loss_dict["route_wp_loss"] = (route_wp_loss, ones)
            return summarise_losses(loss_dict, self._loss_weights())

        # --- Full path: Context/Lat/Lon heads + ACM ---
        v_pool = self._pool(V)

        # Context
        ctx_logits = self.ctx_head(v_pool)

        # Lateral action
        e_hlc_cond = self.acm.hlc_emb(di.hlc)
        lat_logits = self.lat_action_head(v_pool, e_hlc_cond)

        # GT conditioning schedule
        # C1: always predict, never teacher-force
        if self.abl_no_gt_cond:
            use_gt_ctx = use_gt_lat = use_gt_lon = False
        else:
            progress = self._teacher_progress()
            use_gt_ctx = self.use_gt_conditioning and (progress < self.gt_ctx_ratio)
            use_gt_lat = self.use_gt_conditioning and (progress < self.gt_lat_ratio)
            use_gt_lon = self.use_gt_conditioning and (progress < self.gt_lon_ratio)

        e_spd = self.speed_encoder(di.vehicle_speed).squeeze(1)

        # Longitudinal action
        # B1: single-pass — lon head takes v_pool directly, no Pass-1
        if self.abl_single_pass:
            lon_logits = self.lon_action_head(v_pool, e_spd)
        else:
            # Pass-1: get e_ctx for lon head (dummy lon_logits=0)
            e_ctx_for_lon, _, _ = self.acm(
                ctx_logits, lat_logits, torch.zeros(B, 8, device=V.device),
                di.hlc, di.vehicle_speed,
                use_gt_ctx=use_gt_ctx, gt_ctx=cl, gt_action=tl,
            )
            # B2: lon head uses v_pool; default: e_ctx from Pass-1
            if self.abl_lon_from_vpool:
                lon_logits = self.lon_action_head(v_pool, e_spd)
            else:
                lon_logits = self.lon_action_head(e_ctx_for_lon, e_spd)

        # Pass-2 (or single ACM call for B1): full γ, β
        _, gamma, beta = self.acm(
            ctx_logits, lat_logits, lon_logits,
            di.hlc, di.vehicle_speed,
            use_gt_ctx=use_gt_ctx, use_gt_lat=use_gt_lat, use_gt_lon=use_gt_lon,
            gt_ctx=cl, gt_action=tl,
            prev_lat_action_id=di.prev_lat_action_id if self.use_prev_action_id else None,
            prev_lon_action_id=di.prev_lon_action_id if self.use_prev_action_id else None,
            prev_phase=di.prev_phase if self.use_prev_action_id else None,
            drop_action_tokens=self.abl_drop_action_tokens,  # A2
            drop_all_tokens=self.abl_drop_all_tokens,        # A3
        )

        # A1: identity FiLM — γ=1, β=0 (keep all heads/losses, remove modulation)
        if self.abl_no_acm:
            gamma = torch.ones_like(gamma)
            beta  = torch.zeros_like(beta)

        # Gradient masking: detach ACM for samples without context labels.
        # C2 (ablation_no_grad_mask): skip this — allow gradients to flow freely.
        if not self.abl_no_grad_mask:
            has_label = (cl.ctx_speed_kind != IGNORE_INDEX)  # [B]
            if not has_label.all():
                mask = has_label.float().unsqueeze(1)           # [B, 1]
                gamma = gamma * mask + gamma.detach() * (1 - mask)
                beta  = beta  * mask + beta.detach()  * (1 - mask)

        V_mod = (gamma.unsqueeze(1) * V + beta.unsqueeze(1)).to(V.dtype)
        speed_wps = self.wp_head(V_mod)

        # ======== Losses ========
        # Waypoint loss
        if self.speed_wps_mode == "2d":
            label_wps = dl.waypoints[:, :speed_wps.size(1)]
        else:
            label_wps = dl.waypoints_1d[:, :speed_wps.size(1)]
        wp_loss = F.mse_loss(speed_wps, label_wps, reduction="none").sum(-1).mean(-1)
        loss_dict["speed_wps_loss"] = (wp_loss, ones)

        # Context losses
        ce_kw = dict(reduction="none", ignore_index=IGNORE_INDEX)
        loss_dict["ctx_speed_kind_loss"] = (
            F.cross_entropy(ctx_logits["speed_kind"], cl.ctx_speed_kind, **ce_kw), ones)
        loss_dict["ctx_speed_sub_loss"] = (
            F.cross_entropy(ctx_logits["speed_sub"], cl.ctx_speed_sub, **ce_kw), ones)
        loss_dict["ctx_route_kind_loss"] = (
            F.cross_entropy(ctx_logits["route_kind"], cl.ctx_route_kind, **ce_kw), ones)
        loss_dict["ctx_route_sub_loss"] = (
            F.cross_entropy(ctx_logits["route_sub"], cl.ctx_route_sub, **ce_kw), ones)

        # Lateral action loss
        lat_weight = self.rs_weights if self.rs_weights is not None else None
        loss_dict["lat_action_loss"] = (
            F.cross_entropy(lat_logits, tl.lat_action_id, weight=lat_weight, **ce_kw), ones)

        # Longitudinal action loss
        lon_weight = self.sa_weights if self.sa_weights is not None else None
        loss_dict["lon_action_loss"] = (
            F.cross_entropy(lon_logits, tl.lon_action_id, weight=lon_weight, **ce_kw), ones)

        # Phase loss (optional)
        if self.predict_phase and self.phase_head is not None:
            phase_logits = self.phase_head(v_pool)
            loss_dict["phase_loss"] = (
                F.cross_entropy(phase_logits, tl.phase, **ce_kw), ones)

        # Route waypoint loss
        if self.route_wp_head is not None:
            route_wps = self.route_wp_head(V_mod)
            route_label = dl.route_adjusted[:, :route_wps.size(1)]
            route_wp_loss = F.mse_loss(route_wps, route_label, reduction="none").sum(-1).mean(-1)
            loss_dict["route_wp_loss"] = (route_wp_loss, ones)

        return summarise_losses(loss_dict, self._loss_weights())

    def training_step(self, batch: RoadGlyphExample, _batch_idx: int = 0):
        output = self.forward_loss(batch)
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in output.loss_averages.items():
            self.log(f"train_losses/{k}", v.detach(), sync_dist=True)
        return {"loss": output.loss}

    def validation_step(self, batch: RoadGlyphExample, _batch_idx: int = 0):
        output = self.forward_loss(batch)
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for k, v in output.loss_averages.items():
            self.log(f"val_losses/{k}", v.detach(), sync_dist=True)
        return {"loss": output.loss}

    def configure_optimizers(self):
        from deepspeed.ops.adam import FusedAdam

        param_groups = [
            ParamGroup(r"^(?!vision_model\.).*", self.lr, self.weight_decay),
            ParamGroup(r"^vision_model\..*", self.vision_lr or self.lr, self.weight_decay),
        ]
        optimizer_class = (
            FusedAdam if isinstance(self.trainer.strategy, pl.strategies.DeepSpeedStrategy)
            else torch.optim.AdamW
        )
        optimizer = optimizer_class(
            configure_params_groups(self, param_groups, verbose=False), betas=self.betas
        )
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        max_steps = (
            self.trainer.estimated_stepping_batches
            if self.trainer.max_steps == -1
            else self.trainer.max_steps
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lrs, total_steps=max_steps, pct_start=self.pct_start
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"},
        }


