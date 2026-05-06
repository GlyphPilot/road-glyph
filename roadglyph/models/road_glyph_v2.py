"""
RoadGlyphModelV2: Three changes vs RoadGlyphModel.

1. No route_wp_loss warmup: route_wp_loss_weight used from epoch 0.
2. route_wps predicted from raw V (not V_mod). speed_wps still uses V_mod.
3. num_route_wps default = 20 (overridable in config/experiment yaml).

Fully compatible with existing RoadGlyphModel checkpoints (identical weight structure).
"""
import torch
from torch import Tensor

from roadglyph.models.road_glyph import RoadGlyphModel
from roadglyph.utils.custom_types import RoadGlyphExample, RoadGlyphInput
from simlingo_base_training.utils.custom_types import TrainingOutput
from simlingo_base_training.models.utils import summarise_losses
import torch.nn.functional as F
from roadglyph.utils.custom_types import IGNORE_INDEX
from typing import Dict, Optional, Tuple


class RoadGlyphModelV2(RoadGlyphModel):
    """
    Changes vs RoadGlyphModel:
      - No route_wp warmup: route_wp_loss_weight used from step 0.
      - route_wp_head predicts from raw V (not V_mod).
      - num_route_wps default = 20 (set in config/experiment yaml).
    """

    # ──────── 1. no warmup ────────

    def _route_wp_current_weight(self) -> float:
        """No warmup: use route_wp_loss_weight from epoch 0."""
        return self.route_wp_loss_weight

    # ──────── 2. forward: route_wps from raw V ────────

    def forward(self, driving_input: RoadGlyphInput):
        """Inference: speed_wps from V_mod, route_wps from raw V."""
        V = self._encode_vision(driving_input)

        if not self.use_acm:
            speed_wps = self.wp_head(V)
            route_wps = self.route_wp_head(V) if self.route_wp_head is not None else None
            return speed_wps, route_wps

        v_pool = self._pool(V)
        ctx_logits = self.ctx_head(v_pool)
        e_hlc_cond = self.acm.hlc_emb(driving_input.hlc)
        lat_logits = self.lat_action_head(v_pool, e_hlc_cond)

        e_ctx, _, _ = self.acm(
            ctx_logits, lat_logits,
            torch.zeros(v_pool.size(0), 8, device=v_pool.device),
            driving_input.hlc, driving_input.vehicle_speed,
        )
        e_spd = self.speed_encoder(driving_input.vehicle_speed).squeeze(1)
        lon_logits = self.lon_action_head(e_ctx, e_spd)

        _, gamma, beta = self.acm(
            ctx_logits, lat_logits, lon_logits,
            driving_input.hlc, driving_input.vehicle_speed,
            prev_lat_action_id=driving_input.prev_lat_action_id if self.use_prev_action_id else None,
            prev_lon_action_id=driving_input.prev_lon_action_id if self.use_prev_action_id else None,
            prev_phase=driving_input.prev_phase if self.use_prev_action_id else None,
        )
        V_mod = (gamma.unsqueeze(1) * V + beta.unsqueeze(1)).to(V.dtype)

        speed_wps = self.wp_head(V_mod)
        route_wps = self.route_wp_head(V) if self.route_wp_head is not None else None  # raw V
        return speed_wps, route_wps

    # ──────── 3. forward_loss: route_wps from raw V ────────

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

        ctx_logits = self.ctx_head(v_pool)

        e_hlc_cond = self.acm.hlc_emb(di.hlc)
        lat_logits = self.lat_action_head(v_pool, e_hlc_cond)

        progress = self._teacher_progress()
        use_gt_ctx = self.use_gt_conditioning and (progress < self.gt_ctx_ratio)
        use_gt_lat = self.use_gt_conditioning and (progress < self.gt_lat_ratio)
        use_gt_lon = self.use_gt_conditioning and (progress < self.gt_lon_ratio)

        e_ctx_for_lon, _, _ = self.acm(
            ctx_logits, lat_logits, torch.zeros(B, 8, device=V.device),
            di.hlc, di.vehicle_speed,
            use_gt_ctx=use_gt_ctx, gt_ctx=cl, gt_action=tl,
        )
        e_spd = self.speed_encoder(di.vehicle_speed).squeeze(1)
        lon_logits = self.lon_action_head(e_ctx_for_lon, e_spd)

        _, gamma, beta = self.acm(
            ctx_logits, lat_logits, lon_logits,
            di.hlc, di.vehicle_speed,
            use_gt_ctx=use_gt_ctx, use_gt_lat=use_gt_lat, use_gt_lon=use_gt_lon,
            gt_ctx=cl, gt_action=tl,
            prev_lat_action_id=di.prev_lat_action_id if self.use_prev_action_id else None,
            prev_lon_action_id=di.prev_lon_action_id if self.use_prev_action_id else None,
            prev_phase=di.prev_phase if self.use_prev_action_id else None,
        )

        has_label = (cl.ctx_speed_kind != IGNORE_INDEX)
        if not has_label.all():
            mask = has_label.float().unsqueeze(1)
            gamma = gamma * mask + gamma.detach() * (1 - mask)
            beta  = beta  * mask + beta.detach()  * (1 - mask)

        V_mod = (gamma.unsqueeze(1) * V + beta.unsqueeze(1)).to(V.dtype)
        speed_wps = self.wp_head(V_mod)

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

        lat_weight = self.rs_weights if self.rs_weights is not None else None
        loss_dict["lat_action_loss"] = (
            F.cross_entropy(lat_logits, tl.lat_action_id, weight=lat_weight, **ce_kw), ones)

        lon_weight = self.sa_weights if self.sa_weights is not None else None
        loss_dict["lon_action_loss"] = (
            F.cross_entropy(lon_logits, tl.lon_action_id, weight=lon_weight, **ce_kw), ones)

        if self.predict_phase and self.phase_head is not None:
            phase_logits = self.phase_head(v_pool)
            loss_dict["phase_loss"] = (
                F.cross_entropy(phase_logits, tl.phase, **ce_kw), ones)

        # Route waypoint loss — raw V (V2 change: predicts from V, not V_mod)
        if self.route_wp_head is not None:
            route_wps = self.route_wp_head(V)
            route_label = dl.route_adjusted[:, :route_wps.size(1)]
            route_wp_loss = F.mse_loss(route_wps, route_label, reduction="none").sum(-1).mean(-1)
            loss_dict["route_wp_loss"] = (route_wp_loss, ones)

        return summarise_losses(loss_dict, self._loss_weights())
