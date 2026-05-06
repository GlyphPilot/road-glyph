"""
RoadGlyphModelV3: V1 (WP64, ACM) + Smoothness Loss only.

Changes vs V1:
  - 2nd-order finite-difference penalty on speed_wps & route_wps (jerk suppression)
  - No prev_route_wps / consistency loss (caused degradation in initial V3 experiments)
  - No structural, data, or collate changes (identical to V1)
"""
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from roadglyph.models.road_glyph import RoadGlyphModel
from roadglyph.utils.custom_types import RoadGlyphExample
from simlingo_base_training.utils.custom_types import TrainingOutput
from simlingo_base_training.models.utils import summarise_losses


def _smoothness_loss(wps: Tensor) -> Tensor:
    """2nd-order finite difference (jerk proxy) on waypoint sequence.
    wps: [B, N, D] → returns per-sample scalar [B].
    """
    d1 = wps[:, 1:] - wps[:, :-1]   # velocity  [B, N-1, D]
    d2 = d1[:, 1:] - d1[:, :-1]     # accel diff [B, N-2, D]
    return (d2 ** 2).sum(-1).mean(-1)  # [B]


class RoadGlyphModelV3(RoadGlyphModel):
    """V1 + smoothness loss on speed_wps & route_wps. No structural changes."""

    def __init__(
        self,
        vision_model,
        smoothness_loss_weight: float = 0.1,
        route_smoothness_loss_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(vision_model=vision_model, **kwargs)
        self.smoothness_loss_weight = smoothness_loss_weight
        self.route_smoothness_loss_weight = route_smoothness_loss_weight

    def _loss_weights(self):
        w = {}
        if self.route_wp_head is not None:
            w["route_wp_loss"] = self._route_wp_current_weight()
        if self.smoothness_loss_weight > 0:
            w["speed_smooth_loss"] = self.smoothness_loss_weight
        if self.route_wp_head is not None and self.route_smoothness_loss_weight > 0:
            w["route_smooth_loss"] = self.route_smoothness_loss_weight
        return w if w else None

    def forward_loss(self, example: RoadGlyphExample) -> TrainingOutput:
        di = example.driving_input
        cl = example.ctx_label
        tl = example.action_token_label
        dl = example.driving_label

        V = self._encode_vision(di)
        B = V.size(0)
        ones = torch.ones(B, device=V.device, dtype=torch.long)
        loss_dict: Dict = {}

        # ---------- baseline (no ACM) ----------
        if not self.use_acm:
            speed_wps = self.wp_head(V)
            label_wps = dl.waypoints[:, :speed_wps.size(1)] if self.speed_wps_mode == "2d" \
                        else dl.waypoints_1d[:, :speed_wps.size(1)]
            loss_dict["speed_wps_loss"] = (F.mse_loss(speed_wps, label_wps, reduction="none").sum(-1).mean(-1), ones)
            loss_dict["speed_smooth_loss"] = (_smoothness_loss(speed_wps), ones)
            if self.route_wp_head is not None:
                route_wps = self.route_wp_head(V)
                route_label = dl.route_adjusted[:, :route_wps.size(1)]
                loss_dict["route_wp_loss"] = (F.mse_loss(route_wps, route_label, reduction="none").sum(-1).mean(-1), ones)
                loss_dict["route_smooth_loss"] = (_smoothness_loss(route_wps), ones)
            return summarise_losses(loss_dict, self._loss_weights())

        # ---------- full ACM path ----------
        v_pool = self._pool(V)
        ctx_logits = self.ctx_head(v_pool)
        e_hlc_cond = self.acm.hlc_emb(di.hlc)
        lat_logits = self.lat_action_head(v_pool, e_hlc_cond)

        if self.abl_no_gt_cond:
            use_gt_ctx = use_gt_lat = use_gt_lon = False
        else:
            progress = self._teacher_progress()
            use_gt_ctx = self.use_gt_conditioning and (progress < self.gt_ctx_ratio)
            use_gt_lat = self.use_gt_conditioning and (progress < self.gt_lat_ratio)
            use_gt_lon = self.use_gt_conditioning and (progress < self.gt_lon_ratio)

        e_spd = self.speed_encoder(di.vehicle_speed).squeeze(1)

        if self.abl_single_pass:
            lon_logits = self.lon_action_head(v_pool, e_spd)
        else:
            e_ctx_for_lon, _, _ = self.acm(
                ctx_logits, lat_logits, torch.zeros(B, 8, device=V.device),
                di.hlc, di.vehicle_speed,
                use_gt_ctx=use_gt_ctx, gt_ctx=cl, gt_action=tl,
            )
            lon_logits = self.lon_action_head(v_pool, e_spd) if self.abl_lon_from_vpool \
                         else self.lon_action_head(e_ctx_for_lon, e_spd)

        _, gamma, beta = self.acm(
            ctx_logits, lat_logits, lon_logits,
            di.hlc, di.vehicle_speed,
            use_gt_ctx=use_gt_ctx, use_gt_lat=use_gt_lat, use_gt_lon=use_gt_lon,
            gt_ctx=cl, gt_action=tl,
            prev_lat_action_id=di.prev_lat_action_id if self.use_prev_action_id else None,
            prev_lon_action_id=di.prev_lon_action_id if self.use_prev_action_id else None,
            prev_phase=di.prev_phase if self.use_prev_action_id else None,
            drop_action_tokens=self.abl_drop_action_tokens,
            drop_all_tokens=self.abl_drop_all_tokens,
        )

        if self.abl_no_acm:
            gamma = torch.ones_like(gamma)
            beta  = torch.zeros_like(beta)

        if not self.abl_no_grad_mask:
            from roadglyph.utils.custom_types import IGNORE_INDEX
            has_label = (cl.ctx_speed_kind != IGNORE_INDEX)
            if not has_label.all():
                mask = has_label.float().unsqueeze(1)
                gamma = gamma * mask + gamma.detach() * (1 - mask)
                beta  = beta  * mask + beta.detach()  * (1 - mask)

        V_mod = (gamma.unsqueeze(1) * V + beta.unsqueeze(1)).to(V.dtype)
        speed_wps = self.wp_head(V_mod)

        # ── losses ──
        label_wps = dl.waypoints[:, :speed_wps.size(1)] if self.speed_wps_mode == "2d" \
                    else dl.waypoints_1d[:, :speed_wps.size(1)]
        loss_dict["speed_wps_loss"] = (F.mse_loss(speed_wps, label_wps, reduction="none").sum(-1).mean(-1), ones)
        loss_dict["speed_smooth_loss"] = (_smoothness_loss(speed_wps), ones)

        ce_kw = dict(reduction="none", ignore_index=-100)
        loss_dict["ctx_speed_kind_loss"] = (F.cross_entropy(ctx_logits["speed_kind"], cl.ctx_speed_kind, **ce_kw), ones)
        loss_dict["ctx_speed_sub_loss"]  = (F.cross_entropy(ctx_logits["speed_sub"],  cl.ctx_speed_sub,  **ce_kw), ones)
        loss_dict["ctx_route_kind_loss"] = (F.cross_entropy(ctx_logits["route_kind"], cl.ctx_route_kind, **ce_kw), ones)
        loss_dict["ctx_route_sub_loss"]  = (F.cross_entropy(ctx_logits["route_sub"],  cl.ctx_route_sub,  **ce_kw), ones)
        loss_dict["lat_action_loss"] = (F.cross_entropy(lat_logits, tl.lat_action_id, weight=self.rs_weights, **ce_kw), ones)
        loss_dict["lon_action_loss"] = (F.cross_entropy(lon_logits, tl.lon_action_id, weight=self.sa_weights, **ce_kw), ones)

        if self.predict_phase and self.phase_head is not None:
            loss_dict["phase_loss"] = (F.cross_entropy(self.phase_head(v_pool), tl.phase, **ce_kw), ones)

        if self.route_wp_head is not None:
            route_wps = self.route_wp_head(V_mod)
            route_label = dl.route_adjusted[:, :route_wps.size(1)]
            loss_dict["route_wp_loss"]    = (F.mse_loss(route_wps, route_label, reduction="none").sum(-1).mean(-1), ones)
            loss_dict["route_smooth_loss"] = (_smoothness_loss(route_wps), ones)

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
