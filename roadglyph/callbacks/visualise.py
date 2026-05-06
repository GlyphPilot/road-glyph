"""
Visualisation callback for RoadGlyphModel.
"""
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from roadglyph.utils.custom_types import IGNORE_INDEX, RoadGlyphExample


_STEPS_TO_FIRST_IDX: Dict[int, int] = {}


def _once_per_step(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step not in _STEPS_TO_FIRST_IDX:
            _STEPS_TO_FIRST_IDX[step] = batch_idx
        if _STEPS_TO_FIRST_IDX[step] == batch_idx:
            return fn(self, trainer, pl_module, outputs, batch, batch_idx)
        return None

    return wrapper


def _fig_to_np(fig):
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


class RoadGlyphVisualiseCallback(Callback):
    def __init__(self, interval: int = 1000):
        super().__init__()
        self.interval = interval

    @_once_per_step
    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: RoadGlyphExample,
        batch_idx: int,
    ):
        if trainer.global_step % self.interval != 0:
            return

        with torch.cuda.amp.autocast(enabled=True):
            result = pl_module.forward(batch.driving_input)

        if isinstance(result, tuple):
            speed_wps, route_wps = result
        else:
            speed_wps, route_wps = result, None

        try:
            self._visualise_waypoints(batch, speed_wps, trainer, pl_module)
            if route_wps is not None:
                self._visualise_route_wps(batch, route_wps, trainer, pl_module)
        except Exception as e:
            print(f"RoadGlyphVisualiseCallback error: {e}")

    @rank_zero_only
    def _visualise_waypoints(
        self,
        batch: RoadGlyphExample,
        pred_wps: torch.Tensor,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        if not pl_module.logger:
            return

        gt_wps = batch.driving_label.waypoints[:, :pred_wps.size(1), :].cpu().numpy()
        pred = pred_wps.cpu().numpy()
        b = min(gt_wps.shape[0], 16)
        rows = int(np.ceil(b / 4))
        cols = min(b, 4)

        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=0.8)
        for i in range(b):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.scatter(-pred[i, :, 1], pred[i, :, 0], marker="o", c="b")
            ax.plot(-pred[i, :, 1], pred[i, :, 0], c="b")
            ax.scatter(-gt_wps[i, :, 1], gt_wps[i, :, 0], marker="x", c="g")
            ax.plot(-gt_wps[i, :, 1], gt_wps[i, :, 0], c="g")
            ax.set_title(f"wps {i}")
            ax.grid()
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1.5)

        vis = _fig_to_np(fig)
        pl_module.logger.log_image(
            "visualise/waypoints", images=[Image.fromarray(vis)], step=trainer.global_step
        )
        plt.close("all")

    @rank_zero_only
    def _visualise_route_wps(
        self,
        batch: RoadGlyphExample,
        pred_route: torch.Tensor,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        if not pl_module.logger:
            return

        gt_route = batch.driving_label.route_adjusted[:, :pred_route.size(1), :].cpu().numpy()
        pred = pred_route.cpu().numpy()
        b = min(gt_route.shape[0], 16)
        rows = int(np.ceil(b / 4))
        cols = min(b, 4)

        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=0.8)
        for i in range(b):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.scatter(-pred[i, :, 1], pred[i, :, 0], marker="o", c="r")
            ax.plot(-pred[i, :, 1], pred[i, :, 0], c="r")
            ax.scatter(-gt_route[i, :, 1], gt_route[i, :, 0], marker="x", c="g")
            ax.plot(-gt_route[i, :, 1], gt_route[i, :, 0], c="g")
            ax.set_title(f"route {i}")
            ax.grid()
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1.5)

        vis = _fig_to_np(fig)
        pl_module.logger.log_image(
            "visualise/route_wps", images=[Image.fromarray(vis)], step=trainer.global_step
        )
        plt.close("all")

