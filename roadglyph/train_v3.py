import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)

if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Register V3 configs into Hydra ConfigStore before @hydra.main is called
import roadglyph.config_v3  # noqa: F401

import hydra
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from roadglyph.callbacks.visualise import RoadGlyphVisualiseCallback
from roadglyph.config import RoadGlyphTrainConfig
from simlingo_base_training.utils.logging_project import setup_logging


@hydra.main(config_path="config", config_name="config_v3", version_base="1.1")
def main(cfg: RoadGlyphTrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"

    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

    data_module = hydra.utils.instantiate(
        cfg.data_module,
        encoder_variant=cfg.model.vision_model.variant,
        predict=False,
    )
    model = hydra.utils.instantiate(
        cfg.model,
        route_as=cfg.data_module.route_as,
        vision_model={"use_global_img": cfg.data_module.use_global_img},
    )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    setup_logging(cfg)

    resume_path = cfg.resume_path
    resume_wandb = False
    if resume_path is not None:
        if not os.path.exists(resume_path):
            resume_wandb = True
        elif cfg.resume:
            resume_wandb = True
        if not cfg.resume:
            resume_path = None

    loggers = []
    if not cfg.debug and cfg.enable_wandb:
        wandblogger = WandbLogger(
            project=cfg.wandb_project,
            id=cfg.wandb_name,
            name=cfg.wandb_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            resume=resume_wandb,
        )
        wandblogger.watch(model)
        loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2,
            loss_scale=cfg.fp16_loss_scale,
            logging_batch_size_per_gpu=cfg.data_module.batch_size,
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath="./checkpoints",
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=cfg.val_every_n_epochs,
    )
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        dirpath="./checkpoints",
        filename="best-{epoch:03d}-{val_loss:.4f}",
    )
    callbacks = [
        checkpoint_callback,
        best_checkpoint_callback,
        ModelSummary(max_depth=3),
        RoadGlyphVisualiseCallback(interval=1000),
    ]
    if not cfg.debug:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    print(f"Number of GPUS: {cfg.gpus}")

    trainer = Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=callbacks,
        devices=cfg.gpus,
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        logger=loggers,
        precision=cfg.precision,
        strategy=strategy,
        sync_batchnorm=True,
        max_epochs=cfg.max_epochs,
        overfit_batches=0,
    )

    trainer.fit(model, data_module, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
