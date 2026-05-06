"""
DataModule for RoadGlyphModel.
"""
import itertools
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import LlavaNextProcessor

from simlingo_base_training.utils.custom_types import DrivingLabel
from simlingo_base_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics

from roadglyph.dataloader.dataset import RoadGlyphCARLAData
from roadglyph.utils.custom_types import (
    RoadGlyphExample,
    RoadGlyphInput,
    ContextFactorLabel,
    ActionTokenLabel,
)


def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, "\0") for s in strings]
    return torch.tensor([bytearray(s, "utf-8") for s in padded_strings], dtype=torch.uint8)


class RoadGlyphDataModule(LightningDataModule):
    def __init__(self, **cfg):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
        self.cfg = cfg

        if "resnet" in self.encoder_variant or "InternViT" in self.encoder_variant:
            self.processor = None
        elif self.encoder_variant is not None:
            self.processor = LlavaNextProcessor.from_pretrained(self.encoder_variant)

    def _dataset_cfg(self, **overrides):
        """Return cfg dict with overrides applied (avoids duplicate-kwarg errors)."""
        out = {k: v for k, v in self.cfg.items() if k not in overrides}
        out.update(overrides)
        return out

    def setup(self, stage=None):
        if not self.predict:
            if self.train_partitions is not None:
                bucket_list = list(self.train_partitions.keys())
                sample_weights = list(self.train_partitions.values())
            else:
                bucket_list = ["all"]
                sample_weights = [1.0]

            datasets = {}
            for bucket in bucket_list:
                datasets[bucket] = RoadGlyphCARLAData(
                    **self._dataset_cfg(
                        split="train",
                        bucket_name=bucket,
                        bucket_proportion=1.0,
                    ),
                )

            self.train_dataset = torch.utils.data.ConcatDataset(
                [datasets[b] for b in bucket_list]
            )
            weights_train = []
            for i, bucket in enumerate(bucket_list):
                weights_train.extend([sample_weights[i]] * len(datasets[bucket]))

            num_samples = len(self.train_dataset)
            self.sampler_train = torch.utils.data.WeightedRandomSampler(
                weights=weights_train, num_samples=num_samples, replacement=True,
            )

            self.val_dataset = RoadGlyphCARLAData(
                **self._dataset_cfg(
                    split="val",
                    bucket_name="all",
                ),
            )
        else:
            self.train_dataset = None
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            sampler=self.sampler_train,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def dl_collate_fn(self, data):
        B = len(data)
        T = data[0]["rgb"].shape[0]
        N = data[0]["rgb"].shape[1]
        C = data[0]["rgb"].shape[2]
        H = data[0]["rgb"].shape[3]
        W = data[0]["rgb"].shape[4]
        F_wps = 11
        dT = 0.2

        image_sizes = None

        if self.encoder == "llavanext":
            images_batch = torch.tensor(np.asarray([d["rgb"] for d in data]))
            images_batch = images_batch.view(B * T * N, C, H, W)
            images_batch = list(images_batch)
            processed = self.processor.image_processor(
                images_batch, return_tensors="pt", image_grid_pinpoints=[[336, 672]]
            )
            images_pixel = processed["pixel_values"]
            image_sizes = processed["image_sizes"]
            if not self.use_global_img:
                images_pixel = images_pixel[:, 1:]
            num_patches = images_pixel.shape[1]
            nH, nW = images_pixel.shape[3], images_pixel.shape[4]
            images_pixel = images_pixel.view(B, T, N, num_patches, C, nH, nW)
        elif self.encoder == "internvit":
            # [B, T, N, C, H, W] uint8 → float16, resize 448, ImageNet normalize
            images_pixel = torch.tensor(
                np.asarray([d["rgb"] for d in data]), dtype=torch.float32
            ) / 255.0
            images_pixel = images_pixel.view(B * T * N, C, H, W)
            images_pixel = F.interpolate(
                images_pixel, size=(448, 448), mode="bilinear", align_corners=False
            )
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images_pixel = (images_pixel - mean) / std
            images_pixel = images_pixel.half().view(B, T, N, C, 448, 448)
        else:
            images_pixel = torch.tensor(np.asarray([d["rgb"] for d in data])).half()

        return RoadGlyphExample(
            driving_input=RoadGlyphInput(
                camera_images=images_pixel,
                image_sizes=image_sizes,
                camera_intrinsics=torch.repeat_interleave(
                    get_camera_intrinsics(W, H, 110).unsqueeze(0), B * N, dim=0
                ).view(B, N, 3, 3).float(),
                camera_extrinsics=torch.repeat_interleave(
                    get_camera_extrinsics().unsqueeze(0), B * N, dim=0
                ).view(B, N, 4, 4).float(),
                vehicle_speed=torch.tensor(
                    [[d["speed"]] for d in data], dtype=torch.float32
                ),
                map_route=torch.tensor(
                    np.asarray([d["map_route"] for d in data]), dtype=torch.float32
                ),
                target_point=torch.tensor(
                    np.asarray([d["target_point"] for d in data]), dtype=torch.float32
                ),
                hlc=torch.tensor([d["hlc"] for d in data], dtype=torch.long),
                prev_lat_action_id=torch.tensor(
                    [d["prev_lat_action_id"] for d in data], dtype=torch.long
                ),
                prev_lon_action_id=torch.tensor(
                    [d["prev_lon_action_id"] for d in data], dtype=torch.long
                ),
                prev_phase=torch.tensor(
                    [d["prev_phase"] for d in data], dtype=torch.long
                ),
            ),
            driving_label=DrivingLabel(
                time_delta_sec=torch.tensor([dT * i for i in range(F_wps)]).repeat(B, 1).float(),
                waypoints=torch.tensor(
                    np.asarray([d["waypoints"] for d in data]), dtype=torch.float32
                ),
                waypoints_1d=torch.tensor(
                    np.asarray([d["waypoints_1d"] for d in data]), dtype=torch.float32
                ),
                route_adjusted=torch.tensor(
                    np.asarray([d["route_adjusted"] for d in data]), dtype=torch.float32
                ),
            ),
            ctx_label=ContextFactorLabel(
                ctx_speed_kind=torch.tensor(
                    [d["ctx_speed_kind"] for d in data], dtype=torch.long
                ),
                ctx_speed_sub=torch.tensor(
                    [d["ctx_speed_sub"] for d in data], dtype=torch.long
                ),
                ctx_route_kind=torch.tensor(
                    [d["ctx_route_kind"] for d in data], dtype=torch.long
                ),
                ctx_route_sub=torch.tensor(
                    [d["ctx_route_sub"] for d in data], dtype=torch.long
                ),
            ),
            action_token_label=ActionTokenLabel(
                lat_action_id=torch.tensor(
                    [d["lat_action_id"] for d in data], dtype=torch.long
                ),
                lon_action_id=torch.tensor(
                    [d["lon_action_id"] for d in data], dtype=torch.long
                ),
                phase=torch.tensor(
                    [d["phase"] for d in data], dtype=torch.long
                ),
            ),
            run_id=encode_uint8([d["measurement_path"] for d in data], 1000),
            timestamp=torch.zeros(B, dtype=torch.int64),
        )
