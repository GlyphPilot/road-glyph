"""
RoadGlyphDataModuleV3: extends V1 datamodule with prev_route_wps collation.
"""
import numpy as np
import torch

from roadglyph.dataloader.datamodule import RoadGlyphDataModule, encode_uint8
from roadglyph.dataloader.dataset_v3 import RoadGlyphCARLADataV3
from roadglyph.utils.custom_types import ActionTokenLabel, ContextFactorLabel
from roadglyph.utils.custom_types_v3 import RoadGlyphExampleV3, RoadGlyphInputV3
from simlingo_base_training.utils.custom_types import DrivingLabel
from simlingo_base_training.utils.projection import get_camera_extrinsics, get_camera_intrinsics

import torch.nn.functional as F


class RoadGlyphDataModuleV3(RoadGlyphDataModule):
    """V1 DataModule with prev_route_wps added to RoadGlyphInputV3."""

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
                datasets[bucket] = RoadGlyphCARLADataV3(
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

            self.val_dataset = RoadGlyphCARLADataV3(
                **self._dataset_cfg(
                    split="val",
                    bucket_name="all",
                ),
            )
        else:
            self.train_dataset = None
            self.val_dataset = None

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
            from transformers import LlavaNextProcessor
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

        return RoadGlyphExampleV3(
            driving_input=RoadGlyphInputV3(
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
                prev_route_wps=torch.tensor(
                    np.asarray([d["prev_route_wps"] for d in data]), dtype=torch.float32
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
