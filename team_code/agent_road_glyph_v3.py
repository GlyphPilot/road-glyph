"""
TemplateDrive agent for Bench2Drive evaluation.
Based on agent_simlingo_base.py, adapted for InternViT-300M + RoadGlyphModel.
"""

import json
import math
import os
import pathlib
import sys
import time
from collections import deque
from pathlib import Path

import carla
import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F_torch
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from leaderboard.autoagents import autonomous_agent
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

import scenario_logger
import team_code.transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from scipy.signal import savgol_filter as _savgol_filter
from roadglyph.models.road_glyph import _remap_state_dict
from roadglyph.utils.custom_types import RoadGlyphInput
from team_code.config_simlingo_base import GlobalConfig
from team_code.nav_planner import LateralPIDController, RoutePlanner
from team_code.simlingo_utils import (
    get_camera_extrinsics,
    get_camera_intrinsics,
    get_rotation_matrix,
    project_points,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# ImageNet normalization for InternViT
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# CARLA command → HLC mapping (RoadOption enum values)
COMMAND_TO_HLC = {
    1: 1,  # Left
    2: 2,  # Right
    3: 3,  # Straight
    4: 4,  # Follow lane
    5: 5,  # Lane change left
    6: 6,  # Lane change right
}


def get_entry_point():
    return 'RoadGlyphAgent'


DEBUG = True
HD_VIZ = False
USE_UKF = True


class RoadGlyphAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file, route_index=None):
        torch.cuda.empty_cache()
        self.camera_for_viz = None
        self.track = autonomous_agent.Track.SENSORS

        if '+' in path_to_conf_file:
            self.config_path = path_to_conf_file.split('+')[0]
            self.save_path_root = path_to_conf_file.split('+')[1]
        else:
            self.config_path = path_to_conf_file
            self.save_path_root = route_index

        self.step = -1
        self.initialized = False
        self.device = torch.device('cuda')
        self.config = GlobalConfig()

        self.last_command = -1
        self.last_command_tmp = -1

        self.route_path = os.environ.get('ROUTES', '')
        route_type = self.route_path.split('data/benchmarks/')[-1].split('/')[0]
        route_number = str(pathlib.Path(self.route_path).stem)

        self.speed_controller = t_u.PIDController(
            k_p=self.config.speed_kp, k_i=self.config.speed_ki,
            k_d=self.config.speed_kd, n=self.config.speed_n,
        )
        self.turn_controller = LateralPIDController(inference_mode=False)

        self.image_buffer = deque(maxlen=5)

        self.carla_frame_rate = 1.0 / 20.0
        self.data_save_freq = 5
        self.lidar_seq_len = 1
        self.logging_freq = 10
        self.logger_region_of_interest = 30.0
        self.dense_route_planner_min_distance = 1.0
        self.dense_route_planner_max_distance = 50.0
        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        # Load Hydra config from checkpoint directory
        self.config_load_path = Path(self.config_path).parent.parent / '.hydra' / 'config.yaml'
        with open(self.config_load_path, 'r') as file:
            cfg = OmegaConf.load(file)
        self.cfg = cfg
        self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        cfg.model.route_as = "target_point"

        # Backward-compat: remap old _code module paths → road_glyph
        _TARGET_REMAP = {
            "roadglyph.models.roadglyph.RoadGlyphModel":
                "roadglyph.models.roadglyph.RoadGlyphModel",
            "roadglyph.models.encoder.internvit.InternViTEncoderModel":
                "roadglyph.models.encoder.internvit.InternViTEncoderModel",
        }
        _PARAM_REMAP = {
            "use_prev_template_id": "use_prev_action_id",
            "gt_S_ratio":           "gt_ctx_ratio",
            "gt_R_ratio":           "gt_lat_ratio",
            "gt_A_ratio":           "gt_lon_ratio",
            "speed_action_weights": "lon_action_weights",
            "route_sub_weights":    "lat_action_weights",
        }
        OmegaConf.set_struct(cfg, False)
        if cfg.model.get("_target_") in _TARGET_REMAP:
            cfg.model._target_ = _TARGET_REMAP[cfg.model._target_]
            print(f"[ckpt] remapped model _target_ → {cfg.model._target_}", flush=True)
        if cfg.model.vision_model.get("_target_") in _TARGET_REMAP:
            cfg.model.vision_model._target_ = _TARGET_REMAP[cfg.model.vision_model._target_]
            print(f"[ckpt] remapped vision_model _target_ → {cfg.model.vision_model._target_}", flush=True)
        for old_key, new_key in _PARAM_REMAP.items():
            if old_key in cfg.model:
                cfg.model[new_key] = cfg.model[old_key]
                del cfg.model[old_key]
                print(f"[ckpt] remapped param {old_key} → {new_key}", flush=True)

        # Apply ablation flags (override in subclasses)
        self._apply_ablation_flags(cfg)

        # Instantiate model
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.model = hydra.utils.instantiate(
            cfg.model, _recursive_=False,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)

        # Load checkpoint (supports DeepSpeed dir and plain .ckpt/.pt)
        ckpt_path = Path(self.config_path)
        if ckpt_path.is_dir():
            ds_model = ckpt_path / "checkpoint" / "mp_rank_00_model_states.pt"
            pt_bin = ckpt_path / "pytorch_model.bin"
            if ds_model.exists():
                ckpt_file = ds_model
            elif pt_bin.exists():
                ckpt_file = pt_bin
            else:
                raise FileNotFoundError(f"No model file in checkpoint dir: {ckpt_path}")
        else:
            ckpt_file = ckpt_path
        print(f"[ckpt] ckpt_file={ckpt_file}", flush=True)

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        ckpt = torch.load(str(ckpt_file), map_location="cpu")
        if isinstance(ckpt, dict) and "module" in ckpt:
            state = ckpt["module"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        if isinstance(state, dict):
            if any(k.startswith("model.") for k in state.keys()):
                state = {k[len("model."):]: v for k, v in state.items()}

        state = _remap_state_dict(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"[ckpt] missing keys (first 5): {missing[:5]}", flush=True)
        if unexpected:
            print(f"[ckpt] unexpected keys (first 5): {unexpected[:5]}", flush=True)
        self.model.eval()

        # ImageNet normalization tensors on GPU
        self.img_mean = IMAGENET_MEAN.to(self.device)
        self.img_std = IMAGENET_STD.to(self.device)

        # Prev template state (for autoregressive inference)
        self.prev_route_sub = torch.zeros(1, dtype=torch.long, device=self.device)
        self.prev_speed_action = torch.zeros(1, dtype=torch.long, device=self.device)
        self.prev_phase = torch.zeros(1, dtype=torch.long, device=self.device)

        # V3: SG filter window sizes
        self.sg_speed_window = 5   # speed_wps (N=10)
        self.sg_route_window = 9   # route_wps (N=64)

        self.iter = self.config_path.split("epoch=")[-1].split("/")[0]
        self.session = self.config_path.split("/")[-4]

        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # UKF
        if USE_UKF:
            self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
            self.ukf = UKF(dim_x=4, dim_z=4,
                           fx=bicycle_model_forward, hx=measurement_function_hx,
                           dt=self.carla_frame_rate, points=self.points,
                           x_mean_fn=state_mean, z_mean_fn=measurement_mean,
                           residual_x=residual_state_x, residual_z=residual_measurement_h)
            self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
            self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
            self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
            self.filter_initialized = False

        self.state_log = deque(maxlen=max((self.lidar_seq_len * self.data_save_freq), 2))

        self.save_path = os.environ.get('SAVE_PATH') + self.save_path_root
        if self.save_path is not None and route_index is not None:
            self.save_path = pathlib.Path(self.save_path) / route_index
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
            self.lon_logger = ScenarioLogger(
                save_path=self.save_path, route_index=route_index,
                logging_freq=self.logging_freq, log_only=True,
                route_only=False, roi=self.logger_region_of_interest,
            )

        self.debug_save_path = str(self.save_path) + '/debug_viz' + f'/{self.session}/iter_{self.iter}/{route_type}/{route_number}_{time.strftime("%Y_%m_%d_%H_%M_%S")}'
        Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
        self.save_path_metric = self.debug_save_path + '/metric'
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)

        if DEBUG:
            self.save_path_img = self.debug_save_path + '/images'
            Path(self.save_path_img).mkdir(parents=True, exist_ok=True)

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            earth_radius_equa = 6378137.0
            def equations(variables):
                x, y = variables
                eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                       - math.cos(x * math.pi / 180.0) * y)
                eq2 = (math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * earth_radius_equa
                       * math.cos(x * math.pi / 180.0) + locy - math.cos(x * math.pi / 180.0) * earth_radius_equa
                       * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
                return [eq1, eq2]
            solution = fsolve(equations, [0.0, 0.0])
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0.0, 0.0

        self._route_planner = RoutePlanner(self.route_planner_min_distance, self.route_planner_max_distance,
                                           self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}
        self.get_hero()

    def sensors(self):
        sensors = []
        for num_cam in self.config.num_cameras:
            sensors += [{
                'type': 'sensor.camera.rgb',
                'x': self.config.__dict__[f'camera_pos_{num_cam}'][0],
                'y': self.config.__dict__[f'camera_pos_{num_cam}'][1],
                'z': self.config.__dict__[f'camera_pos_{num_cam}'][2],
                'roll': self.config.__dict__[f'camera_rot_{num_cam}'][0],
                'pitch': self.config.__dict__[f'camera_rot_{num_cam}'][1],
                'yaw': self.config.__dict__[f'camera_rot_{num_cam}'][2],
                'width': self.config.__dict__[f'camera_width_{num_cam}'],
                'height': self.config.__dict__[f'camera_height_{num_cam}'],
                'fov': self.config.__dict__[f'camera_fov_{num_cam}'],
                'id': f'rgb_{num_cam}',
            }]

        if HD_VIZ:
            sensors += [{
                'type': 'sensor.camera.rgb',
                'x': -5.5, 'y': 0.0, 'z': 3.5,
                'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                'width': 1920, 'height': 1080, 'fov': 110,
                'id': 'rgb_viz',
            }]

        sensors += [
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': self.config.carla_frame_rate, 'id': 'imu'},
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': 0.01, 'id': 'gps'},
            {'type': 'sensor.speedometer', 'reading_frequency': self.config.carla_fps, 'id': 'speed'},
        ]
        return sensors

    def _preprocess_internvit(self, rgb_np):
        """Preprocess RGB image for InternViT: resize 448, ImageNet normalize.
        rgb_np: [H, W, 3] uint8 RGB
        Returns: [1, 1, 1, 3, 448, 448] half tensor on GPU
        """
        img = torch.from_numpy(rgb_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # [1,3,H,W]
        img = F_torch.interpolate(img, size=(448, 448), mode='bilinear', align_corners=False)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.half().unsqueeze(1).unsqueeze(1)  # [1,1,1,3,448,448]
        return img.to(self.device)

    @torch.inference_mode()
    def tick(self, input_data):
        try:
            if HD_VIZ:
                self.hd_cam_for_viz = input_data['rgb_viz'][1][:, :, :3]

            camera = input_data['rgb_0'][1][:, :, :3]
            _, compressed = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb = rgb[:int(rgb.shape[0] - (rgb.shape[0] * 4.8) // 16), :, :]
            rgb = np.array(rgb)
            self.image_buffer.append(rgb)

            # InternViT preprocessing (no LlavaNextProcessor needed)
            camera_images = self._preprocess_internvit(rgb)

            gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
            compass = t_u.preprocess_compass(input_data['imu'][1][-1])

            result = {'rgb': rgb, 'compass': compass}
            speed = input_data['speed'][1]['speed']

            if USE_UKF:
                if not self.filter_initialized:
                    self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
                    self.filter_initialized = True
                self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
                self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
                filtered_state = self.ukf.x
                self.state_log.append(filtered_state)
                result['gps'] = filtered_state[0:2]
            else:
                result['gps'] = np.array([gps_pos[0], gps_pos[1]])

            speed = round(input_data['speed'][1]['speed'], 1)

            waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))
            if len(waypoint_route) > 2:
                target_point, far_command = waypoint_route[1]
                next_target_point, next_far_command = waypoint_route[2]
            elif len(waypoint_route) > 1:
                target_point, far_command = waypoint_route[1]
                next_target_point, next_far_command = waypoint_route[1]
            else:
                target_point, far_command = waypoint_route[0]
                next_target_point, next_far_command = waypoint_route[0]

            if self.last_command_tmp != far_command:
                self.last_command = self.last_command_tmp
            self.last_command_tmp = far_command
            if (target_point != self.target_point_prev).all():
                self.target_point_prev = target_point
                self.commands.append(far_command.value)

            ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
            ego_target_point_torch = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
            ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])

            result['target_point'] = ego_target_point_torch
            target_points = [ego_target_point, ego_next_target_point]
            self.target_points = target_points.copy()
            target_points_np = np.array(target_points)
            result['route'] = torch.from_numpy(target_points_np).to(self.device, dtype=torch.float32).unsqueeze(0)

            result['speed'] = torch.FloatTensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32)

            # HLC from command
            hlc_val = COMMAND_TO_HLC.get(self.commands[-2], 4)
            hlc = torch.tensor([hlc_val], dtype=torch.long, device=self.device)

            dtype = next(self.model.parameters()).dtype
            W, H = rgb.shape[1], rgb.shape[0]

            self.driving_input = RoadGlyphInput(
                camera_images=camera_images.to(dtype=dtype),
                image_sizes=None,
                camera_intrinsics=get_camera_intrinsics(W, H, 110).unsqueeze(0).unsqueeze(0).float().to(self.device),
                camera_extrinsics=get_camera_extrinsics().unsqueeze(0).unsqueeze(0).float().to(self.device),
                vehicle_speed=result['speed'].to(dtype=dtype),
                map_route=result['route'].to(dtype=dtype),
                target_point=result['target_point'].to(dtype=dtype),
                hlc=hlc,
                prev_lat_action_id=self.prev_route_sub,
                prev_lon_action_id=self.prev_speed_action,
                prev_phase=self.prev_phase,
            )
            return result
        except Exception as e:
            print(f"[tick] CRASH step={self.step} err={repr(e)}", flush=True)
            raise

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        cam0 = input_data["rgb_0"][1]
        H, W = cam0.shape[0], cam0.shape[1]

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            return control

        tick_data = self.tick(input_data)

        out = self.model(self.driving_input)
        if isinstance(out, tuple):
            pred_speed_wps, pred_route = out
        else:
            pred_speed_wps, pred_route = out, None

        pred_speed_wps = pred_speed_wps.float() if pred_speed_wps is not None else None
        pred_route = pred_route.float() if pred_route is not None else None

        # V3: Savitzky-Golay smoothing on predicted waypoints
        pred_speed_wps = self._sg_smooth(pred_speed_wps, self.sg_speed_window)
        pred_route     = self._sg_smooth(pred_route,     self.sg_route_window)

        gt_velocity = tick_data['speed']

        if DEBUG and self.step % 5 == 0:
            tvec, rvec = None, None
            cam_viz = cam0
            if cam_viz.ndim == 3 and cam_viz.shape[2] > 3:
                cam_viz = cam_viz[:, :, :3]

            if HD_VIZ and hasattr(self, "hd_cam_for_viz") and self.hd_cam_for_viz is not None:
                cam_viz = self.hd_cam_for_viz
                tvec = np.array([[0.0, 3.5, 5.5]], np.float32)
                cam_rots = [0.0, -15.0, 0.0]
                rot_matrix = get_rotation_matrix(-cam_rots[0], -cam_rots[1], cam_rots[2])
                rvec = cv2.Rodrigues(rot_matrix[:3, :3])[0].flatten()

            H_viz, W_viz = cam_viz.shape[0], cam_viz.shape[1]
            camera_intrinsics = np.asarray(get_camera_intrinsics(W_viz, H_viz, 110))
            cam_viz_rgb = cv2.cvtColor(cam_viz, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cam_viz_rgb)
            draw = ImageDraw.Draw(image)

            if self.target_points is not None:
                target_point_img_coords = project_points(self.target_points, camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in target_point_img_coords:
                    draw.ellipse((points_2d[0]-4, points_2d[1]-4, points_2d[0]+4, points_2d[1]+4), fill=(0, 0, 255, 255))

            if pred_route is not None:
                pred_route_img_coords = project_points(pred_route[0].detach().cpu().numpy(), camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in pred_route_img_coords:
                    draw.ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(255, 0, 0, 255))

            if pred_speed_wps is not None:
                pred_speed_wps_img_coords = project_points(pred_speed_wps[0].detach().cpu().numpy(), camera_intrinsics, tvec=tvec, rvec=rvec)
                for points_2d in pred_speed_wps_img_coords:
                    draw.ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(0, 255, 0, 255))

            image.save(f"{self.save_path_img}/{self.step}.png")

        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if self.save_path_metric is not None and self.step % 1 == 0:
            with open(f"{self.save_path_metric}/metric_info.json", 'w') as outfile:
                json.dump(self.metric_info, outfile, indent=4)

        return control

    def _sg_smooth(self, wps: torch.Tensor, window: int, polyorder: int = 2) -> torch.Tensor:
        """Savitzky-Golay smoothing on waypoint sequence [1, N, D]."""
        if wps is None or wps.shape[1] < window:
            return wps
        np_wps = wps[0].cpu().numpy()
        smoothed = _savgol_filter(np_wps, window_length=window, polyorder=polyorder, axis=0)
        return torch.from_numpy(smoothed).unsqueeze(0).to(wps.device, dtype=wps.dtype)

    def get_metric_info(self):
        def vector2list(vector, rotation=False):
            if rotation:
                return [vector.roll, vector.pitch, vector.yaw]
            else:
                return [vector.x, vector.y, vector.z]

        output = {}
        output['acceleration'] = vector2list(self.hero_actor.get_acceleration())
        output['angular_velocity'] = vector2list(self.hero_actor.get_angular_velocity())
        output['forward_vector'] = vector2list(self.hero_actor.get_transform().get_forward_vector())
        output['right_vector'] = vector2list(self.hero_actor.get_transform().get_right_vector())
        output['location'] = vector2list(self.hero_actor.get_transform().location)
        output['rotation'] = vector2list(self.hero_actor.get_transform().rotation, rotation=True)
        return output

    def control_pid(self, route_waypoints, velocity, speed_waypoints):
        speed = velocity[0].data.cpu().numpy()
        speed_waypoints = speed_waypoints[0].data.cpu().numpy()

        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]) * 2.0

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        assert route_waypoints is not None, \
            "route_waypoints is None — set predict_route_as_wps=True in config"
        steer_wps = route_waypoints[0].data.cpu().numpy()
        route_interp = self.interpolate_waypoints(steer_wps.squeeze())
        steer = self.turn_controller.step(route_interp, speed)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake

    def interpolate_waypoints(self, waypoints):
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]
        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4
        interp = PchipInterpolator(dists, waypoints, axis=0)
        x = np.arange(0.1, dists[-1], 0.1)
        interp_points = interp(x)
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]
        return interp_points

    def _apply_ablation_flags(self, cfg):
        """Hook for subclasses to set ablation flags before model instantiation."""
        pass

    def destroy(self, results=None):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "config"):
            del self.config


# ── Filter Functions ──

def bicycle_model_forward(x, dt, steer, throttle, brake):
    front_wb = -0.090769015
    rear_wb = 1.4178275
    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0, locs_1, yaw, speed = x[0], x[1], x[2], x[3]
    accel = brake_accel if brake else throt_accel * throttle
    wheel = steer_gain * steer
    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)
    return np.array([next_locs_0, next_locs_1, next_yaws, next_speed])


def measurement_function_hx(vehicle_state):
    return vehicle_state


def state_mean(state, wm):
    x = np.zeros(4)
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(np.sum(np.dot(np.sin(state[:, 2]), wm)),
                       np.sum(np.dot(np.cos(state[:, 2]), wm)))
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x


def measurement_mean(state, wm):
    x = np.zeros(4)
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(np.sum(np.dot(np.sin(state[:, 2]), wm)),
                       np.sum(np.dot(np.cos(state[:, 2]), wm)))
    x[3] = np.sum(np.dot(state[:, 3], wm))
    return x


def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y
