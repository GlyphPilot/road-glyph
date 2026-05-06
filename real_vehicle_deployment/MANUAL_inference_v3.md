# RoadGlyph inference_v3 Manual

## Overview

`inference_v3` is the real-time autonomous driving inference program for the RoadGlyph project.
It runs an ONNX neural network model on the NVIDIA DRIVE Pegasus platform, reads live camera images
via GMSL, and optionally controls the vehicle steering and speed over CAN bus.

**Platform:** NVIDIA DRIVE Pegasus (aarch64, CUDA 10.2, DriveWorks 2.2)
**Vehicle:** Hyundai Santafe (wheelbase 2.765 m, steering ratio 12.0)


---

## 1. Building

```bash
cd /path/to/real_vehicle_deployment
mkdir -p build && cd build
cmake ..
make inference_v3 -j4
```

The binary is produced at `build/inference_v3`.

### Dependencies (already configured in CMakeLists.txt)

| Dependency       | Location                                          |
|------------------|---------------------------------------------------|
| ONNX Runtime 1.10| `/media/nvidia/GLAD/ort_build/build/Release`      |
| DriveWorks SDK   | `/usr/local/driveworks-2.2/targets/aarch64-Linux` |
| CUDA 10.2        | `/usr/local/cuda-10.2`                            |
| GLFW             | Bundled with DriveWorks samples (3rdparty)        |


---

## 2. Running

```bash
cd build
./inference_v3 [OPTIONS]
```

### Minimal Example (defaults)

```bash
./inference_v3 --model /path/to/road_glyph_fp32_wp64_op15_v3.onnx
```

This starts with:
- CUDA GPU inference
- GMSL camera group `c`, sensor type `ar0231-rccb-bae-sf3324`
- GUI visualization window (1280x800)
- HLC = follow_lane (4)
- No vehicle control (waypoint prediction only)


---

## 3. Command-Line Options

### Model & Camera

| Option                  | Description                                | Default                                      |
|-------------------------|--------------------------------------------|----------------------------------------------|
| `--model FILE` / `-m`  | Path to the ONNX model file                | `road_glyph_fp32_wp64_op15_v3.onnx`|
| `--camera-type TYPE`    | GMSL camera sensor type string             | `ar0231-rccb-bae-sf3324`                     |
| `--camera-group GRP`    | Camera group: `a`, `b`, `c`, or `d`        | `c`                                          |

### Inference

| Option         | Description                              | Default     |
|----------------|------------------------------------------|-------------|
| `--cpu`        | Force CPU-only inference (disable CUDA)  | GPU (CUDA)  |
| `--speed MPS`  | Initial vehicle speed in m/s             | `0.0`       |
| `--hlc VALUE`  | Initial High-Level Command (see below)   | `4`         |

### Display & Recording

| Option            | Description                                                              | Default |
|-------------------|--------------------------------------------------------------------------|---------|
| `--no-gui`        | Headless mode: no visualization window, console-only output              | GUI on  |
| `--save-frames`   | Save every camera frame as PPM image + speed to disk                     | off     |
| `--log-can`       | Log CAN + RTK data at 20 Hz to a timestamped CSV file (for evaluation)  | off     |

### RTK GPS

| Option             | Description                                     | Default                  |
|--------------------|-------------------------------------------------|--------------------------|
| `--rtk IP:PORT`    | NovAtel RTK TCP address                         | `192.168.1.100:3002`     |
| `--no-rtk`         | Disable RTK GPS connection                      | RTK enabled by default   |

RTK is enabled by default. Receives GPGGA at 10 Hz from NovAtel ICOM2 (TCP 3002).
An NTRIP relay must forward correction data to ICOM1 (TCP 3001) to achieve RTK Fix.

### Vehicle Control Mode

By default, the program only predicts waypoints and does NOT send any commands to the vehicle.
Use one of these flags to enable CAN-based vehicle control:

| Option              | Steering | Speed (Accel/Brake) | Description                  |
|---------------------|----------|---------------------|------------------------------|
| *(none)*            | No       | No                  | Waypoint prediction only     |
| `--pp-steer`        | Yes      | No                  | Pure Pursuit steering only   |
| `--pp-full`         | Yes      | Yes                 | Pure Pursuit + speed control |
| `--stanley-steer`   | Yes      | No                  | Stanley steering only        |
| `--stanley-full`    | Yes      | Yes                 | Stanley + speed control      |

**WARNING:** `--pp-full` and `--stanley-full` will send real throttle/brake commands to the vehicle
over CAN. The program will display a safety confirmation prompt and require you to type `YES`
before starting. A 3-second countdown follows.

Speed is capped at 15 km/h with a safety brake that activates above 15 km/h and releases below 13 km/h.

### Help

```bash
./inference_v3 --help
./inference_v3 -h
```


---

## 4. High-Level Command (HLC) Values

The HLC tells the model what driving maneuver to plan:

| Value | Name           | Description                    |
|-------|----------------|--------------------------------|
| 1     | LEFT           | Turn left at intersection      |
| 2     | RIGHT          | Turn right at intersection     |
| 3     | STRAIGHT       | Go straight at intersection    |
| 4     | FOLLOW_LANE    | Follow current lane (default)  |
| 5     | CHANGE_LEFT    | Change to the left lane        |
| 6     | CHANGE_RIGHT   | Change to the right lane       |

Set the initial HLC with `--hlc VALUE`. You can change it at runtime with keyboard keys (see below).


---

## 5. Keyboard Controls (Runtime)

### GUI Mode (default)

| Key              | Action                                            |
|------------------|---------------------------------------------------|
| `1` or `Left`    | Set HLC to LEFT                                   |
| `2` or `Right`   | Set HLC to RIGHT                                  |
| `3`              | Set HLC to STRAIGHT                               |
| `4` or `Space`   | Set HLC to FOLLOW_LANE                            |
| `5`              | Set HLC to CHANGE_LEFT                            |
| `6`              | Set HLC to CHANGE_RIGHT                           |
| `O`              | Toggle override: MANUAL (default) / AUTO          |
| `T`              | Cycle through tuning parameters                   |
| `Up Arrow`       | Increase selected tuning parameter by +0.5        |
| `Down Arrow`     | Decrease selected tuning parameter by -0.5        |
| `ESC`            | Quit the program                                  |

### Headless Mode (`--no-gui`)

Same keys work via terminal raw input. `ESC` (alone) also quits.

### Override Toggle (O key)

- **MANUAL** (default): CAN control commands are NOT sent to the vehicle, even if a control mode is selected. The model still predicts waypoints and steering values are computed but not applied.
- **AUTO**: CAN control commands are actively sent to the vehicle. The steering/speed values computed by the selected controller are applied.

You MUST press `O` to switch to AUTO before the vehicle will respond to commands.


---

## 6. Tuning Parameters

When a control mode is active, press `T` to cycle through tunable parameters,
then use `Up/Down` arrow keys to adjust by +/-0.5:

| Parameter       | Controller    | Description                              | Default | Min   |
|-----------------|---------------|------------------------------------------|---------|-------|
| PP ld_min       | Pure Pursuit  | Minimum look-ahead distance (m)          | 5.0     | 1.0   |
| PP ld_max       | Pure Pursuit  | Maximum look-ahead distance (m)          | 10.0    | 2.0   |
| PP ld_gain      | Pure Pursuit  | Speed multiplier for look-ahead (m/mps)  | 1.8     | 0.1   |
| ST k_gain       | Stanley       | Cross-track error gain                   | 2.5     | 0.1   |
| ST k_soft       | Stanley       | Softening constant (prevents division by zero at low speed) | 0.5 | 0.1 |

The current tuning target and its value are displayed on the GUI OSD and printed to the console
when changed.


---

## 7. Model Inputs and Outputs

### Inputs

| Name      | Shape                    | Type    | Description                          |
|-----------|--------------------------|---------|--------------------------------------|
| camera    | [1, 1, 1, 3, 448, 448]  | float32 | RGB image, ImageNet-normalized       |
| speed     | [1, 1]                   | float32 | Current vehicle speed in m/s         |
| hlc       | [1]                      | int64   | High-level command (1-6)             |

Image preprocessing: bottom 4.8/16 of the frame is cropped, then bilinear-resized to 448x448,
normalized with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

### Outputs

| Name            | Shape         | Type    | Description                                                 |
|-----------------|---------------|---------|-------------------------------------------------------------|
| speed_wps       | [1, 10, 2]   | float32 | 10 speed waypoints (x=forward m, y=left m) in ego frame     |
| route_wps       | [1, 64, 2]   | float32 | 64 route waypoints (x=forward m, y=left m) in ego frame     |
| lat_logits      | [1, 4]       | float32 | Lateral action: other, go_around, overtake, give_way         |
| lon_logits      | [1, 8]       | float32 | Longitudinal action: other, remain_stop, stop_now, slow_down, maintain, maint_red, increase, wait_gap |
| ctx_speed_kind  | [1, 4]       | float32 | Speed context kind: none, dynamic, static, policy            |
| ctx_speed_sub   | [1, 6]       | float32 | Speed context sub: NA, vehicle, pedestrian, stop_sign, traffic_light, construction |
| ctx_route_kind  | [1, 3]       | float32 | Route context kind: none, static, policy                     |
| ctx_route_sub   | [1, 5]       | float32 | Route context sub: NA, construction, lane_change, turn, route_adj |


---

## 8. CAN Bus Protocol

The program communicates over `can0` SocketCAN interface.

### Reading (Vehicle Feedback)

| CAN ID | Name             | Signal               | Bytes | Encoding                       |
|--------|------------------|----------------------|-------|--------------------------------|
| 0x51   | Vehicle_Info_1   | Steering_Angle_FB    | 5-6   | 16-bit LE signed, x0.1 deg    |
| 0x51   | Vehicle_Info_1   | Switch_state         | 7     | 8-bit unsigned, bitmask        |
| 0x52   | Vehicle_Info_2   | Vehicle_Speed        | 1-2   | 16-bit LE unsigned, x0.1 km/h |
| 0x52   | Vehicle_Info_2   | Override_feedback    | 0     | 8-bit unsigned, enum           |

#### Switch_state Bit Mapping (0x51 byte 7)

| Bit | Value | Name   | Note                                                                 |
|-----|-------|--------|----------------------------------------------------------------------|
| 0   | 1     | E-stop | Emergency stop button                                                |
| 1   | 2     | Auto   | Autonomous driving standby switch                                    |
| 2   | 4     | APM    | Automatic parking mode switch                                        |
| 3   | 8     | ASM    | Automatic steering switch                                            |
| 4   | 16    | —      | Listed as AGM in DBC but does not match vehicle behavior             |
| 5   | 32    | AGM    | Automatic acceleration/braking switch (confirmed on vehicle)         |

**Note:** The DBC file defines AGM=16, but vehicle testing confirms AGM=32 (bit 5).

#### Override_feedback Values (0x52 byte 0)

| Value | State              | Description                                               |
|-------|--------------------|-----------------------------------------------------------|
| 0     | Receiving commands | Control_CMD being sent but Auto not yet activated         |
| 1     | Auto active        | Autonomous control is engaged                             |
| 2     | Steer override     | Driver has gripped the wheel and intervened in steering   |
| 3     | Accel override     | Driver has pressed the accelerator pedal                  |
| 4     | Brake override     | Driver has pressed the brake pedal                        |
| 5     | Idle               | No control commands being sent (default state)            |
| 6     | E-Stop             | Emergency stop activated                                  |

**Note:** DBC signal names differ from actual vehicle behavior (e.g., DBC "Manual"=0 → actual meaning is "receiving commands").

### Writing (Control Commands) — only when override = AUTO

**Control_CMD (CAN ID 0x150)** — sent every 20 ms:

| Byte | Field              | Description                                   |
|------|--------------------|-----------------------------------------------|
| 0    | Override           | 0 = manual, 1 = autonomous                    |
| 1    | Alive_Count        | Rolling counter 0-255                         |
| 5    | Angular_Speed_CMD  | Steering wheel rotation speed (fixed at 50)   |

**Driving_CMD (CAN ID 0x152)** — sent every 20 ms when override = AUTO:

| Byte | Field   | Range      | Description                     |
|------|---------|------------|---------------------------------|
| 0-1  | Accel   | 650-3400   | Accelerator (650 = idle)        |
| 2-3  | Brake   | 0-17000    | Brake pressure                  |
| 4-5  | Steer   | int16      | Steering angle in degrees       |
| 6    | Gear    | 5          | Gear (5 = D)                    |
| 7    | Reserved| 0          | Unused                          |

On shutdown, the program sends a final Control_CMD with Override = 0 to return to manual mode.


---

## 9. GUI Visualization

The 1280x800 window displays:

- **Background:** Live GMSL camera feed
- **Red line + yellow markers:** Route waypoints (64 points) in BEV overlay (every 5th point marked)
- **Green line:** Speed waypoints (10 points)
- **White dot:** Vehicle position (bottom center)
- **Cyan dot:** Pure Pursuit target waypoint (when PP control active)
- **Top-left OSD:** Inference time, FPS, target speed, CAN speed, steering feedback, HLC, control mode, override status, tuning parameters, frame count
- **Top-right OSD:** Token predictions (lat_action, lon_action, spd_kind, spd_sub, rte_kind, rte_sub) with softmax probabilities


---

## 10. Data Logging

### Frame Saving (`--save-frames`)

Saves every camera frame as PPM (P6 binary) with the filename pattern:
```
{save_dir}/{frame_number:06d}_spd{speed_kmh}.ppm
```

### CAN Logger (`--log-can`)

Creates a CSV file named `can_log_{YYYYMMDD_HHMMSS}.csv` in the working directory
with columns recorded at 20 Hz:

```
timestamp_us,speed_kmh,steering_angle_deg,override,hlc,rtk_lat,rtk_lon,rtk_quality,switch_state,override_feedback
```

| Column             | Description                                                                      |
|--------------------|----------------------------------------------------------------------------------|
| timestamp_us       | Unix timestamp (microseconds)                                                    |
| speed_kmh          | Vehicle speed from CAN (km/h)                                                    |
| steering_angle_deg | Steering angle feedback from CAN (deg)                                           |
| override           | Software override flag (0 = manual, 1 = autonomous)                              |
| hlc                | High-Level Command (1–6)                                                         |
| rtk_lat            | RTK latitude (WGS84, 8 decimal places)                                           |
| rtk_lon            | RTK longitude (WGS84, 8 decimal places)                                          |
| rtk_quality        | GPGGA quality (0=none, 1=GPS, 2=DGPS, 4=RTK Fix, 5=RTK Float)                  |
| switch_state       | Vehicle switch bitmask (bit0=Estop, 1=Auto, 2=APM, 3=ASM, 5=AGM)               |
| override_feedback  | Vehicle control state (0=receiving, 1=Auto, 2=Steer, 3=Accel, 4=Brake, 5=Idle, 6=Estop) |


---

## 11. Stopping the Program

- Press `ESC` (GUI or terminal)
- Press `Ctrl+C` (sends SIGINT)
- The program handles SIGINT/SIGTERM gracefully: stops CAN transmission (sends override=0), flushes logs, releases DriveWorks and CUDA resources, and prints inference statistics.


---

## 12. Common Usage Scenarios

### Visualization only (no vehicle control)
```bash
./inference_v3 -m /path/to/model.onnx
```

### Steering test with Pure Pursuit
```bash
./inference_v3 -m /path/to/model.onnx --pp-steer
# Then press O at runtime to enable AUTO
```

### Full autonomous driving with Stanley + data logging + RTK
```bash
./inference_v3 -m /path/to/model.onnx --stanley-full --log-can
# RTK is enabled by default (192.168.1.100:3002)
# Type YES at the safety prompt, then press O to engage
```

### Run without RTK
```bash
./inference_v3 -m /path/to/model.onnx --log-can --no-rtk
```

### Headless data collection (no display)
```bash
./inference_v3 -m /path/to/model.onnx --no-gui --save-frames --log-can
```

### CPU-only test
```bash
./inference_v3 -m /path/to/model.onnx --cpu --no-gui
```


---

## 13. Troubleshooting

| Problem                          | Solution                                                         |
|----------------------------------|------------------------------------------------------------------|
| `onnxruntime_cxx_api.h` not found | Check that ORT path in CMakeLists.txt matches your installation |
| Camera start fails               | Verify camera group (`--camera-group a/b/c/d`) and cable connections |
| CAN interface not found          | Run `sudo ip link set can0 up type can bitrate 500000`           |
| No speed reading from CAN        | Check that the vehicle ignition is on and CAN bus is active      |
| GLFW window creation fails       | Ensure a display is connected, or use `--no-gui`                 |
| ORT model load fails             | Verify the ONNX file path and opset compatibility                |
