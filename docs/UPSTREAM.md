# Official SplatAD integration

The official algorithm remains in `third_party/neurad-studio` and the custom
`third_party/splatad` rasterizer. `NeuradBackend` does not edit their source.
One explicit parser patch is applied by `scripts/apply_upstream_patches.py`:
`patches/pandaset-actor-time.patch` fixes PandaSet actor capture times using
observed Pandar64 returns and a calibrated angular fallback. It preserves
the upstream base commit, verifies source/patch hashes, and records per-box
provenance. This is a corrected SplatAD-Drive data pipeline; dynamic runs
using it must not be described as unmodified upstream baseline results.
Training manifests bind the audited parser and timing-helper source hashes.
The CPU reference backend is a separate differentiable implementation for
geometry checks; its simple RGB/intensity model is not a pretrained SplatAD.

## Sources inspected

- [NeuRAD SplatAD model](https://github.com/georghess/neurad-studio/blob/main/nerfstudio/models/splatad.py)
- [PandaSet parser](https://github.com/georghess/neurad-studio/blob/main/nerfstudio/data/dataparsers/pandaset_dataparser.py)
- [AD parser normalization and timestamps](https://github.com/georghess/neurad-studio/blob/main/nerfstudio/data/dataparsers/ad_dataparser.py)
- [LiDAR tile preparation](https://github.com/georghess/neurad-studio/blob/main/nerfstudio/data/datamanagers/full_images_lidar_datamanager.py)
- [Dynamic actor interpolation](https://github.com/georghess/neurad-studio/blob/main/nerfstudio/model_components/dynamic_actors.py)
- [Checkpoint loading](https://github.com/georghess/neurad-studio/blob/main/nerfstudio/utils/eval_utils.py)
- [Custom CUDA rasterizer](https://github.com/carlinds/splatad)

Use the checked-out commits recorded in the repository's upstream lock file;
links above identify the source files, not an instruction to update to HEAD.
The inspected NeuRAD checkout is `8ba9b5116b8a2822a80c64f63a4ae64c5871aa68`.

## Loading and rendering

```python
from pathlib import Path
from splatad_drive.backends.neurad import NeuradBackend

backend = NeuradBackend.from_config(
    Path("outputs/experiment/splatad/run/config.yml"),
    data=Path("data/pandaset"),  # optional relocation
)
rgb = backend.render_camera(camera_request)
scan = backend.render_lidar(lidar_request)
```

The wrapper uses `eval_setup(..., strict_load=True)` and calls the official
`get_camera_outputs` / `get_lidar_outputs`. A checkpoint must be an official
SplatAD model. Configurations/checkpoints contain serialized Python objects;
load only trusted experiment files. The official datamanager can still need
the dataset in inference mode. Camera/LiDAR appearance embeddings come from
the checkpoint; arbitrary new sensor identities cannot be added at inference.

Camera IDs accept parser sensor names (such as `front`) or their numeric
embedding indices. LiDAR IDs accept a name (`Pandar64`) or a zero-based ordinal
among checkpoint LiDAR sensors (`0` normally maps to embedding index `6`).

Public camera transforms map OpenCV camera coordinates (right, down, forward)
to the **parser-centered world**. The adapter right-multiplies their rotation
by `diag(1,-1,-1)` before constructing Nerfstudio `Cameras`. Timestamp zero is
the official parser's minimum sensor timestamp, not Unix epoch. The exported
`time_offset` recovers absolute time. Neither world normalization nor time
offset is applied twice. Distance is in metres.

The public LiDAR basis is the upstream rasterizer's Cartesian spherical basis:
`x=r*cos(elev)*cos(azim)`, `y=r*cos(elev)*sin(azim)`, `z=r*sin(elev)`.
Do not relabel or rotate a PandaSet parser's calibration based on a sensor-name
assumption: its `T_world_lidar` already defines the actual physical axes.
In the native PandaSet frame the vehicle's forward direction is **-Y**, whereas
reference examples use +X forward. Azimuth zero always means local +X; no
implicit conversion to a forward/left/up frame is performed.

The current official adapter supports full uniform 360-degree azimuth samples
starting at `-pi`, ending before `pi`, and strictly increasing elevations whose
count is a multiple of eight. The last azimuth tile is padded to 32 columns
internally, then removed. Elevations need not be uniformly spaced. Use the
Pandar64 beam calibration for experiments intended to reproduce PandaSet.
Partial FOV, descending azimuth grids, nonuniform azimuth, and a changed beam
divergence are rejected. Capture order can differ from angle order. The
reference backend supports more general angular grids.

`LiDARRequest.time_offsets` accepts finite seconds relative to `timestamp`,
either `[A]` per column or `[E,A]` per beam. These times are authoritative even
when `scan_duration=0`, and can decrease or wrap within the increasing azimuth
grid. Measured hardware times should be supplied when available. Both backends
preserve their origin; the official model internally moves pose and timestamp
together when centering times for its CUDA kernel.

Without explicit offsets, `scan_duration` generates a synthetic increasing
azimuth schedule centered on the request timestamp. This is not the PandaSet
hardware scan direction. The checkpoint validator's `--rolling-shutter` mode
uses the native phase estimate
`atan2(-cos(azimuth), -sin(azimuth)) / (2*pi) * scan_period`: -Y forward occurs
at zero, and +X occurs at `-scan_period/4`. The period comes from the complete
raw LiDAR timestamp sequence, not subsampled train/eval times. This phase was
checked against real 028 capture times, but is still an angular estimate for
the requested raster; it does not reproduce irregular per-return timing.

`linear_velocity` (m/s) and optional `angular_velocity` (rad/s, default zero)
are sensor-local three-vectors. Camera uses CV right/down/forward axes; the
adapter converts both vectors to GL metadata and upstream converts them back
for its projection kernel. LiDAR retains its native local axes. The official
backend applies upstream's first-order RS correction, not full exposure
integration. The reference backend explicitly rejects nonzero angular velocity.
The old saved requests omit angular velocity; `baseline_requests_native_motion.json`
and `validation_motion_requests/` preserve native measured metadata separately.
Camera timestamps are
the center-row capture time. The adapter disables learned pose/time offsets
for explicitly supplied novel poses so the request is honored exactly.

The wrapper preserves the official learned RGB CNN and intensity/ray-drop
MLP. Camera `depth` is expected camera-z depth. Official LiDAR `depth` is the
alpha-weighted range sum; the adapter returns `range=depth/alpha` and also
retains `depth_sum` and `median_range`. Empty rays return zero range.
`ray_drop` is the learned probability, `alpha` is geometric accumulation,
and `hit_probability=alpha*(1-ray_drop)` is the adapter's joint proxy.
`points` is an organized `[elevation,azimuth,3]` instantaneous local scan;
filter on hit probability before exporting a return point cloud. Do not
compare normalized `range` to the paper's unnormalized `depth` metric without
explicitly identifying which quantity was evaluated.

The official renderer hardcodes clipping distances. To honor request near/far,
the adapter creates a private Python function binding to the exact upstream
render code, with a rasterizer closure overriding only clipping arguments.
It never replaces the process-global rasterizer. Camera pixel variance remains
the upstream fixed `0.3`; changing it is explicitly rejected.

## Editing and concurrency

```python
backend.set_actor_pose(actor_id, timestamp, T_world_actor)
backend.set_actor_visible(actor_id, False)
backend.clear_edits()
```

`actor_id` is the zero-based trajectory-list position, not the annotation UUID.
The static Gaussian ID is the number of actors and cannot be edited as an actor.
Pose edits are anchored trajectory transforms: the requested pose at time `t`
defines `delta = target @ inverse(original(t))`. The entire trajectory and
world linear velocity receive this delta, retaining motion during scan/readout.
This operation is an anchored rigid edit, not keyframe insertion.

The inspected official model transforms actor means but supplies Gaussian
quaternions directly to the rasterizer. Applying a full actor rotation again
would change an unedited checkpoint. The adapter therefore rotates covariance
quaternions by the **edit delta only**. This preserves the official baseline
and consistently rotates an edited anisotropic object.

Visibility temporarily sets the relevant opacity logits to negative infinity.
Actor methods, opacity logits, quaternion values and request configuration are
restored in `finally` blocks, including after a renderer exception. A reentrant
lock serializes access to each backend instance. One backend owns its pipeline;
do not share that same mutable model with another backend or render it directly
from another thread. The adapter is an inference API and does not preserve
autograd for training through scene edits.

## PandaSet audit

`audit_pandaset` instantiates `PandaSetDataParserConfig`, preserving its official
default training split of `0.5`, 6 cameras, Pandar64 and cuboid handling. Audit
uses `allow_per_point_times=True` and `add_missing_points=False`; the training
method may separately add synthetic nonreturn rays. `max_frames` limits exported
evidence, not the parser's input split.

Exports include camera intrinsics, both camera conventions, parser normalization,
raw upstream calibration, sensor/actor timestamps, selected LiDAR NPZ sweeps,
world point PLY, actor trajectories/bbox occupancy counts and front-camera
projection overlays. Point columns are `[x,y,z,intensity,relative_time]`.
Per-point times are not required to be ordered; sensor/actor times are checked
per trajectory. PandaSet raw points are already ego-compensated in world space;
the official parser expresses them in a common scan-reference frame. Audit
projection uses that frame directly, without a second deskew. The training
datamanager separately removes ego motion to build its per-point spherical
rasterization inputs.

`ego_trajectory.ply` deliberately represents the Pandar64 origin as a rig
trajectory proxy, recorded in `ego_trajectory.json`; it is not a separately
measured rear-axle/body trajectory. Actor dimensions are upstream **width,
length,height**, not the length,width,height order proposed in the initial plan.

Numerical checks can pass independently; `numeric_checks_passed` records that
result. Automatic `audit.json` remains `REVIEW_REQUIRED` / `NO-GO` until visual
review is separately recorded. Nominal-pose LiDAR overlays do not prove camera
rolling shutter, moving actor timing, semantic alignment or absence of trajectory
jumps. Actor bbox occupancy is a descriptive count, never a correctness PASS.
No metric here claims the official model was trained or reproduced paper scores.

## Tests and actual environment

`tests/test_upstream_contract.py` checks camera conversion, tile padding/timing,
SE(3) rejection, private clipping overrides, actor edit restoration after an
exception, and audit status semantics. Its source-contract check parses the
real checked-out upstream module. CPU/fake-model tests are not a CUDA checkpoint
integration test; actual parser, rasterizer and training results belong in the
execution reports.

On the current Windows host the isolated `.venv-gpu` inherits the existing
CUDA-enabled PyTorch installation. NumPy `2.2.6` produced a process abort in
linear algebra after importing PyTorch; a local `numpy==1.24.4` override fixed
the reproduction. This also fits upstream's `numba==0.57` requirement. The
upstream `opencv-python==4.10.0.84` pin avoids the newer OpenCV NumPy 2 requirement.
These changes are confined to the project environment, not the Conda base.

On 2026-09-23 the actual isolated Windows environment successfully imported
`SplatADModelConfig`, executed `train.py splatad --help`, and executed
`train.py splatad pandaset-data --help` (all exit code 0). The project's exact
300-step smoke command also parsed successfully with an appended `--help`;
this validation did not execute training. Evidence is in
`outputs/official_train_help.log`, `outputs/official_pandaset_help.log`, and
`outputs/official_smoke_command_help.log`. Missing optional Waymo/py123d
parsers print notices but do not block PandaSet.

The official parser registry imports Argoverse, nuScenes and ZOD even for a
PandaSet run. The missing dataset dependencies and their transitive binary
imports were installed into `.venv-gpu`, with commands retained in
`outputs/upstream_dependency_commands.ps1` and the installation log in
`outputs/upstream_dependency_install.log`. The `--no-deps` installs protect
the working NumPy/PyTorch versions; the script is an incremental repair for
this environment, not a standalone replacement for full environment setup.
`outputs/requirements_windows_gpu_actual.txt` records local venv packages.
