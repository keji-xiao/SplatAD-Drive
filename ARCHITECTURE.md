# SplatAD-Drive architecture

```text
PandaSet 028 -> official PandaSetDataParser -> numerical + visual evidence
                                   |
                         official ns-train splatad
                                   |
                          config.yml + checkpoint
                                   |
CameraRequest / LiDARRequest -> SplatADDrive -> NeuradBackend
                                   |                |
                            ReferenceBackend        +-- official CNN/MLP
                                   |                +-- custom gsplat CUDA
                       PyTorch covariance splats
                                   |
                         RGB / range / intensity
                                   |
                    editing / metrics / export / benchmark
```

The public world is the parser's recentered world, not an invented raw GPS frame.
Camera poses are camera-to-world in OpenCV coordinates. The official adapter flips
the last two rotation columns when constructing Nerfstudio Cameras. The renderer
itself continues to use the pinned upstream camera/LiDAR implementation.
LiDAR axes are sensor-local and defined by the calibration; native PandaSet
forward is -Y, while azimuth zero means +X. Explicit LiDAR time offsets are
relative to the request timestamp and independent of the angular grid order.
Without offsets, scan_duration specifies a synthetic increasing-column scan.

Static reference Gaussians use actor id -1. Official checkpoints use their own
static id (= n_actors); checkpoint export does not conflate these conventions.
Dynamic reference means and orientations live in actor coordinates. Translation
interpolates linearly and rotation uses quaternion SLERP. Timestamps remain
float64 to retain subsecond precision even for Unix epoch values.

The reference renderer forms covariance R diag(s²) Rᵀ and projects it with a
perspective or spherical Jacobian. Front-to-back alpha compositing yields RGB,
accumulated depth, expected depth, and intensity. LiDAR evaluates each calibrated
elevation directly, wrapping azimuth residuals at ±pi. This is deliberately slow
and explicit, suitable for numerical and API checks. Memory is chunked by rays.

Official actor editing serializes access to a pipeline and restores temporary
state in finally blocks. It applies a rigid trajectory delta and masks opacity
for removal in both sensors. Private function globals override clip distances
without modifying upstream source/module functions. Adapter tests validate
restoration on exceptions; real checkpoint validation requires trained data.

Audits distinguish numerical PASS from visual review. Nominal timestamp overlays
are useful diagnostics but do not certify rolling shutter or dynamic timing.
File integrity checks are separately recorded from coordinate correctness.

Commands never silently switch an unavailable official checkpoint to a reference
scene. Missing libraries or data fail with an actionable error. Experiment output
records backend, input requests, checkpoint path, dataset, hardware and seeds.

Pruning is offline mask selection. Integrating it into training requires updating
all parameter tensors, optimizer states and MCMC buffers consistently; this
implementation intentionally does not advertise untested MCMC modifications.
