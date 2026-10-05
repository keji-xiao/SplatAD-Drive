# SplatAD-Drive

面向自动驾驶的 Camera / LiDAR Gaussian 场景接口、动态 actor 编辑、PandaSet 审计与实验工具。

## GitHub 仓库内容

仓库包含源码、训练配置、测试、上游补丁、实验报告，以及用于复核剪枝结论的精选 JSON 证据。`outputs/pruning_comparison/summary.json` 记录的输入文件与 SHA-256 一并保留；可运行 `python scripts/summarize_pruning_results.py` 重新汇总。

PandaSet 数据、训练 checkpoint、虚拟环境、编译缓存和完整渲染产物仅保留在实验机器，未随 Git 仓库上传。下文“当前机器立即运行”描述的是原实验环境；新克隆仓库请先运行参考后端示例，或按“真实 PandaSet 与官方训练”准备数据和上游依赖。固定上游版本见 `third_party/versions.json`，第三方归属见 `THIRD_PARTY_NOTICES.md`。

本机已有真实数据、训练模型和验证结果，直接运行与进度见 [交付说明](reports/DELIVERY.md)。

项目有两个显式后端：`NeuradBackend` 加载固定版本的**官方 SplatAD 模型**，使用它的 CUDA rasterizer、CNN/MLP 和 checkpoint；`ReferenceBackend` 是本项目实现的可微 PyTorch 几何参考，支持 CPU/GPU、小场景和自动测试。参考后端的运行结果不代表论文复现或真实场景训练效果。

## 当前机器立即运行

在 `C:\codex`，已建立 `.venv-gpu`，复用已有 conda `3dgs` 的 CUDA PyTorch。没有替换原环境的包。

本机真实 PandaSet 028 已下载并通过文件校验。截至 2026-09-30，已完成官方 SplatAD 的 **30000 步全序列、6 路 Camera/LiDAR 联合训练**：进程 exit 0，step 29999 checkpoint 含 **500 万个 Gaussian**；最后两个 checkpoint 均通过检查，各核验 136 个模型及优化器张量有限。断点训练也已实际执行两轮 3000→3002→3004，完成状态与 checkpoint 均通过检查。

最终 baseline 的官方 held-out 评估已完成，覆盖 **240 张图像、40 帧 LiDAR**：Camera PSNR **25.1963 dB**、SSIM **0.78331**、LPIPS **0.21774**。原生 **1920×1080 RGB + 64×1800 LiDAR** 已完成 3 个 actor 的移除、1 m 平移、15° yaw 旋转及恢复，恢复误差均为 0；另有 20 个 held-out front 时刻的联合渲染全部有限。2026-09-30 最新完整测试 **270 passed、1 skipped**，包含真实 CUDA（`reports/tests_release.log`）。

静态推理剪枝已通过完整官方评估，质量指标与 baseline 相同：Gaussian 数减少 **26.55%**，固定请求 LiDAR 吞吐从 **0.7218 提升到 0.9149 MR/s**。含真实角速度和零 RS 的两批公共输出对照均为 **44/44 精确一致**。详见 [剪枝结果与范围](reports/PRUNING_RESULT.md)。推理模型在 `outputs/pruned_baseline_alpha001/`，原始可续训模型保留。

已确认并修复 Camera 远裁剪导致的天空/远景丢失：公共默认 `far` 现为与官方一致的 `1e10 m`，LiDAR 仍为 200 m。视觉复核已查看 20 个 held-out front 时刻的全部 RGB/overlay contact 面板、其中两个完整 overlay，以及全部 3 个 actor 的编辑诊断图。天空和远景恢复，粗尺度道路/车辆/建筑布局相符；仍有边缘波纹、纹理模糊及遮挡残影，actor 0 的可见部分弥散，不能宣称全部车辆都能干净完整移除或逐像素对齐。详见 [30000 步结果](reports/BASELINE_RESULT.md)、[动态编辑结果](reports/DYNAMIC_EDITING_RESULT.md) 和 [实现状态](docs/IMPLEMENTATION_STATUS.md)；300 步 smoke 和 [3000 步结果](reports/INTERMEDIATE_RESULT.md) 保留为独立历史实验。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 render --load-config outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/config.yml --requests outputs/baseline_requests_native_motion.json --device cuda --output outputs/render_baseline
.\.venv-gpu\Scripts\python.exe -m pytest -m "not cuda and not upstream" -q
.\.venv-gpu\Scripts\python.exe -m splatad_drive doctor --output outputs/environment_gpu.json
```

第一条加载已训练的真实 30000 步模型，使用已保存的 front/Pandar64 标定、原生采样时刻和 rolling shutter，输出 RGB、depth、LiDAR NPZ/PLY 和 overlay。该请求为 **1920×1080 RGB、64×1800 LiDAR**，Camera far=1e10 m、LiDAR far=200 m，运行需要本机数据和 checkpoint。较低分辨率请求另存于 `outputs/baseline_requests_full_range.json`（960×540）。

普通 Python / Linux 也可运行参考后端：

```bash
python -m pip install -e '.[test]'
python -m splatad_drive demo --device cpu
python -m pytest -m 'not cuda and not upstream' -q
```

Demo 实际生成 RGB、LiDAR range、PLY、RGB/LiDAR overlay、车辆移除/平移和 rolling shutter 图像；执行 20 步 RGB + intensity 可微优化，保存 loss 曲线数值及场景 JSON。该优化只训练颜色/反射强度，用来验证梯度与运行链路。

## 真实 PandaSet 与官方训练

官方源固定在 `third_party/versions.json`，下载说明和本次验收见 `reports/`。仅使用当前环境的默认 Python 会得到 CPU 版 Torch，应使用上面的虚拟环境或下面的 Linux 环境。

```bash
python scripts/bootstrap_sources.py
python scripts/download_pandaset.py --help
# 下载器支持 Range 提取单独序列，避免下载 44.5GB 整包。
python -m splatad_drive audit --data data/pandaset --sequence 028 --frames 20
```

真实模型推荐 Ubuntu 22.04、Python 3.10、CUDA toolkit 11.8。创建独立环境后运行 `bash scripts/setup_linux.sh`。该脚本沿用官方依赖，执行真实 CUDA 测试并保存实际安装版本。上游依赖的可安装性仍需以日志为准，单纯 `import gsplat` 不足以证明 CUDA 编译成功。

```bash
python -m splatad_drive doctor --check-upstream
python -m pytest -m cuda -v
python -m splatad_drive train --config configs/smoke.yaml
```

最后一条默认打印完整 argv。审计通过后检查 `outputs/audit/lidar_camera_overlay/` 至少 20 帧及轨迹，填写 `configs/alignment_review.example.json` 的副本；`audit_sha256` 必须匹配实际审计文件，且需记录实际审阅者。执行：

```bash
python -m splatad_drive train --config configs/smoke.yaml --execute --audit outputs/audit/audit.json --alignment-review outputs/audit/alignment_review.json
```

Windows 使用 `scripts/run.ps1` 可自动设置本项目的 CUDA 编译缓存、已下载的 LPIPS 权重缓存及 UTF-8 输出：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 train --config configs/smoke.yaml --execute --audit outputs/audit/audit.json --alignment-review outputs/audit/alignment_review.json
```

上面的旧 `outputs/audit/` review 限定 front / 前 20% / 最多 300 步。后续 `outputs/audit_corrected/` 已完成 6 路相机、每路 20 帧共 120 张 overlay 的 contact-sheet 审阅，并抽查原尺寸图像；其 `alignment_review.json` 绑定该审计的 SHA256，覆盖全序列实验训练至 30000 步。nominal projection 仍有随时间变化的边缘偏差，这不是像素级 rolling-shutter 对齐或重建质量验收。

中间训练使用显式记录的 [actor 捕获时间补丁](patches/pandaset-actor-time.json)：优先取 Pandar64 原始框内返回点时间中位数，028 中 21 个 actor-frame 有真实返回、2 个使用校准相位 fallback，并保留少点标记。补丁源、SHA256、审计和运行证据都已保存，没有把静态 smoke 结果当作动态校时结论。重建固定源码环境时运行 `python scripts/apply_upstream_patches.py --help` 查看补丁应用方式。

三级配置为 `smoke.yaml`（300 步/front/前 20%）、`intermediate.yaml`（3000 步/6 cameras/全序列）、`baseline.yaml`（30000 步/6 cameras/全序列），三阶段均完成训练检查。分别写入独立 experiment 目录，保留 launch manifest、训练日志、官方 config/checkpoint/TensorBoard；baseline 每 2000 步保存，保留最近两个 checkpoint。训练完成和最终画质、编辑及性能验收分别记录。

复查已完成的 baseline 训练及其评估：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/check_training_run.py outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z --output outputs/baseline_training_check.json
.\.venv-gpu\Scripts\python.exe -m splatad_drive eval-official --load-config outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/config.yml --output outputs/eval_baseline_official.json
```

## Python API

```python
from splatad_drive import SplatADDrive, CameraRequest, LiDARRequest

drive = SplatADDrive.from_checkpoint("outputs/.../config.yml")
rgb = drive.render_camera(camera_request)  # rgb, depth, alpha
lidar = drive.render_lidar(lidar_request)   # range, intensity, hit_probability, ray_drop, points
drive.set_actor_pose(actor_id=0, timestamp=1.5, T_world_actor=T_world_actor)
drive.set_actor_visible(actor_id=0, visible=False)
```

`T_world_sensor` 把传感器局部坐标映射到 **官方 parser 的居中世界坐标**；时间是 parser-relative 秒，位置为米。相机公共 API 为 OpenCV 右/下/前，适配器负责转 Nerfstudio OpenGL。LiDAR 局部轴由标定定义，方位角零点为局部 +X、正仰角朝 +Z；PandaSet 原生前向为 -Y，参考示例前向为 +X。角度为弧度；四元数为 wxyz；速度为 sensor-local 米/秒。LiDAR 的可选 `time_offsets[A]` 或 `[E,A]` 指定相对请求时刻的逐束采样时间，优先于 `scan_duration` 生成的合成扫描顺序。CameraRequest 与 LiDARRequest 的完整字段见 `src/splatad_drive/` 中的 dataclass。

`SceneState` 可按一次请求覆盖 actor 状态。参考后端 pose 编辑插入轨迹关键帧；官方后端使用锚点处刚体增量作用于整条已学习轨迹，以保持 RS 内的运动连续性。两者语义差异详见 `docs/UPSTREAM.md`。

`SensorRig(T_ego_camera, T_ego_lidar).requests(T_world_ego, camera_request, lidar_request, timestamp=t)` 用固定标定同时生成 ego 新视角的两种请求，不分别平移传感器。

## 渲染、评估与 profiling

```bash
python -m splatad_drive render --scene outputs/demo/reference_scene.json --requests configs/demo_requests.json --output outputs/render
python -m splatad_drive render --load-config outputs/.../config.yml --device cuda --requests requests.json --edits edits.json
python -m splatad_drive inspect --scene outputs/demo/reference_scene.json --output outputs/gaussians
python -m splatad_drive evaluate --prediction predicted.npz --target target.npz
python -m splatad_drive eval-official --load-config outputs/.../config.yml
python scripts/check_training_run.py outputs/training/smoke_300
python -m splatad_drive profile --device cuda --warmup 10 --iterations 50
python -m splatad_drive profile --load-config outputs/.../config.yml --device cuda --requests requests.json --warmup 100 --iterations 500
```

请求 JSON 含 `camera` / `lidar`，字段与 dataclass 一致。两者均可传 `angular_velocity[3]`（sensor-local rad/s，默认零）；Camera 使用 CV 轴，LiDAR 使用原生轴。已保存带实际角速度的 `outputs/baseline_requests_native_motion.json`，可替换上面的旧请求文件。官方后端应用上游一阶 RS 模型，参考后端明确拒绝非零角速度。

`edits.json` 是列表，每项含 `actor_id` 及 `visible`，或 `timestamp` + `T_world_actor`。NPZ 评估字段：RGB=`rgb[H,W,3]`；LiDAR=`range/intensity/ray_drop[H,W]`、可选 `points[H,W,3]`；GT 的 `hit` 为 bool。GT 可提供同射线形状的二值 `evaluation_mask`，以排除未观测/填充格，LiDAR 指标和对应点的 Chamfer 均应用它；预测文件不能自行屏蔽错误。输出 PSNR/SSIM、range/intensity 误差、drop 统计；Chamfer 是双向平均**平方距离之和，单位 m²**。未提供 LPIPS 模型时明确输出 unavailable，绝不填写虚构值。

参考 profiling 明确标为 reference，不与论文 MR/s 对比。CUDA 使用事件和同步，并分别记录 wall time、分辨率、rays、显存；官方 profiling 必须提供固定 checkpoint/request。

最终 500 万 Gaussian 模型在 RTX 3090、固定原生请求下，100 次 warmup 后测量 500 次：**1920×1080 Camera 26.5046 FPS**，**64×1800 LiDAR 6.2653 FPS / 0.721761 MR/s**。两传感器分别测量，不是联合帧率；显存仅覆盖进程内 PyTorch allocator。完整同步时间、显存范围与非吞吐用途的算子 trace 见 [baseline 性能记录](reports/BASELINE_RESULT.md)。

## 算法扩展边界

`pruning.py` 实现 actor-aware / LiDAR-supported candidate selection：低支持且低 opacity 才成为候选、强 LiDAR hit 保护、每 actor 最小保留数、稳定排序。它返回可审查 mask，尚未修改官方 MCMC 优化器的 parameter/state；没有真实 A/B 指标时不宣称性能提升。

参考后端不包含学习型 CNN/MLP、MCMC densification、曝光时间积分或传感器角速度；ray-drop 使用 `1-alpha`，官方后端则返回实际学习型 ray-drop。局部 LiDAR 输出是各射线时刻的坐标，nominal overlay 不是完整运动去畸变检验。

## 来源

- [NeuRAD/SplatAD 官方模型与数据入口](https://github.com/georghess/neurad-studio)
- [SplatAD 官方 CUDA rasterizer](https://github.com/carlinds/splatad)
- [SplatAD 论文](https://arxiv.org/abs/2411.16816)
- [PandaSet 官方 devkit](https://github.com/scaleapi/pandaset-devkit)、[推荐数据镜像仓库](https://huggingface.co/datasets/georghess/pandaset)

第三方代码与数据保留各自许可证；本地不打包再分发数据。技术方案中的论文指标是背景参考，不是本项目已测结果。
