# PandaSet 028 官方 SplatAD 3000 步中间结果

**3000 次真实 Camera/LiDAR 联合训练和 checkpoint 数值检查 PASS。** 模型含 500 万个 Gaussian、3 个动态 actor，实际完成 actor 编辑与恢复，并在官方 held-out split 上执行评估。已审阅的公共请求图像使用 Camera far=200 m，天空可见伪影；后续发现这一裁剪范围小于官方范围，不能直接据此断言模型未收敛。本报告不是完整 baseline 或论文复现验收。记录更新至 2026-09-27；随后完成的 30000 步训练单独记录在 [BASELINE_RESULT.md](BASELINE_RESULT.md)，不改写本阶段数值。

## 运行与数据范围

| 项目 | 实际值 |
|---|---|
| Run | `outputs/training/intermediate_3000/splatad/2026-09-26_051956_834730Z/` |
| 配置 | `configs/intermediate.yaml`；实际官方配置和 argv 在 run 的 `config.yml`/`launch.json` |
| 数据 | PandaSet `028` 全序列，front/front_left/front_right/back/left/right + Pandar64 |
| Split | 官方 `train_split_fraction=0.5`；训练 240 图像/40 sweeps，held-out 评估 240 图像/40 sweeps |
| Seed / 预算 | 42 / 3000 次，零起始 0..2999 |
| 平台 | Windows、Python 3.10.6、PyTorch 2.7.1+cu118、RTX 3090 |
| NeuRAD 基础 commit | `8ba9b5116b8a2822a80c64f63a4ae64c5871aa68` |
| Rasterizer commit | `6e31ad766d39e0c33f9034a2ed772d51364b2343` |
| Actor 时间补丁 | `patches/pandaset-actor-time.patch`，清单 `patches/pandaset-actor-time.json` |
| Patch SHA256 | `6cbaacf33bd4dee8e73b653641fd68a5c888cabb2ebcf1c6e1d588c5d5d8955b`，launch 记录已应用 |
| Audit SHA256 | `24f7865e8f3b865160553dffce6d484cab146c78646502ee25eb963d851883df` |
| 退出记录 | `completion.json`: `returncode=0`、`status=exited_zero` |
| 完成时间 | 2026-09-26 05:28:43 UTC / 上海时间 13:28:43 |

修正后的 `outputs/audit_corrected/` 导出 240 张图像、40 帧 LiDAR 和 3 actor，数值检查通过。实际查看了 6 路相机各 20 帧、共 120 张 overlay 的 contact sheets，并抽查原尺寸 right_0210/back_0120 和前期 front/actor 证据。道路、建筑和停放车辆无明显整体轴向/尺度错位；近处侧后方仍有时间相关边缘偏差。审阅记录允许全序列实验训练，不表示逐像素 rolling-shutter 对齐或重建质量通过。

补丁在原始 dataset-world cuboid 内取 Pandar64 `d=0` 返回点的 float64 捕获时间中位数；028 的 21 个 actor-frame 使用真实返回、2 个无返回使用 camera-derived 校准 pose 和原生 -Y 前向相位。少点及 fallback 保留标记，见 `outputs/audit_corrected/actor_timing_diagnostics.json`。逐束训练 ray 的原始时间不被 actor 关键帧时间替代。

## Checkpoint 和训练记录

`outputs/intermediate_training_check.json` 实际读取 `nerfstudio_models/step-000002999.ckpt`，内部 step=2999，**5,000,000 个 Gaussian，134 个模型及优化器张量均有限**；参数形状和目标预算检查通过。推理加载报告 3 个 actor。

| TensorBoard tag | 首记录 step / value | 末记录 step / value | 记录数 |
|---|---:|---:|---:|
| `Train Loss` | 0 / 0.320640 | 2990 / 0.0662764 | 300 |
| `main_loss` | 0 / 0.317740 | 2990 / 0.0647405 | 263 |
| `depth_loss` | 30 / 0.799406 | 2900 / 0.0357046 | 37 |
| `intensity_loss` | 30 / 0.285948 | 2900 / 0.00389682 | 37 |
| `ray_drop_loss` | 30 / 0.100906 | 2900 / 0.0119940 | 37 |
| `alpha_sum_until_points_loss` | 30 / 0.0382745 | 2900 / 0.00992533 | 37 |
| `gaussian_count` | 0 / 2,604,056 | 2990 / 5,000,000 | 263 |

以上记录均有限；loss 是官方加权训练量，不是验证集误差。Camera/LiDAR 按不同 batch 采样，首末值不是同一观测的 A/B 对照。日志每 10 步抽样不能证明每个未记录值有限；最终模型检查覆盖 step 2999。训练越过 `warmup_length=500`，并观察到 Gaussian 数增长，与此前仍在 warmup 内的 300 步 smoke 不同。

上游 `Train Total (time)=449.179 s` 是训练循环计时；launch 至 completion 约 526.667 s，包含加载和初始化等开销。最后记录的 `GPU Memory (MB)=9483.576` 实际为 PyTorch 累计峰值 MiB，不是整张显卡占用或稳定渲染显存。两种计时均不作为渲染吞吐量。

## 官方 held-out 评估

源文件为 `outputs/eval_intermediate_official.json`，调用官方 `get_average_eval_image_metrics`，checkpoint step 2999，seed 42。覆盖全序列 **240 张 Camera 图像、40 帧 LiDAR**，FID 未请求。下表标准差沿用官方逐样本统计。

| 指标 | 均值 | 标准差 |
|---|---:|---:|
| Camera PSNR (dB) | 22.786295 | 1.227113 |
| Camera SSIM | 0.674620 | 0.044182 |
| Camera LPIPS | 0.464629 | 0.053359 |
| LiDAR intensity RMSE | 0.0633371 | 0.00292414 |
| LiDAR ray-drop accuracy | 0.981564 | 0.00477127 |
| LiDAR `depth_median_l2` (m²) | 0.00877150 | 0.00583881 |
| LiDAR `depth_mean_rel_l2` | 0.0130569 | 0.00707541 |
| LiDAR 官方 `chamfer_distance` | 0.554674 | 0.191449 |

`depth_median_l2` 是 range 残差平方的中位数，单位 m²，不应写成米或 RMSE。官方 Chamfer 使用双向平方距离总和除以 GT 点数，区别于本项目 NPZ evaluator 的双向均值之和；数值不能跨口径直接比较。ray-drop accuracy 也不能单独说明不同类别的召回质量。评估内的 FPS/rays-per-second 仅作过程信息，本报告不据此声称稳定渲染性能。

本次是一个真实序列和单个中间 checkpoint 的指标。旧 smoke 只用 front/前 20%，未包含同等动态内容，不能把两次评估数值差当作训练提升或退化，也不能与论文多序列指标直接等同。

## 动态编辑、ego 新视角与视觉缺陷

`outputs/validation_intermediate/validation.json` 是单个 **train** 样本的公共 API 推理检查，不是上述 held-out 评估。样本 index=39 对应 front 原始帧 79；Camera 为 **960×540**，LiDAR 为 **64×1800**。baseline、actor 移除/平移/恢复、ego 平移的所有检查输出均有限。

| 检查 | 实际结果 |
|---|---|
| Actor | checkpoint 内 3 个，本次编辑 actor 0 |
| 移除 | RGB 最大绝对变化 0.187476，LiDAR range 最大变化 9.163979 m |
| 平移 | 1.0 m；RGB 最大绝对变化 0.273895，LiDAR range 最大变化 7.094395 m |
| 恢复 | 参数恢复检查通过；Camera/LiDAR 已比较输出的最大绝对误差 **0.0** |
| Ego 新视角 | 两传感器共同平移 0.5 m；相对外参最大误差 **1.4901161e-8** |

此次启用 rolling shutter：Camera 使用原生 readout，LiDAR 使用经过真实点验证的 native -Y 前向相位估计，周期 **0.100007800 s** 来自原始未拆分的 sweep timestamps。该 novel raster 时间不是每束实际硬件测量；公共请求尚未接入角速度，传感器各自在本身采样时刻渲染，不假定同步。

生成的 RGB、depth、LiDAR NPZ/PLY、编辑图、恢复图和对照图保存在 `outputs/validation_intermediate/sample_000/`，训练曲线为 `outputs/intermediate_curves/curves.png`。主任务已实际查看 baseline、comparison 和 actor_edit_diagnostics，观察记录在 `outputs/validation_intermediate/visual_review.md`：**当前请求下天空和远处建筑有明显拉长伪影，细节仍模糊**；actor 0 位于 front 右侧小范围并被部分遮挡，外观弥散，裁剪图不足以证明完整干净的车辆移除。该请求使用 far=200 m，后续发现它会裁掉更远的背景 Gaussian；天空伪影的成因需用官方 far=1e10 m 对照，不能全归因于训练预算。自动报告的 `visual_quality=UNREVIEWED` 和 `NUMERIC_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED` 保持原样；另存的视觉观察不升级为整体质量 PASS。一次 actor 0 的数值响应与恢复，不代表全部 3 actor、所有帧或遮挡编辑均已验收。

## 保留的后续边界

300 步 smoke 及其失败重试、成功 run、评估和专用 profiling 原样保留，详见 `reports/SMOKE_RESULT.md` 和最新 `docs/IMPLEMENTATION_STATUS.md`。本次 3000 步结果不覆盖它们，也不把旧 run 的完成状态改写为成功。

后续完整 **30000 步 baseline 已完成训练检查**，对应另一个独立 run，结果见 `reports/BASELINE_RESULT.md`。本中间报告不提前填入最终模型的重建质量、动态验证、专用同步 profiling 或 pruning A/B 结论。真实两轮续训 3000→3002→3004 另见 `outputs/resume_validation.json`，其 PASS 验证四次新增迭代与保存/加载链路，不证明 GPU 逐位重放。

复查训练证据：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/check_training_run.py outputs/training/intermediate_3000/splatad/2026-09-26_051956_834730Z --output outputs/intermediate_training_check.json
```
