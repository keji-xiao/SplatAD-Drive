# SplatAD-Drive 实现与证据边界

本页更新至 **2026-09-30**。真实 PandaSet 028 的官方 SplatAD **30000 步、6 路相机、全序列联合训练已完成并通过 checkpoint 检查**，最终官方 held-out 评估、原生分辨率 3 actor 编辑数值检查和 20 个 held-out front 时刻的联合渲染也已完成。此前 300/3000 步阶段独立保留，最终固定原生请求的同步性能已实测；20 个 front 时刻和 3 actor 诊断图的视觉复核已完成并记录已知伪影，数值 PASS 不等于完整质量或论文复现。

## 已取得的证据

9 月 30 日新增：静态推理剪枝的完整官方评估与 baseline 指标相同，两批公共输出比较各 44/44 精确一致；带实际角速度的三 actor 编辑恢复误差均为 0。详见 [剪枝报告](../reports/PRUNING_RESULT.md) 和 [交付进度](../reports/DELIVERY.md)。Nsight 硬件计数器因驱动权限未取得，不能算作完成。

| 层次 | 实际完成 | 原始证据 | 能证明的范围 |
|---|---|---|---|
| 参考后端 | 462 个 Gaussian 合成道路；Camera/LiDAR 渲染、actor 编辑、ego 新视角、简化 rolling shutter；20 步颜色和反射强度拟合 | `outputs/demo/report.json`、`outputs/demo/fit_history.json` | 小型合成案例中的接口、几何、梯度和编辑行为 |
| 官方 CUDA 算子 | 历史专项 44 passed；9 月 26 日逐束时间测试 2 passed，真实核验证 ±20 ms 对应 4.98/5.02 m 距离 | `reports/cuda_rasterizers.log`、`reports/tests_lidar_time_offsets_cuda.log` | RTX 3090 上实际执行固定版本扩展；两次独立记录不拼成一次全量测试 |
| 时间接口回归 | 26 passed，覆盖显式时间方向、原点、形状、有限值及参考渲染梯度 | `reports/tests_lidar_time_offsets_cpu.log` | `[A]`/`[E,A]` 采样时间与参考/官方适配器契约 |
| 历史测试快照 | 全量 141 passed、1 skipped；另有 GBK 输出和 NumPy checkpoint 回归 23 passed | `reports/tests_all.log`、`reports/training_regression.log` | 各自执行时的快照，不宣称是最新修改后的全量结果 |
| 最终完整测试 | **270 passed、1 skipped、2 warnings**，包含真实 CUDA 算子 | `reports/tests_release.log`、`reports/tests_release.xml` | skip 为仅适用于 CUDA 不可用环境的 guard；不是遗漏真实 GPU 测试。此前 206 项快照另行保留 |
| 数据完整性 | 744 文件，逐文件 ZIP CRC32/SHA256；480 JPEG、240 gzip、23 JSON 可读 | `data/pandaset/download_manifest_028.json`、`data/pandaset/integrity_028.json` | 028 文件完整、可解码；不等于运动去畸变对齐 |
| 修正后 parser 审计 | train split 240 图像、40 LiDAR、3 actor；数值检查通过；6 路各 20 帧共 120 overlay 已实际查看 contact sheets，并抽查原尺寸图 | `outputs/audit_corrected/audit.json`、`visual_review.md`、`alignment_review.json` | 无明显轴向、尺度和整体投影错位；近处侧后方仍有时间相关边缘偏差，不是像素级 RS 验收 |
| Actor 时间补丁 | 21 个 actor-frame 使用框内真实返回时间中位数，2 个使用校准相位 fallback，保留 low-support 标记 | `patches/pandaset-actor-time.json`、`outputs/audit_corrected/actor_timing_diagnostics.json` | 修正该序列已调查的坐标/相位问题，不能推广为所有数据或完整 deskew 验证 |
| 环境 | Python 3.10.6、PyTorch 2.7.1+cu118、RTX 3090；tiny-cuda-nn 前后向、官方训练和推理实际执行 | `outputs/environment_gpu.json`、`reports/tinycudann_cuda_probe.log`、各 run 日志 | 超出依赖导入检查的真实执行；未替换原全局/conda 环境 |
| 300 步 smoke | front + LiDAR、前 20%；exit 0、step 299、1,488,899 Gaussian、128 个有限张量 | `outputs/smoke_training_check.json`、`reports/SMOKE_RESULT.md` | 历史短程验收，actor_count=0，不代表完整动态序列 |
| 3000 步 intermediate | 6 cameras + LiDAR、全序列；exit 0、step 2999、5,000,000 Gaussian、134 个有限张量、3 actor | `outputs/intermediate_training_check.json`、`reports/INTERMEDIATE_RESULT.md` | 中间训练预算、checkpoint 完整性、已记录联合 loss 有限 |
| Intermediate 动态编辑 | train 样本 39/原始 front 帧 79；960×540 RGB、64×1800 LiDAR；actor 0 移除、1 m 平移、恢复；ego 平移 0.5 m | `outputs/validation_intermediate/validation.json`、`visual_review.md` | 输出有限、修改有局部响应、恢复误差 0；ego 相对外参最大误差 1.49e-8；不证明干净完整移除 |
| Intermediate 官方评估 | held-out 240 图像、40 LiDAR；PSNR 22.7863 dB、SSIM 0.67462、LPIPS 0.46463 | `outputs/eval_intermediate_official.json` | 该中间 checkpoint 和固定 split 的真实指标，非论文多序列结果 |
| 真实续训 | legacy→新格式→新格式，累计 3000→3002→3004；两次均 exit 0，各保留两份有限 checkpoint | `outputs/resume_validation.json` | 四次新增 CUDA 训练迭代和保存/加载链路，不证明逐位重放 |
| 30000 步 baseline | exit 0；最终 step 29999、5,000,000 Gaussian；step 28000/29999 两份 checkpoint 各 136 个有限张量 | `outputs/baseline_training_check.json`、`reports/BASELINE_RESULT.md` | 完整配置预算、模型/优化器完整性、已记录 loss 有限；本次视觉观察另存，不代表论文或部署质量 |
| Baseline 官方评估 | held-out 240 图像、40 LiDAR；PSNR 25.1963 dB、SSIM 0.78331、LPIPS 0.21774 | `outputs/eval_baseline_official.json` | 最终 step 29999 的单序列完整 held-out 指标，非论文多序列复现 |
| Baseline 原生编辑数值 | 1920×1080 RGB、64×1800 LiDAR；同一 front 时刻分别编辑全部 3 actor：移除、1 m 平移、15° yaw、恢复；恢复误差均 0，ego 相对标定误差 1.49e-8 | `outputs/validation_baseline_full_range/validation.json`、`reports/DYNAMIC_EDITING_RESULT.md` | Camera far=1e10 m、原生时间/RS 下的有限性及状态恢复；非所有帧/遮挡质量验收 |
| 20 时刻联合渲染 | eval split index 0,12,…,228，20 个 front 时刻的 RGB/LiDAR 输出均有限；960×540 和 64×1800；全部 RGB/overlay contact 面板与两个完整 overlay 已审阅 | `outputs/validation_baseline_sequence/validation.json`、`visual_review.md` | 粗几何和输出可用性检查；按选项跳过 actor/ego 编辑，不是逐像素对齐验收 |
| 原生 actor 视觉复核 | 同一 front 时刻的 actor 0/1/2 编辑诊断图均已实际查看；Camera 和 LiDAR 有局部响应 | `outputs/validation_baseline_full_range/visual_review.md` | actor 1/2 主体变化可见，actor 0 遮挡且弥散；残影/模糊仍在，不声称干净完整移除或未观测背景准确 |
| 远裁剪修复 | 同 step29999、front 训练帧79、960×540，仅 far200→1e10，PSNR 16.7093→23.8027 dB；已查看天空/远景恢复 | `outputs/far_clip_comparison.json` | 单训练视角的控制对照，不是 held-out 均值；公共 Camera 默认已修正 |
| 最终固定请求性能 | 5M Gaussian、RTX 3090；100 warmup/500 iterations；1920×1080 Camera 26.5046 FPS，64×1800 LiDAR 6.2653 FPS / 0.721761 MR/s | `outputs/profile_baseline_official.json` | 当前 CUDA stream events 与同步 wall time；分别测量两传感器，不是联合帧率、模型加载时延或论文速度复现 |
| 真实 CUDA 跟踪 | 同 checkpoint/原生请求，10 warmup/3 插桩迭代；记录 Camera/LiDAR 光栅化、fused projection、cuDNN convolution | `outputs/renderer_trace_baseline/manifest.json`、`operators.txt` | 执行路径与算子分析证据，有插桩开销，不是吞吐基准 |

父 ZIP 的 SHA256 是来源提供的预期摘要。未下载 44.5 GB 整包，`full_archive_sha256_verified=false`；不能写成完整父归档 SHA256 已验证。

## 审计、补丁与视觉结论

旧 `outputs/audit/alignment_review.json` 只覆盖 front、序列前 20%、最多 300 步。修正后的 `outputs/audit_corrected/alignment_review.json` 由 `Codex direct image review` 实际审阅，绑定 audit SHA256 `24f7865e8f3b865160553dffce6d484cab146c78646502ee25eb963d851883df`，覆盖 6 路、全序列、最多 30000 步实验训练。

`audit.json` 本身保留 `REVIEW_REQUIRED`/`NO-GO` 自动状态，数值检查和另行记录的视觉审阅分别保存。训练入口核对数值、SHA256 和 review 范围；实验训练审阅不等于重建质量或自动驾驶部署认证。nominal-pose projection 保留时间相关边缘偏差，未逐像素验收滚动快门或完整 deskew。

固定 NeuRAD 上应用显式 `pandaset-actor-time` 补丁。清单记录 base commit、文件摘要和 patch SHA256，各训练 run 的 launch 记录补丁应用状态及实际源码摘要。补丁优先取 Pandar64 `d=0` 原始框内点的 float64 时间中位数，无返回时使用 camera-derived LiDAR pose 和原生 -Y 前向相位，保留少点/fallback 诊断。Actor 关键帧时间不替代每条训练 ray 的真实时间。

公共 LiDAR `time_offsets[A]`/`[E,A]` 是相对请求时刻的显式秒数。PandaSet novel raster 使用 `atan2(-cos(azimuth), -sin(azimuth))*period/(2*pi)` 相位估计，周期来自完整原始 timestamps；不是逐束真实硬件测量。2026-09-30 公共接口已补齐 sensor-local `angular_velocity`（rad/s，默认零），Camera CV/GL 轴转换、LiDAR 本地轴传递均有契约测试，真实 CUDA 验证旋转的投影方向；参考后端明确拒绝非零角速度。旧请求保留，新请求另存 `outputs/validation_motion_requests/` 和 `outputs/baseline_requests_native_motion.json`。

Intermediate 的旧 far200 图像审阅见 `outputs/validation_intermediate/visual_review.md`，原始观察保留。后续在最终模型的同一训练视角完成控制对照，确认公共 Camera 的旧 far200 会裁掉远景；修改为官方 far1e10 后天空和远处建筑恢复，PSNR 从 16.7093 升至 23.8027 dB。这不是新一次训练，也不是 held-out 指标提升。公共 Camera 默认现为 1e10 m，LiDAR 默认仍为 200 m。

视觉审阅记录已完成，审阅日期为 2026-09-30、审阅者为 `Codex direct image review`。`outputs/validation_baseline_sequence/visual_review.md` 记录直接查看 `rgb_contact.jpg` 和 `overlay_contact.jpg` 各自全部 20 个面板，以及 sample_005、sample_019 的完整 overlay。范围为约 0.140–7.741 s 的 20 个 held-out front 时刻，不是六路各 20 帧，也不含这 20 帧的 actor 编辑。道路、建筑和车辆布局随视角前进连续，未观察到整帧轴反转或整体投影偏移；仍有纹理模糊、波纹及局部重建伪影，overlay 只支持粗尺度对应，不是逐像素 RS/deskew 对齐。

`outputs/validation_baseline_full_range/visual_review.md` 记录直接查看 sample_000/001/002 的 `actor_edit_diagnostics.png`，对应同一 front 训练时刻分别编辑 actor 0/1/2。Actor 1 移除后主体明显消失；actor 2 移除后显露后方内容，平移和旋转有可见变化，但遮挡边缘仍有残影。Actor 0 被树木/前景遮挡，可见部分弥散，RGB 改动较难辨认，不能把 LiDAR 范围响应解释为干净完整重建或移除。原生 baseline 仍有边缘、纹理波纹和过度平滑。自动 `validation.json` 继续保留 `visual_quality=UNREVIEWED`，与已完成的人工观察分开保存，不自动改写为整体质量 PASS。

## 训练阶段与历史记录

| 阶段 | Run | 当前状态 |
|---|---|---|
| 300 步 smoke | `outputs/training/smoke_300/splatad/2026-09-23_073446_729232Z/` | 训练检查 PASS；front/前 20%，仍在 500 步 warmup 内 |
| 3000 步 intermediate | `outputs/training/intermediate_3000/splatad/2026-09-26_051956_834730Z/` | 训练、动态编辑数值和官方评估完成；有已知视觉伪影 |
| 30000 步 baseline | `outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/` | 训练、官方评估、原生 3 actor 编辑、20 front 时刻渲染、固定请求性能及限定范围视觉复核完成；已知画质限制保留 |

Baseline 总 Train Loss 首末抽样为 0.320640（step 0）和 0.0194520（step 29990）；RGB、depth、intensity、ray-drop loss 均有有限记录。每 10 步的样本不证明每个未记录值有限，不同 batch 的首末值也不是同一图像的误差改善。Gaussian 数从 2,604,056 增到 5,000,000。周期 checkpoint 每 2000 步保存，保留最近两份；末尾另存 step 29999，两份均经实际读取检查。

历史失败 run 保持原样：`2026-09-23_023519_714812Z` 在 LPIPS 下载失败，无 checkpoint；`2026-09-23_023800_593525Z` 保存了 step 299，但 wrapper 的 GBK/emoji 输出报错，旧 completion 的 returncode 仍为 null。`outputs/validation_smoke/` 对应后者；后续 `outputs/validation_smoke_completed/`、`outputs/eval_smoke_official.json`、`outputs/profile_smoke_official.json` 对应成功 smoke。旧 `reports/SMOKE_RESULT.md` 是当时快照，其待办不覆盖此页新增证据。

Smoke 与 intermediate/baseline 的相机数、序列范围、动态内容和补丁状态不同，不能用 PSNR 差直接判断训练提升或退化。Smoke 同步 profiling 只对应 1,488,899 Gaussian、480×270 Camera 与 64×1800 LiDAR，不能套用到 500 万 Gaussian 模型。

## 评估、性能和扩展边界

- Intermediate 和 baseline 官方评估均使用 `get_average_eval_image_metrics`、seed 42、`train_split_fraction=0.5`、全序列 240 张图像/40 帧 LiDAR，未请求 FID。最终 baseline 指标来自已完成的 `outputs/eval_baseline_official.json`；同范围对比见 `reports/BASELINE_RESULT.md`，过程中的 step 27500 评估不替代最终 step 29999。
- Intermediate `depth_median_l2=0.00877150` 是 range 残差平方中位数（m²），不是米或 RMSE；`intensity_rmse=0.0633371`，`ray_drop_accuracy=0.981564`。Accuracy 需结合类别分布理解。
- 官方 `chamfer_distance=0.554674` 使用双向平方距离总和除以 GT 点数，区别于 NPZ evaluator 的双向均值之和，不跨口径直接比较。
- Baseline 对应值为 `depth_median_l2=0.00161220 m²`、`intensity_rmse=0.0565821`、`ray_drop_accuracy=0.982593`、官方 `chamfer_distance=0.391052`，采用相同口径。
- 官方评估的 FPS/rays-per-second 是过程信息。最终独立 profiling 在 100 次 warmup 后测量 500 次：Camera 均值/P95 为 37.7292/39.1639 ms，同步 wall 均值 37.8656 ms；LiDAR 为 159.6097/163.6575 ms，wall 159.7345 ms。仅适用于记录的 checkpoint 和固定原生请求，不作联合帧率或论文速度结论。
- Profiling 显存仅覆盖进程内 PyTorch CUDA allocator，排除非 PyTorch 分配，不是整卡或单传感器独占显存：Camera peak allocated 4,653,307,392 bytes、相对测量前增量 2,081,592,320 bytes；LiDAR 对应 3,813,043,200 / 1,277,529,600 bytes。Reserved 缓存池另计，完整原值见 `reports/BASELINE_RESULT.md`，不与 allocated 相加。
- 10 warmup/3 iterations 的算子 trace 仅核查执行路径；嵌套/不同层级事件和插桩计时不用于吞吐量，也不替代 500 次同步 profiling。静态推理剪枝 A/B 已完成，见 [剪枝结果](../reports/PRUNING_RESULT.md)；没有论文多序列速度复现。
- `pruning.py` 仅返回保护 LiDAR support 和 actor 最小保留数的选择 mask，尚未接入官方 MCMC 参数/优化器迁移；没有质量提升或显存节省的实验结论。
- ReferenceBackend 没有官方 CNN/MLP、MCMC 或完整曝光积分，`ray_drop=1-alpha` 仅为几何参考。

## 可复查入口

```powershell
.\.venv-gpu\Scripts\python.exe scripts/check_training_run.py outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z --output outputs/baseline_training_check.json
```

每次 run 保存独立 launch、UTF-8 训练日志、completion、官方 config、TensorBoard 和 checkpoint。检查器通过受限 `torch.load(weights_only=True)` 读取已知格式，只为必要 NumPy scalar 建立有限 allowlist，核验步数、形状、有限值、日志及退出状态；文件存在本身不足以判定成功。

固定基础版本为 NeuRAD `8ba9b5116b8a2822a80c64f63a4ae64c5871aa68` 与 rasterizer `6e31ad766d39e0c33f9034a2ed772d51364b2343`，额外补丁单独记录。上游默认 30001 次（0..30000），本项目明确 30000 次（0..29999）；最终独立评估与默认 step 30000 的计划评估行为需要区分，wrapper 不隐式改写实验预算。
