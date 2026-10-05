# PandaSet 028 官方 SplatAD 30000 步 baseline

**完整 30000 次真实联合训练和 checkpoint 数值检查 PASS，最终官方 held-out 评估已完成。** 训练进程正常退出，最终 step 29999 模型含 500 万个 Gaussian，保留的两个 checkpoint 分别通过 136 个模型及优化器张量的有限值检查。原生 1920×1080 的 3 actor 编辑和 20 个 held-out front 时刻联合渲染也已通过数值检查；截至 2026-09-28，固定原生请求性能及上述样本的视觉复核均已完成，已知伪影和范围限制保留，不声称论文复现。

## 可复查的运行

| 项目 | 实际值 |
|---|---|
| Run | `outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/` |
| 检查结果 | `outputs/baseline_training_check.json`: `status=PASS`、`blockers=[]` |
| 配置 | `configs/baseline.yaml`；完整 argv/实际官方 config 在 run 的 `launch.json`/`config.yml` |
| 数据 | 真实 PandaSet 028，全序列，front/front_left/front_right/back/left/right + Pandar64 |
| Seed / 预算 | 42 / 30000 次，零起始 step 0..29999；本 run 从头训练 |
| 退出记录 | `completion.json`: `returncode=0`、`status=exited_zero` |
| 完成时间 | 2026-09-27 07:17:43 UTC / 上海时间 15:17:43 |
| 环境 | Windows、Python 3.10.6、PyTorch 2.7.1+cu118、RTX 3090 |
| NeuRAD 基础 commit | `8ba9b5116b8a2822a80c64f63a4ae64c5871aa68` |
| Rasterizer commit | `6e31ad766d39e0c33f9034a2ed772d51364b2343` |
| Actor 时间补丁 | `patches/pandaset-actor-time.json`；launch 记录已应用 |
| Patch SHA256 | `6cbaacf33bd4dee8e73b653641fd68a5c888cabb2ebcf1c6e1d588c5d5d8955b` |
| Audit / review | `outputs/audit_corrected/`；6 路各 20 帧审阅、全序列实验范围，SHA256 绑定 |
| Checkpoint 周期 | 每 2000 步保存，保留最近两个；结束时保存最终 step 29999 |

Baseline 通过 `scripts/train_official.py` 启动官方模型训练，并记录 checkpoint wrapper 的源码摘要。它没有使用 resume probe 的参数作为初始模型。审计、actor 时间补丁及 3000 步中间模型的证据见 [INTERMEDIATE_RESULT.md](INTERMEDIATE_RESULT.md)，中间模型指标不作为本模型最终指标。

## Checkpoint 完整性与恢复验证

| 保存文件 | 内部 step | 完成迭代数 | Gaussian 数 | 有限张量数 |
|---|---:|---:|---:|---:|
| `nerfstudio_models/step-000028000.ckpt` | 28000 | 28001 | 5,000,000 | 136 |
| `nerfstudio_models/step-000029999.ckpt` | 29999 | 30000 | 5,000,000 | 136 |

检查器实际读取两份 checkpoint，核验步数、参数形状、模型及优化器有限性、运行退出记录和训练日志，而非仅检查文件存在。两份模型均通过，最终步数满足配置预算。

另一个独立实验 `outputs/resume_validation.json` 为 PASS：从 legacy intermediate checkpoint 恢复到累计 3002 次，再从新保存格式恢复到 3004 次；两次均 exit 0，每个 run 保留两份原子保存的 checkpoint，检查张量均有限。这证明真实 CUDA 下四次新增迭代的保存/恢复链路；不证明 GPU 逐位重放，也不是 baseline 的质量评估。

## 已记录的训练量

| TensorBoard tag | 首记录 step / value | 末记录 step / value | 记录数 |
|---|---:|---:|---:|
| `Train Loss` | 0 / 0.320640 | 29990 / 0.0194520 | 3000 |
| `main_loss` | 0 / 0.317740 | 29980 / 0.0281728 | 2588 |
| `depth_loss` | 30 / 0.799416 | 29990 / 0.00864535 | 412 |
| `intensity_loss` | 30 / 0.285948 | 29990 / 0.00134294 | 412 |
| `ray_drop_loss` | 30 / 0.100906 | 29990 / 0.00527964 | 412 |
| `alpha_sum_until_points_loss` | 30 / 0.0382736 | 29990 / 0.00289058 | 412 |
| `gaussian_count` | 0 / 2,604,056 | 29980 / 5,000,000 | 2588 |

以上抽样记录均有限。总 loss 每 10 步记录一次，Camera/LiDAR batch 不同，首末训练量不等于同一观测的 A/B 改善，也不能证明所有未记录中间值有限。训练期间的部分图像/全图评估曲线保存在 TensorBoard 及 `outputs/baseline_curves/`，不得把 step 27500 的过程评估代替最终 step 29999 held-out 指标。

`Train Total (time)=6100.25 s` 是官方训练循环计时；最后记录的 `GPU Memory (MB)=9849.335` 实际单位是 MiB，为 PyTorch 累计峰值分配量。这不是端到端 wall time、整卡占用或稳定渲染吞吐率。

## 最终官方 held-out 评估

`outputs/eval_baseline_official.json` 已完成，checkpoint 为 step 29999，调用官方 `get_average_eval_image_metrics`。与 3000 步报告一致：sequence 028、全序列、`train_split_fraction=0.5`、seed 42，held-out **240 Camera 图像/40 LiDAR sweeps**，未请求 FID。以下是相同评估范围的两次独立训练结果；不混入范围更小的 smoke，也不把训练过程中的 step 27500 指标代替最终评估。

| 指标 | Intermediate 3000 步 | Baseline 30000 步 | Baseline 标准差 |
|---|---:|---:|---:|
| Camera PSNR (dB) ↑ | 22.786295 | **25.196257** | 2.061511 |
| Camera SSIM ↑ | 0.674620 | **0.783307** | 0.052894 |
| Camera LPIPS ↓ | 0.464629 | **0.217737** | 0.042737 |
| LiDAR intensity RMSE ↓ | 0.0633371 | **0.0565821** | 0.00323933 |
| LiDAR ray-drop accuracy ↑ | 0.981564 | **0.982593** | 0.00496932 |
| LiDAR `depth_median_l2` (m²) ↓ | 0.00877150 | **0.00161220** | 0.000514795 |
| LiDAR `depth_mean_rel_l2` ↓ | 0.0130569 | **0.00903961** | 0.00598676 |
| LiDAR 官方 `chamfer_distance` ↓ | 0.554674 | **0.391052** | 0.107099 |

Baseline 在这些同范围指标上较好；两者是独立 run，不宣称严格控制所有变量的 A/B 或论文复现。`depth_median_l2` 是 range 残差平方的中位数（m²），不是米或 RMSE。官方 Chamfer 为双向平方距离总和除以 GT 点数，与 NPZ evaluator 的双向均值之和不同。ray-drop accuracy 需要结合类别分布理解。文件中的 FPS/rays-per-second 仅是评估过程计时，不作为专用稳定性能结论。

## 最终编辑、视觉与性能的当前状态

最终原生验证见 `outputs/validation_baseline_full_range/validation.json`：**1920×1080 RGB + 64×1800 LiDAR**、Camera far=1e10 m、原生时刻及 rolling shutter；同一 front 样本 index=39 分别对 **3 个 actor** 执行移除、**1 m 平移、15° yaw 旋转和恢复**。所有检查输出有限，恢复误差全部 **0.0**、参数恢复检查通过；ego 共同平移 0.5 m 后，相对外参最大误差 **1.4901161e-8**。详细数值和边界见 [DYNAMIC_EDITING_RESULT.md](DYNAMIC_EDITING_RESULT.md)。旧 far200 验证目录原样保留。

远裁剪控制对照已经完成，见 `outputs/far_clip_comparison.json`：同 step29999 模型、同 front 训练帧79、960×540，仅 Camera far200→1e10，PSNR **16.7093→23.8027 dB**。主任务直接查看后确认天空和远处建筑恢复，因此初始天空问题不能全部归因于训练未收敛。该单训练视角对照与本报告的 240 张 held-out 评估不同，不能混为同一指标。公共 `CameraRequest.far` 默认已改为 1e10 m；LiDAR 保持 200 m。原生请求为 `outputs/baseline_requests_native.json`，960×540 请求另存 `outputs/baseline_requests_full_range.json`。

`outputs/validation_baseline_sequence/validation.json` 已完成 **20 个 held-out front 时刻**的联合渲染，RGB 为 960×540、LiDAR 为 64×1800，检查输出全部有限。该轮按选项跳过 actor 和 ego 编辑，不能算作 20 帧动态编辑验证。已完成的 `outputs/validation_baseline_sequence/visual_review.md` 记录审阅全部 20 个 RGB contact 面板、全部 20 个 overlay contact 面板，以及 sample_005、sample_019 的完整 overlay；约 0.140–7.741 s 的场景布局连续，无明显整帧轴向反转或整体投影偏移。车辆、树叶和建筑边缘仍有纹理模糊、波纹和局部伪影。RGB 与 LiDAR 在道路、车辆、立面上的粗尺度对应可见，但不代表逐像素对齐。

原生编辑视觉复核记录为 `outputs/validation_baseline_full_range/visual_review.md`：sample_000/001/002 的全部 3 张 actor 诊断图已查看，它们是同一 front 时刻分别编辑 actor 0/1/2。Actor 1 移除后车辆主体明显消失；actor 2 移除显露后方内容，平移/旋转使车体和轮廓发生变化，边缘仍有残影。Actor 0 被树木与前景遮挡、可见部分弥散，RGB 改动较难辨认，不能仅凭 LiDAR 响应声称干净完整移除。原生 baseline 仍有纹理波纹、边缘失真和过度平滑。两份复核的审阅者均为 `Codex direct image review`，日期为 2026-09-28；原始自动 `UNREVIEWED` 状态保持不变，有限性与人工观察分别保存。

2026-09-30 最新完整测试为 **270 passed、1 skipped、2 warnings**（`reports/tests_release.log`/`.xml`），其中包含实际 CUDA；skip 是用于 CUDA 不可用环境的 guard，在当前 GPU 环境下不适用。此前 206 项测试日志仍单独保留。

## 固定原生请求的同步性能

证据为 `outputs/profile_baseline_official.json`：官方 step29999、**5,000,000 Gaussian**、RTX 3090，固定 `outputs/baseline_requests_native.json`。Camera 为 **1920×1080**、far=1e10 m、readout 约 0.03 s；LiDAR 为 **64×1800=115200 rays**、far=200 m、scan period 0.100007800 s，使用请求中的显式逐束时间。每个传感器分别 **100 次 warmup、500 次测量**；GPU 时间使用当前 CUDA stream 的 events，并另报同步 wall time。

| 固定请求 | GPU 均值 | GPU 中位数 | GPU P95 | 同步 wall 均值 | FPS | 吞吐量 |
|---|---:|---:|---:|---:|---:|---:|
| Camera 1920×1080 | 37.7292 ms | 37.5634 ms | 39.1639 ms | 37.8656 ms | **26.5046** | **54.9600 MP/s** |
| LiDAR 64×1800 | 159.6097 ms | 160.5448 ms | 163.6575 ms | 159.7345 ms | **6.2653** | **0.721761 MR/s** |

这些是固定 checkpoint、固定视角及请求下的单传感器渲染测量，不能相加或等同于同时输出 Camera/LiDAR 的联合帧率；也不能作为包含离线训练、模型加载或文件导出的端到端时延。不能推广到全部场景、任意分辨率或论文硬件/设置。

显存字段的范围严格为 **该进程的 PyTorch CUDA allocator**，不含非 PyTorch 分配，不能解读为整张 GPU 占用或该传感器独占显存。原始 byte 值如下；incremental peak 是 peak allocated 减去测量前 allocated，并非另一次隔离模型的测量。Reserved 包含缓存池，不能与 allocated 相加。

| 显存字段（bytes） | Camera | LiDAR |
|---|---:|---:|
| 测量前 allocated | 2,571,715,072 | 2,535,513,600 |
| peak allocated | 4,653,307,392 | 3,813,043,200 |
| incremental peak allocated | 2,081,592,320 | 1,277,529,600 |
| peak reserved | 6,595,543,040 | 6,599,737,344 |

## 算子跟踪的范围

`outputs/renderer_trace_baseline/manifest.json` 和 `operators.txt` 对同 step29999、同原生请求进行 **10 次 warmup、3 次插桩迭代**，记录真实 CUDA 操作。可见 `_RasterizeToPoints`、`_RasterizeToPixels`、LiDAR/Camera fused projection，以及 `aten::cudnn_convolution`。例如 `_RasterizeToPoints` 3 次累计 CUDA total 为 424.343 ms，`_RasterizeToPixels` 3 次为 9.753 ms，cuDNN convolution 18 次为 34.910 ms。

这份 trace 用来核查执行路径和定位待分析的算子；它有插桩开销，包含不同层级的事件，不相加为端到端时延，不用于推导吞吐量。性能结论仅引用上面的 500 次同步测量。后续静态低透明度推理剪枝已完成独立质量与速度对照，见 [剪枝报告](PRUNING_RESULT.md)；没有多序列论文复现结论。

复查本次训练：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/check_training_run.py outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z --output outputs/baseline_training_check.json
```
