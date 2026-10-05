# SplatAD-Drive 真实 baseline 性能与优化对照

**固定原生请求的真实速度为 Camera 26.5046 FPS、LiDAR 6.2653 FPS / 0.721761 MR/s。** 无编辑路径优化的独立 ABBA 对照只得到约 **1.4% Camera、1.2% LiDAR** 的小幅吞吐改善，已比较公共输出的最大绝对差异全部为 0。插桩跟踪显示 LiDAR 点光栅化核平均约 **141.4 ms**，是该请求下主要耗时所在；这次接口开销优化没有消除该瓶颈。

本文更新至 2026-09-30。9 月 28 日已完成剪枝推理模型的独立性能测量：Camera **27.4769 FPS**、LiDAR **7.94205 FPS / 0.914925 MR/s**，相对记录中的完整 baseline 为 **1.03668× / 1.26763×**。这是同机器、同固定请求、不同进程的观测，未做多轮交错复测；质量验证另见 [剪枝报告](PRUNING_RESULT.md)，不能据此推广为任意视角无损或普适加速。

## 测量证据的分工

| 证据 | 条件 | 用途 |
|---|---|---|
| `outputs/profile_baseline_official.json` | 100 warmup、500 次测量 | 固定原生请求下的同步性能及分配器显存 |
| `outputs/renderer_trace_baseline/manifest.json`、`operators.txt` | 10 warmup、3 次插桩迭代 | 真实 CUDA 执行路径和算子耗时定位，不是吞吐基准 |
| `outputs/edit_overhead_benchmark.json` | 每个 round 25 warmup、100 次测量；legacy→optimized→optimized→legacy | 同模型、同请求、无编辑条件下的适配器路径对照及输出一致性 |
| `outputs/profile_pruned_official.json` | 100 warmup、500 次测量；与完整模型单独运行 | 3,672,621 Gaussian 推理候选的固定请求性能；质量 A/B 尚未完成 |

前三项均指向官方 baseline `step-000029999.ckpt`（`outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/`）、**5,000,000 Gaussian**；第四项使用从该 checkpoint 导出的推理候选。硬件均为 RTX 3090，请求均为 `outputs/baseline_requests_native.json`，两个 500 次 profile 保存的 `request_values` 已逐字段核对相同。Camera 为 **1920×1080**、far=1e10 m、readout 约 0.03 s；LiDAR 为 **64×1800=115200 rays**、far=200 m、scan period 0.100007800 s，使用显式逐束时间。

这是一个固定视角的测量，不是全数据集性能分布，不代表训练/加载/文件导出的端到端时延，也不构成多序列论文速度复现。

## 500 次同步测量

GPU 时间来自当前 CUDA stream events；报告另存同步 wall time。两传感器分别测量，不能将单传感器 FPS 相加，也不能直接称为同时输出 Camera/LiDAR 的联合帧率。

| 请求 | GPU 均值 | 中位数 | P95 | 同步 wall 均值 | FPS | 吞吐量 |
|---|---:|---:|---:|---:|---:|---:|
| Camera 1920×1080 | 37.7292 ms | 37.5634 ms | 39.1639 ms | 37.8656 ms | 26.5046 | 54.9600 MP/s |
| LiDAR 64×1800 | 159.6097 ms | 160.5448 ms | 163.6575 ms | 159.7345 ms | 6.2653 | 0.721761 MR/s |

显存范围是**进程内 PyTorch CUDA allocator**，排除非 PyTorch 分配，不是整张显卡占用或单传感器独占显存。Incremental peak 是 peak allocated 减去测量前 allocated；reserved 包含缓存池，不能与 allocated 相加。

| 原始显存字段（bytes） | Camera | LiDAR |
|---|---:|---:|
| 测量前 allocated | 2,571,715,072 | 2,535,513,600 |
| peak allocated | 4,653,307,392 | 3,813,043,200 |
| incremental peak allocated | 2,081,592,320 | 1,277,529,600 |
| peak reserved | 6,595,543,040 | 6,599,737,344 |

## Trace 定位的实际耗时

在 3 次插桩迭代中，`rasterize_to_points_fwd_kernel` 执行 3 次，累计 CUDA 时间 **424.286 ms**，平均 **141.429 ms**。对应 Python/autograd 包装行 `_RasterizeToPoints` 的 CUDA total 为 **424.343 ms**，平均 **141.448 ms**。二者是同一调用链的不同层级，**不能相加**。

| 跟踪条目 | 调用数 | 累计 CUDA 时间 | 含义 |
|---|---:|---:|---|
| LiDAR `rasterize_to_points_fwd_kernel` | 3 | 424.286 ms | 实际点光栅化核，平均 141.429 ms |
| Camera `rasterize_to_pixels_fwd_kernel` | 3 | 9.753 ms | 实际像素光栅化核，平均 3.251 ms |
| `aten::cudnn_convolution` | 18 | total 34.910 ms | Camera 学习型解码路径中的卷积 |
| `_FullyFusedLidarProjection` | 3 | total 2.012 ms | LiDAR 投影包装行 |
| `_FullyFusedProjection` | 3 | total 0.878 ms | Camera 投影包装行 |

`lidar_complete_render` / `camera_complete_render` 父级 annotation、包装行和对应 kernel 行可能覆盖相同执行，不能求和得到端到端总耗时，百分比也不能跨这些层级直接相加。Trace 的插桩开销和仅 3 次迭代使它不适合推导 FPS；上面的 500 次同步结果才是吞吐记录。

在此视角与模型下，LiDAR 点光栅化是明确的主要分析对象。Camera 的完整开销还包含学习型解码和张量操作，不能只用 3.251 ms 的光栅核声称整个 Camera 可达相应帧率。这些观察不证明核已达到硬件极限，也不预设剪枝或改写 CUDA 能获得多少收益。

## 无编辑路径的 ABBA 对照

优化针对普通推理：没有 actor pose 编辑且没有隐藏 actor 时，直接进入渲染，省去扫描 Gaussian id、构建编辑 mask、空参数备份及临时方法替换。存在实际编辑时仍走编辑/恢复路径；本次性能对照不测带编辑请求，也没有修改训练或 Gaussian 数量。

旧代码快照为 `outputs/profiling_sources/neurad_before_noop.py`，SHA256 `4a063347df23a3a1db3384d813d9fcef3c316a9a6b96bd36ed90e8852b19590a`；对照记录的优化版 SHA256 为 `6bbf63dcf41fabded38865166c7c6c56d18ef1d42449a7e8fbdf3724da6df6e6`。每个传感器顺序均为 **legacy、optimized、optimized、legacy**；每个 round 单独 25 次 warmup、100 次测量。ABBA 可减轻单向时间漂移，但不是无限重复或统计显著性证明。

| 请求 | Legacy round 1 | Optimized round 2 | Optimized round 3 | Legacy round 4 |
|---|---:|---:|---:|---:|
| Camera GPU 均值 | 38.0098 ms | 37.7225 ms | 37.6048 ms | 38.3671 ms |
| LiDAR GPU 均值 | 159.3034 ms | 158.5025 ms | 158.3934 ms | 161.4790 ms |

| 请求 | 两轮 legacy 均值 | 两轮 optimized 均值 | Legacy / optimized | 吞吐改善 |
|---|---:|---:|---:|---:|
| Camera | 38.1884 ms | 37.6636 ms | 1.013934× | **约 1.39%** |
| LiDAR | 160.3912 ms | 158.4480 ms | 1.012264× | **约 1.23%** |

这是小幅适配器开销优化，不是大幅加速；尤其没有改变约 141.4 ms 的 LiDAR 光栅核所揭示的主要耗时。不同轮次存在波动，未提供多次独立 ABBA 的置信区间。不得把独立 500 次报告的均值与这里某一个 round 随意配成更大的收益。

同请求输出一致性已实际比较：Camera 的 `rgb/depth/alpha`，LiDAR 的 `range/intensity/hit_probability/ray_drop/alpha/points/depth_sum/median_range`，**每个公共输出的最大绝对差异均为 0.0**。这证明此模型和该无编辑请求的输出一致，不替代全部状态/全部视角的行为测试。

## 剪枝推理模型：固定请求性能

候选清单为 `outputs/pruned_baseline_alpha001/manifest.json`，模式为 **inference-only static opacity filter**：仅移除 static id=3 且 `sigmoid(opacity)<0.001` 的点，5000000→3672621，移除 1327379（约 26.55%）。Actor 0/1/2 的 65/196/72 个 Gaussian 全部保留，源 checkpoint 与 config 未改变。导出候选不可用于恢复训练；它省略了优化器等状态，所以文件体积下降也不能全部归因于 Gaussian 剪枝。

`outputs/profile_pruned_official.json` 和 `reports/profile_pruned_official.log` 已生成。候选与完整模型均在同一 RTX 3090、相同原生请求、当前 CUDA stream events 计时下，每传感器执行 100 次 warmup、500 次测量，另记录同步 wall time。

| 请求 | Baseline GPU 均值 | 候选 GPU 均值 / P95 | 候选同步 wall 均值 | 候选 FPS | 候选吞吐量 | Baseline/候选均值 |
|---|---:|---:|---:|---:|---:|---:|
| Camera 1920×1080 | 37.7292 ms | 36.3942 / 38.8143 ms | 36.5367 ms | **27.4769** | 56.9762 MP/s | **1.03668×** |
| LiDAR 64×1800 | 159.6097 ms | 125.9120 / 131.8049 ms | 126.0427 ms | **7.94205** | **0.914925 MR/s** | **1.26763×** |

这里的比值是两份独立进程测量的均值之比，约对应 Camera 3.67%、LiDAR 26.76% 的吞吐增加，**不是剪枝 ABBA 或多轮交错测量**。硬件状态、运行顺序及分配器状态等差异未由独立交错重复消除，因此只报告本次固定条件的观测，不保证所有视角/场景有同样收益，也不将上一节无编辑路径的 1.4%/1.2% 收益归给剪枝。

候选显存仍仅覆盖**进程内 PyTorch CUDA allocator**，排除非 PyTorch 分配，不能解释为整卡或单传感器独占显存：

| 候选显存字段（bytes） | Camera | LiDAR |
|---|---:|---:|
| 测量前 allocated | 2,355,551,232 | 2,289,403,904 |
| peak allocated | **4,317,131,264** | **3,225,517,056** |
| incremental peak allocated | 1,961,580,032 | 936,113,152 |
| peak reserved | 6,452,936,704 | 6,457,131,008 |

相较上述完整模型记录，peak allocated 分别少 336,176,128 / 587,526,144 bytes。这是不同进程下的同口径 allocator 观测；incremental peak 为峰值减测量前 allocated，reserved 不与 allocated 相加。

剪枝模型后续已完成相同 240 图像/40 LiDAR 官方评估，主要质量指标相同；两批 20+2 请求（分别零传感器角速度与实际角速度）各 44 组公共输出精确一致，见 [独立质量证据](PRUNING_RESULT.md)。上文无编辑路径的 max-abs=0 和剪枝实验属于不同对照。性能表使用的旧请求角速度为零，不能称为新 motion 请求的测量。静态低 opacity 过滤不同于原方案的 actor-aware/LiDAR-supported 训练剪枝，后者尚未实现真实支持度累计和优化器状态迁移。

## 复现固定请求性能

Nsight Compute 硬件计数器尚未取得：2026-09-30 最小 CUDA 探针已用 `--target-processes all` 正确连接 venv 的实际子进程，但驱动返回 `ERR_NVGPUCTRPERM`，没有任何 kernel counter 结果，见 `reports/ncu_access_probe_child.log`。首次未跟踪子进程的日志也保留，不能把其 exit 0 当作 profiling 成功。未改变驱动权限或全局时钟；不能填造 occupancy、DRAM/L2 throughput 或 warp-stall 数值。现有 CUDA event 吞吐与 PyTorch CUDA trace 是独立且已成功的证据。

在 `C:\codex` 分别执行下面两条命令；它们使用已有模型与请求，各自启动独立进程，结果写入新的 `_rerun.json`，保留已报告原始文件。执行时固定同一 GPU 并避免其他 GPU 工作；即使如此，两次顺序运行仍不等于多轮交错 A/B。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 profile --load-config outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/config.yml --requests outputs/baseline_requests_native.json --device cuda --warmup 100 --iterations 500 --output outputs/profile_baseline_official_rerun.json
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 profile --load-config outputs/pruned_baseline_alpha001/config.yml --requests outputs/baseline_requests_native.json --device cuda --warmup 100 --iterations 500 --output outputs/profile_pruned_official_rerun.json
```
