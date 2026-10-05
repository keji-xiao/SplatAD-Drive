# 静态低透明度推理剪枝对照

更新日期：2026-09-30。PandaSet 028 的 step 29999 模型从 **5,000,000** 个 Gaussian 降到 **3,672,621**，减少 **26.54758%**。相同 split/seed 的完整官方评估中，下列质量指标没有变化；固定原生请求的 LiDAR 吞吐提高约 **26.8%**。这是该模型上的离线推理优化，不能称为原方案的训练期 LiDAR-supported pruning。

## 策略及来源

`scripts/prune_inference_checkpoint.py` 只删除静态 ID=3 且 `sigmoid(opacity)<0.001` 的 Gaussian，严格保留原行顺序，并以同一 mask 筛选全部 Gaussian 参数。三个 actor 的 65/196/72 个 Gaussian、轨迹和解码器全部保留。

依据是固定 CUDA 源码在 Camera/LiDAR 累积特征、深度和透射率前跳过 `alpha<1/255`（`third_party/splatad/gsplat/cuda/csrc/rasterization.cu`）。在有限、有效的 PSD 投影协方差和非负抗锯齿正则条件下，补偿系数不大于 1，有效 alpha 不超过基础 opacity。实现没有强制把补偿上界 clamp 到 1，因此不能把它说成任意病态输入或任意新视角下无条件无损；使用更保守的 0.001，并实际比较输出。

导出清单：`outputs/pruned_baseline_alpha001/manifest.json`。原模型 SHA256 为 `f1ae69b3866746644283cec254f5258174b7f7599b5a70c0d993fd9cd300bc9c`，推理模型为 `bd98062ed081ea33a98af3927b26b7a378d80956061adbf42533c64e3f4ab710`。原检查点与配置未改变，导出后逐张量重读完全一致。

推理检查点标为 `inference_only`，训练恢复入口会拒绝加载它进行续训。文件从 1,650,431,993 bytes 变为 421,406,721 bytes，但同时移除了 Adam 等训练状态，**不能把全部文件体积下降归因于剪枝**。

## 完整官方 held-out 评估

同 sequence 028、完整 clip、train split=0.5、seed=42、step=29999，240 张 Camera 图像和 40 帧 LiDAR；未请求 FID。数据来源为 `outputs/eval_baseline_official.json` 与 `outputs/eval_pruned_official.json`。原生官方流程包含自身运动元数据，独立于公共请求文件。

| 指标 | Baseline | 剪枝后 |
|---|---:|---:|
| PSNR (dB) | 25.1962566376 | 25.1962566376 |
| SSIM | 0.7833068967 | 0.7833068967 |
| LPIPS | 0.2177369446 | 0.2177369446 |
| intensity RMSE | 0.0565821379 | 0.0565821379 |
| ray-drop accuracy | 0.9825926423 | 0.9825926423 |
| depth_median_l2 (m²) | 0.001612198888 | 0.001612198888 |
| depth_mean_rel_l2 | 0.009039614350 | 0.009039614350 |
| 官方 Chamfer | 0.3910522461 | 0.3910522461 |

`depth_median_l2` 为平方距离中位数，非米或 RMSE。官方 Chamfer 为双向平方距离总和除以 GT 点数，与本项目 NPZ evaluator 的双向均值之和不同。以上有限样本结果不能推广为所有模型或未测试视角的保证。

## 公共输出逐元素比较

`outputs/pruning_comparison/public_outputs.json`：20 个 held-out front 时刻的 native readout 请求，加首尾两个零 RS 请求，共 **44/44** 组传感器对照通过，所有比较张量**逐元素完全相等**。包括 Camera RGB/depth/alpha，以及 LiDAR range/intensity/hit-probability/ray-drop/alpha/points/depth_sum/median_range。该批旧请求的传感器角速度为零，保留线速度和逐束时间；新增角速度批次独立记录，不覆盖旧证据。

比较脚本一次只加载一个模型，CPU 保存基准输出，校验实际加载路径、checkpoint/config/request SHA、形状、dtype、有限性和全部公共通道。设定容差为 atol=rtol=1e-5，但上述结果比容差内一致更严格，为实测精确一致；不会以放宽容差掩盖失败。

角速度补齐后的第二批 `outputs/pruning_comparison/public_outputs_motion.json` 也已完成 **44/44 PASS，全部逐元素精确一致**。请求来自同 20 帧的原生角速度元数据，Camera GL→CV 转换，LiDAR 保持本地轴；最大角速度范数分别 0.053676 和 0.053897 rad/s。来源、原字段保留和 SHA 记录在 `outputs/validation_motion_requests/manifest.json`。这两批不能称为 40 个不同时间点。

## 固定请求性能

RTX 3090，Camera 1920×1080、LiDAR 64×1800，两个模型使用逐字段一致的 `outputs/baseline_requests_native.json`。分别启动独立进程，各 100 次 warmup、500 次 CUDA event 同步测量。这里是旧角速度为零请求，不是全部序列的平均速度。文件为 `outputs/profile_baseline_official.json` 与 `outputs/profile_pruned_official.json`。

| 项目 | Baseline | 剪枝后 |
|---|---:|---:|
| Camera 均值 (ms) | 37.7292 | 36.3942 |
| Camera FPS | 26.5046 | 27.4769 |
| LiDAR 均值 (ms) | 159.6097 | 125.9120 |
| LiDAR FPS | 6.2653 | 7.9421 |
| LiDAR MR/s | 0.721761 | 0.914925 |
| Camera peak allocated (bytes) | 4,653,307,392 | 4,317,131,264 |
| LiDAR peak allocated (bytes) | 3,813,043,200 | 3,225,517,056 |

Camera 吞吐比 1.03668×，LiDAR 1.26763×。显存为进程内 PyTorch allocator 峰值，不是整卡或孤立模型显存。测量不是多轮交错实验，不保证其它视角和硬件获得相同比例。完整 wall time/P95、trace 和无编辑路径 ABBA 对照见 [性能报告](PERFORMANCE_RESULT.md)。无编辑路径约 1% 的开销优化与本项剪枝分别记录，不能相加冒充一个实验。

## 使用与研究边界

`outputs/validation_pruned_motion/validation.json` 已完成原生分辨率下三 actor 的移除、1 m 平移、15° yaw、恢复以及 ego 共同平移 0.5 m。所有检查输出有限，三次参数恢复均成功、公开输出恢复最大误差均为 **0**，相对外参最大误差 **1.49e-8**。三个编辑诊断图已实际查看，详见 [视觉记录](../outputs/validation_pruned_motion/visual_review.md)。Actor 0 遮挡和外观弥散、actor 1/2 局部残影仍是质量限制，数值通过不意味着遮挡背景重建完美。

在项目根目录使用已有推理模型：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 render --load-config outputs/pruned_baseline_alpha001/config.yml --requests outputs/baseline_requests_native_motion.json --device cuda --output outputs/render_pruned
```

新请求带传感器本地角速度；参考后端会明确拒绝非零角速度，官方后端传给上游一阶 RS 核。该渲染命令与上表旧请求的性能测量范围不同。

原方案的 visibility/GT-hit 计数并没有由投影半径自动得到。`pruning.py` 中的 support-aware mask 仍是外部统计驱动的研究候选选择器，未实现真实贡献累计或 MCMC 优化器状态迁移。本次选择可验证的静态低 opacity 推理过滤，既保留已有训练模型，也避免把尚未实现的训练算法写成实验成果。
