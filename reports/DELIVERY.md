# SplatAD-Drive 本机交付与进度

更新：2026-09-30。项目目录 `C:\codex`。**本机可运行的实验系统已打通；按包含研究扩展的原方案估算，整体进度约 95%。** 该百分比是工程里程碑估算，不是测试覆盖率或训练比例；实际训练为 30000/30000 步。

| 里程碑 | 权重/已完成 | 证据 |
|---|---:|---|
| 环境与实际 CUDA | 10/10 | `.venv-gpu`、真实 CUDA 前后向及 RS 测试 |
| 数据下载、校验、时间与坐标审计 | 15/15 | PandaSet 028 的 744 文件；6 路 120 张审计 overlay；显式 actor 时间补丁 |
| 双传感器渲染与公共 API | 20/20 | Camera/LiDAR、actor pose/visibility、ego rig、逐束时间及线/角速度 |
| 三级训练、checkpoint 与恢复 | 20/20 | 300/3000/30000 步、原子保存、两轮真实断点续训 |
| 质量评估、编辑与几何导出 | 15/15 | 240 图像/40 LiDAR；3 actor；20 front 时刻；静态/actor PLY |
| 性能分析与已验证的推理剪枝 | 10/10 | CUDA timing、算子 trace、44+44 组公共输出精确一致、官方指标相同 |
| 运行入口、测试、结果整理 | 5/5 | README、270 passed/1 skipped、可重算的汇总 JSON |
| 原方案的进一步研究 | 5/0 | 训练期真实 LiDAR support/contribution 累计及剪枝；硬件计数器分析 |

## 直接运行

在 PowerShell 切换到 `C:\codex` 后运行已有剪枝模型，无需重新训练：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 render --load-config outputs/pruned_baseline_alpha001/config.yml --requests outputs/baseline_requests_native_motion.json --device cuda --output outputs/render_pruned
```

输出包括 RGB PNG、Camera/LiDAR NPZ、点云及 overlay。原始可续训模型在 `outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/`；剪枝模型标为 inference-only，不能用于续训。

已有结果无需等待渲染即可查看：[20 时刻 GT/重建动画](../outputs/validation_baseline_sequence/reconstruction.gif)、[剪枝后车辆编辑](../outputs/validation_pruned_motion/sample_002/actor_edit_diagnostics.png)、[完整质量/剪枝报告](PRUNING_RESULT.md)、[性能分析](PERFORMANCE_RESULT.md)。动画对应原始模型的已记录请求，不冒称新版角速度动画。

## 已验证结果

- 原模型 500 万 Gaussian，剪枝后 3,672,621，减少 26.55%，三个 actor 全部保留。
- 完整官方 held-out 质量相同：PSNR 25.1963 dB、SSIM 0.78331、LPIPS 0.21774；LiDAR 指标也相同。
- 固定 1920×1080/64×1800 请求：Camera 26.50→27.48 FPS；LiDAR 6.27→7.94 FPS。两传感器单独测量，不能称为联合帧率。
- 含/不含真实角速度的两批对照各 44/44 组精确一致；它们覆盖同 20 个时间点，不是 40 个不同时间点。
- 剪枝模型在带角速度的原生请求下完成三 actor 编辑，恢复误差均为 0；最新全量测试 270 passed、1 skipped。

`scripts/summarize_pruning_results.py` 从原始评估、profiling、逐元素比较、编辑报告及 JUnit XML 重新生成 [机器可读汇总](../outputs/pruning_comparison/summary.json)，包含输入 SHA 和范围一致性检查。

## 原方案的修正与尚未完成项

没有照搬“投影可见就算真实贡献”的剪枝假设。当前已交付的是保留全部 actor 的静态低 opacity 推理过滤，有完整质量/速度证据；训练期间实际 Camera/LiDAR contribution、GT-hit 累计和 MCMC 优化器状态迁移未实现，不能称为 LiDAR-supported 训练算法已完成。

Nsight Compute 的最小 CUDA 探针返回 `ERR_NVGPUCTRPERM`，硬件计数器被驱动权限阻止。没有伪造 occupancy、带宽或 warp-stall 结果，也没有改动系统驱动权限。已完成的 CUDA event 计时和 PyTorch CUDA trace 不依赖该权限。

系统仍是单序列实验实现。图像有边缘/纹理伪影，actor 0 遮挡重，未观测背景的编辑质量有限；采用上游一阶 RS，而非完整曝光积分。Overlay 仅补偿逐束平移，未完整处理相机逐行、角旋转与 actor 跨传感器时差。跨机器发布包、多序列复现和驾驶规划闭环不在这次本机结果中。
