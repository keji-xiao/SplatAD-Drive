# PandaSet 028 官方 SplatAD 300 步 smoke 结果

**结论：本次真实短程联合训练 smoke PASS。** 该结论仅覆盖指定 300 次迭代、进程正常退出、checkpoint 完整性及已记录训练量的有限性，不是论文质量、完整序列动态场景或性能验收。

## 可复查的运行标识

| 项目 | 实际值 |
|---|---|
| Run | `outputs/training/smoke_300/splatad/2026-09-23_073446_729232Z/` |
| 结果摘要 | `outputs/smoke_training_check.json` |
| NeuRAD commit | `8ba9b5116b8a2822a80c64f63a4ae64c5871aa68` |
| SplatAD rasterizer commit | `6e31ad766d39e0c33f9034a2ed772d51364b2343` |
| 数据 | 真实 PandaSet sequence `028`，front Camera + Pandar64 LiDAR，序列前 20% |
| 数据划分 | 官方配置 `train_split_fraction=0.5`；本报告引用训练日志，不引用 held-out 评估 |
| 配置 | `configs/smoke.yaml`；实际解析配置与完整 argv 保存在该 run 的 `config.yml` 和 `launch.json` |
| Seed / 预算 | 42 / 300 次，零起始编号 0..299 |
| 平台 | Windows，Python 3.10.6，PyTorch 2.7.1+cu118，NVIDIA GeForce RTX 3090 |
| 审阅绑定 | audit SHA256 `9d995243333bd424143227328dd02d53919eba56010fd3568d4b72d2c5ace2a1`；仅允许 front / 前 20% / 最多 300 次 |
| 退出记录 | `completion.json`: `returncode=0`, `status=exited_zero` |
| 完成时间 | 2026-09-23 07:35:33 UTC，即上海时间 15:35:33 |

## Checkpoint 与训练量

Checkpoint 为该 run 下 `nerfstudio_models/step-000000299.ckpt`，大小 **498,748,752 字节**。内部 `step=299` 与文件名、300 次预算一致，含 **1,488,899 个 Gaussian**。检查器实际读取了模型与优化器状态，核验参数形状和 **128 个张量**的有限性，没有仅凭文件存在判定成功。

下面数值直接来自该 run 的 TensorBoard 记录。loss 已包含官方配置的相应权重；它们不是 range RMSE、像素均方误差或验证集指标。

| TensorBoard tag | 首记录：step / value | 末记录：step / value | 记录数 | 有限性 |
|---|---:|---:|---:|---|
| `Train Loss` | 0 / 0.331673 | 290 / 0.110373 | 30 | 全部有限 |
| `Train Loss Dict/main_loss` | 0 / 0.328482 | 260 / 0.0511611 | 17 | 全部有限 |
| `Train Loss Dict/depth_loss` | 10 / 0.784644 | 290 / 0.0712704 | 13 | 全部有限 |
| `Train Loss Dict/intensity_loss` | 10 / 0.340674 | 290 / 0.00698714 | 13 | 全部有限 |
| `Train Loss Dict/ray_drop_loss` | 10 / 0.112483 | 290 / 0.00834135 | 13 | 全部有限 |
| `Train Loss Dict/alpha_sum_until_points_loss` | 10 / 0.0456044 | 290 / 0.0209199 | 13 | 全部有限 |
| `Train Metrics Dict/gaussian_count` | 0 / 1,488,899 | 260 / 1,488,899 | 17 | 全部有限 |

训练按 Camera/LiDAR batch 采样，loss 每 10 步记录一次，首末 batch 和传感器可能不同。因此这些首末值只描述日志，不证明误差单调下降或 held-out 重建质量。checkpoint 则验证到最后 step 299。未记录的中间值不能由每 10 步一次的 loss 样本逐一证明。

上游配置保留 `strategy=mcmc`、`warmup_length=500`、`compensate_rs_camera=true`、`compensate_rs_lidar=true` 和 `add_missing_points=true`。300 步仍在 warmup 范围内，Gaussian count 未变化；本次没有验证 warmup 后的 MCMC relocation/densification，也没有验证完整序列动态 actor 的学习效果。

## 时间与显存的含义

- `Train Total (time)` 为 **16.044878 s**，是上游训练循环计时；不含完整数据加载、缓存和模型初始化，不能将它直接解释成端到端训练时间。
- launch 与 completion 的 UTC 时间差约 **46.90 s**，覆盖本次 wrapper 启动至退出记录。
- `Train Iter (time)` 有 300 个有限记录，首条 **3.436703 s**，末条 **0.0432985 s**。二者包含不同迭代条件，不作为稳定训练吞吐量或渲染 FPS。
- `GPU Memory (MB)` 最后记录为 **1044.9502 MiB**：上游实际使用 `torch.cuda.max_memory_allocated()/1024**2`，因此该标签虽写 MB，数值单位是 MiB。它是 step 290 时已记录的 PyTorch 累计分配峰值，不是整张显卡占用，也不是最终模型独占显存。

## 失败记录与修复边界

早期 `2026-09-23_023519_714812Z` 在下载 LPIPS/AlexNet 权重时失败。随后 `2026-09-23_023800_593525Z` 生成了 step 299 checkpoint，但父进程控制台 GBK 无法打印末尾庆祝 emoji，wrapper 的 completion 保持 `returncode=null`。这些旧 completion 没有改写。本报告仅以重新执行且 exit 0 的 `2026-09-23_073446_729232Z` 作为 smoke 成功证据。

日志修复先把原始行写入 UTF-8 文件，再按 stdout 编码用 `backslashreplace` 输出不可表示字符。Checkpoint 检查仍使用 `weights_only=True`，仅允许官方 scheduler 中需要的 NumPy scalar 类型，未改成不受限 pickle 加载。针对这些行为的最新训练回归日志为 **23 passed**（`reports/training_regression.log`）。更早全量测试为 **141 passed / 1 skipped**；最后两项修复后尚未重新执行全量，不把两份记录相加。

## 后续阶段的门槛

当前 smoke 范围内无检查器 blocker，可确认真实 Camera/LiDAR 训练、保存和检查链路可运行。以下工作仍未完成：3000 步中间训练、完整 baseline、固定 held-out split 指标、训练后 20 帧 RGB/LiDAR 对齐、真实动态 actor 编辑、官方模型稳定渲染性能和 pruning A/B。当前限定范围的 alignment review 不能自动授权更大数据范围或更长预算。

单独的 `outputs/validation_smoke/validation.json` 目前基于旧的 `2026-09-23_023800_593525Z` checkpoint，只有一个 train 样本及 0.5 m ego 平移的数值检查，视觉质量标记为 `UNREVIEWED`，actor 编辑因无动态 actor 而跳过；不能作为本成功 run 的 held-out 质量或动态编辑验收。

复查本 run：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/check_training_run.py outputs/training/smoke_300/splatad/2026-09-23_073446_729232Z --output outputs/smoke_training_check.json
```
