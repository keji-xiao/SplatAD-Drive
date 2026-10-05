# Baseline Gaussian 分组导出与几何检查

**已在 CPU 导出最终 step 29999 的全部 5,000,000 个 Gaussian 中心，静态和三个 actor 分组计数一致，四份 PLY 重读校验通过。** 这验证了导出文件及分组完整性，不表示场景几何质量通过。实际参数存在显著长尾，actor 分组也非常稀疏，详见下表。

源检查点为 `outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/nerfstudio_models/step-000029999.ckpt`，SHA256 为 `f1ae69b3866746644283cec254f5258174b7f7599b5a70c0d993fd9cd300bc9c`。导出于 2026-09-28 完成，使用 [export_checkpoint_gaussians.py](../scripts/export_checkpoint_gaussians.py) 调用本项目 `restricted_checkpoint_load`，以 `map_location="cpu", weights_only=True` 读取；没有实例化官方模型、使用 GPU、裁剪点或改写检查点。

## 文件与坐标

| 文件 | Gaussian 中心数 | 坐标 |
|---|---:|---|
| [static_gaussians.ply](../outputs/gaussians_baseline/static_gaussians.ply) | 4,999,667 | 检查点的 model/world 坐标，单位米 |
| [actor_000_local.ply](../outputs/gaussians_baseline/actor_000_local.ply) | 65 | Actor 0 的 box 局部坐标，单位米 |
| [actor_001_local.ply](../outputs/gaussians_baseline/actor_001_local.ply) | 196 | Actor 1 的 box 局部坐标，单位米 |
| [actor_002_local.ply](../outputs/gaussians_baseline/actor_002_local.ply) | 72 | Actor 2 的 box 局部坐标，单位米 |

分组来自真实 `_model.gauss_params.id`；静态 ID=3，由 `_model.dynamic_actors.actor_sizes` 的行数决定。三个 actor 总计 333 个中心，占总数 0.00666%。局部文件没有施加轨迹姿态，不能直接与静态 PLY 叠加为某一时刻的世界场景。静态坐标也没有施加 dataparser 的逆变换，不能当作原始 PandaSet 全局坐标。

PLY 是 binary little-endian XYZ/RGB 中心点云，总计约 75 MB。颜色为 `round(clamp(features_dc[:, :3], 0, 1) * 255)`，仅用于预览学习特征，**不是 RGB decoder 的输出，也未进行 SH 转换**。静态分组有 2,615,207 个中心的至少一个特征通道发生裁剪，三个 actor 对应为 53、165、41 个。文件不包含协方差、不透明度、完整特征或解码器，不能独立重现 SplatAD 渲染。

## 实测几何

所有 AABB 都覆盖原始分组的全部中心，没有按不透明度筛选。完整精度、坐标分位数、参数有限性、文件大小及 SHA256 见 [manifest.json](../outputs/gaussians_baseline/manifest.json)。

| 分组 | AABB min，米 | AABB max，米 |
|---|---|---|
| 静态 | (-1707204.625, -889773.313, -1208179.250) | (4456901.500, 980458.938, 411736.469) |
| Actor 0 | (-14.356, -3.030, -11.158) | (13.940, 17.950, 16.438) |
| Actor 1 | (-25.628, -5.927, -12.630) | (22.826, 5.046, 4.465) |
| Actor 2 | (-5.982, -6.018, -6.584) | (3.560, 3.790, 1.551) |

| 分组 | 到所在坐标系原点的距离 p50 / p99 / 最大，米 | 超出 actor 原始 box | 超出 padded box |
|---|---|---|---|
| 静态 | 24.267 / 198.911 / 4,505,280.317 | 不适用 | 不适用 |
| Actor 0 | 4.346 / 19.586 / 23.046 | 31 / 65（47.69%） | 28 / 65（43.08%） |
| Actor 1 | 1.441 / 12.875 / 28.818 | 59 / 196（30.10%） | 49 / 196（25.00%） |
| Actor 2 | 2.176 / 7.224 / 7.876 | 22 / 72（30.56%） | 15 / 72（20.83%） |

静态距离 p99.9 已达到 3,142.072 米；全体 AABB 被少数极远中心主导，不能解释为正常道路空间范围。检查点有限性并不能排除这种长尾。

Actor box 尺寸分别为 `(2.782, 10.397, 3.414)`、`(1.985, 4.641, 1.593)`、`(1.968, 4.779, 1.741)` 米，顺序为局部 x/y/z 对应的 width/length/height。测试条件为任一轴 `abs(local_xyz) > actor_sizes/2`；padded box 每侧再增加 `(0.25, 0.25, 0.10)` 米，与官方 `actor_bounds()` 定义一致。这里只检查中心，未检查 Gaussian 椭球范围。

Padded box 外且 `sigmoid(opacity) >= 0.1` 的中心数分别为 2、7、3；框外中心的 opacity 总和占各 actor opacity 总和的 8.42%、3.80%、5.05%。这描述参数权重，不是可见性或渲染贡献；不能据此断言框外点全部无影响，也不能自动删除。当前点云导出没有解释长尾产生原因，没有证明完整物体几何或动态编辑质量。

## 验证范围

- 实际读取七组 Gaussian 参数：means、scales、quats、features_dc、features_rest、opacities、id；检查形状、行数和有限性。检查 actor size、padding 及 ID 范围。
- 全部中心恰好分为静态和三个 actor；每份 PLY 的 payload 重新读取后与待写数据逐项一致，文件长度与 vertex count 一致。
- CPU 小样本验证涵盖 `_model.` 前缀、静态 ID 推导、分组计数、已有导出保护、RGB 特征裁剪、空 PLY、非法 ID 和非有限坐标拒绝；禁用 CUDA 初始化时完成。
- `EXPORTED_AND_VERIFIED` 和 PLY 的 `PASS` 仅表示上述检查。未进行贡献统计、剪枝、重新训练、碰撞验证或物体几何质量验收。

复现命令（已安装本项目依赖的环境）：

```powershell
python scripts/export_checkpoint_gaussians.py --checkpoint outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/nerfstudio_models/step-000029999.ckpt --output outputs/gaussians_baseline
```

脚本保护已有同名导出，重复执行时应选择新的输出目录。
