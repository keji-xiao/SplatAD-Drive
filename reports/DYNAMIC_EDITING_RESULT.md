# 真实 baseline 的动态编辑与联合渲染结果

**原生分辨率下的动态编辑数值机制已通过。** 最终 step 29999 官方 checkpoint 中的 3 个 actor 均实际执行移除、1 m 平移、15° yaw 旋转和恢复；所有检查输出有限，3 次恢复的最大输出差异均为 0。另有 20 个 held-out front 时刻完成联合渲染并通过有限性检查。本文更新至 2026-09-28，数值结论不等于所有遮挡区域或所有视角的视觉质量通过。

## 验证条件

| 项目 | 实际范围 |
|---|---|
| 模型 | `outputs/training/baseline_30000/splatad/2026-09-27_053445_795104Z/nerfstudio_models/step-000029999.ckpt` |
| Gaussian / actor | 5,000,000 / 3 |
| 原始数值报告 | `outputs/validation_baseline_full_range/validation.json` |
| 数据/样本 | PandaSet 028，train split，front index=39 / 原始 front 帧79；3 个样本对应分别编辑 actor 0/1/2，不是 3 个不同时间 |
| Camera | **1920×1080**，原生标定，OpenCV 公共坐标，far=1e10 m |
| LiDAR | **64×1800**，Pandar64 原生 elevation，sensor-local 坐标，far=200 m |
| 采样时间 | Camera 为中心曝光时刻，LiDAR 为原生 sweep 参考时刻；两者保留自身时间，不假定同步 |
| Rolling shutter | 启用 Camera 原生 readout 与 LiDAR native -Y 前向相位估计，周期 0.100007800 s |
| 请求文件 | `outputs/baseline_requests_native.json`；各样本另存 `requests.json`、`sensors.json` |

公共接口对官方模型采用锚点刚体增量作用于 actor 轨迹，保留时间内运动，编辑状态在上下文退出时恢复。参考后端插入关键帧的语义不同，不以参考合成案例替代本次真实 checkpoint 结果。

## 移除、平移、旋转与恢复

下表为每次编辑相对 baseline 的**最大绝对变化**。RGB 在 [0,1] 范围，range 单位为米；变化量说明编辑有响应，不是物理真值误差或编辑质量分数。

| Actor | 移除 RGB / range | 平移 RGB / range | 旋转 RGB / range | 恢复最大误差 |
|---|---:|---:|---:|---:|
| 0 | 0.402388 / 20.788193 m | 0.390916 / 10.481190 m | 0.275708 / 7.344643 m | **0.0** |
| 1 | 0.629749 / 18.499504 m | 0.631298 / 12.445473 m | 0.655679 / 5.250664 m | **0.0** |
| 2 | 0.566002 / 35.909531 m | 0.644070 / 34.625977 m | 0.426283 / 5.242775 m | **0.0** |

3 个 actor 的参数恢复检查也均为 true。移除、平移、旋转、恢复及 baseline 中检查的 Camera RGB/depth/alpha、LiDAR range/intensity/hit-probability/ray-drop/points 均具有预期形状和有限值。完整均值、附加通道和逐项检查保存在原始 JSON，不凭一幅截图判定恢复成功。

Ego 验证将两传感器共同平移 **0.5 m**，分别渲染 Camera 与 LiDAR；相对外参最大误差为 **1.4901161e-8**，输出有限。这说明本次请求保持传感器刚性标定，不表示经过完整车辆动力学或规划闭环验证。

## 20 个 held-out front 时刻

`outputs/validation_baseline_sequence/validation.json` 的 split 为 eval、样本数为 **20**，index 为 0,12,…,228，各样本 `sensors.json` 均为 **front**。这覆盖 20 个 held-out front 时刻的 RGB/LiDAR 联合渲染，不是 6 路相机各 20 帧；6 路 120 张 overlay 的早期 parser 审计和官方 240 张图像评估是不同证据。

此轮 RGB 为 **960×540**、LiDAR 为 **64×1800**，所有检查输出有限，RGB/深度/NPZ/PLY 与 overlay 已导出。Actor 编辑按选项标记 `SKIPPED_BY_OPTION`，ego 编辑也未执行，因此不将其写为 20 帧动态编辑通过。已实际查看全部 20 帧 RGB/overlay contact sheet，并抽查完整 overlay；观察见 [序列视觉记录](../outputs/validation_baseline_sequence/visual_review.md)，预览见 [GT/重建动画](../outputs/validation_baseline_sequence/reconstruction.gif)。原始自动数值状态保持 `NUMERIC_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED`。

## 远裁剪修复与视觉边界

初次 `outputs/validation_baseline/` 使用 Camera far=200 m。后续控制对照保持同一个 step29999 checkpoint、同 front 训练帧79、同 960×540 GT，只把 far 改为 1e10 m，PSNR 从 **16.7093** 到 **23.8027 dB**，见 `outputs/far_clip_comparison.json`。实际看图确认天空和远处建筑恢复，说明旧请求截去了远景；这个单训练视角结果不是 held-out 均值，也不是新训练带来的提升。

公共 `CameraRequest.far` 现默认 1e10 m，与官方范围一致；LiDAR 仍为 200 m。Full-range 原生 baseline 已被主任务直接查看，仍有边缘和纹理失真。旧中间模型中的 actor 0 被部分遮挡且外观弥散；不能从数值恢复或整体轮廓直接宣称完整、干净的车辆移除。最终 actor 编辑裁剪图和序列图像的视觉审阅与数值报告分开保留；本次自动报告仍为 `visual_quality=UNREVIEWED`。

LiDAR 相位是 novel raster 的校准时间估计，不是每束真实硬件采样时间。本文旧 baseline 请求仅含线速度。2026-09-30 公共接口已补齐 sensor-local `angular_velocity`，通过坐标转换契约及真实 CUDA 定向测试；新 motion 请求单独保存。Overlay 仍仅进行 LiDAR 逐束传感器平移补偿，使用 Camera nominal center pose，未包含角运动或 actor 跨时间补偿，因此不称为像素级 RS/deskew 对齐验收。

## 回归与尚未完成的范围

`reports/tests_release.log`/`.xml` 记录最新 **270 passed、1 skipped、2 warnings**，包含真实 CUDA。唯一 skip 为 `test_unavailable_cuda_is_explicit`，它只适用于 CUDA 不可用环境；本机 CUDA 已可用。此前 206 项快照单独保留。

三个 actor 的最终原生诊断图均已直接审阅，详见 [编辑视觉记录](../outputs/validation_baseline_full_range/visual_review.md)。Actor 1/2 的移除和移动清晰可见，但存在残影；actor 0 的遮挡和弥散更重。

最终模型已完成专用同步 profiling：固定请求下 Camera 26.5046 FPS、LiDAR 6.2653 FPS，完整范围见 [baseline 报告](BASELINE_RESULT.md)。完整物理曝光、全部动态时刻/遮挡质量、多序列泛化及驾驶闭环不在本次证据内。静态低透明度推理剪枝的质量/速度 A/B 另行记录；原 `pruning.py` 的 LiDAR-supported candidate mask 尚未接入训练状态迁移。

2026-09-30，剪枝模型与新增实际角速度请求的三个 actor 原生验证也已完成，见 `outputs/validation_pruned_motion/validation.json`：所有检查输出有限，参数恢复成功、恢复误差均为 0，相对外参误差 1.49e-8。三张编辑诊断图已审阅，见 [剪枝编辑视觉记录](../outputs/validation_pruned_motion/visual_review.md)，仍保留同类遮挡和外观伪影限制。
