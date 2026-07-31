# v1.1 质量优先重训

本轮目标是在不改变 Ascend 310B 固定 ONNX 合同的前提下，重新训练四个正式架构。旧的
`model-suite-v1.0.0` 永久保留，新结果先进入 `model-suite-v1.1.0-rc1`。本轮已经完成，但自动
指标和人工偏好均未证明其优于 v1.0，因此 v1.1.0-rc1 不晋级、不发布。

## 训练日程

| 模型 | Controls | Pitch | Refine / Calibrate |
| --- | --- | --- | --- |
| `gru_ir_96_64` | 2 次覆盖，legacy，`1e-3` | 1 次，`1e-5` | 1 次，`1e-4` |
| `film_fdn_128_96` | 2 次覆盖，legacy，`1e-3` | 1 次，`1e-5` | 1 次，`1e-4`，batch 6 |
| `gru_ir_fullwet_96_64` | 继承 `gru_ir_96_64` | 继承 pitch | 1 次 perceptual_v2，`3e-4`，冻结 IR |
| `film_ir_fullwet_96_64` | 2 次覆盖，perceptual_v2，`3e-4` | 1 次，`1e-5` | 1 次，`1e-4`，batch 6 |

每次覆盖保证所有训练片段恰好出现一次，随后追加 20% 逆分层权重样本。默认 batch 8；实测
`film_fdn_128_96/refine` 在 batch 8 超过 31 GiB，因此两个高显存 refine 阶段固定使用 batch 6，其余阶段
保持 batch 8。训练继续使用 FP16 AMP、vectorized synthesis、fused Adam、2% warmup、cosine decay
和梯度裁剪 1.0。

训练默认写入 TensorBoard event 文件；每个模型阶段单独位于
`<experiment-dir>/tensorboard/`，同时保留 `metrics.jsonl`。TensorBoard 只负责过程观察，模型质量
仍以标准自动报告、试听 WAV 和人工评测为准。

## 产物与门禁

候选 checkpoint 记录 stage、detune 状态、数据集哈希、十音色样本数量、可训练参数、覆盖率和精确
sampler 位置。最终导出仍为 FP32 ONNX opset 13、batch 1、单个 250 Hz 控制帧、16 复音、16 kHz、
每次 64 个音频样本和显式 GRU 状态。

自动比较要求综合中位数不高于旧版 0.98、任一分组不高于 1.05、响度 P95 回归不超过 1 LU，且
通过 CPU、ONNX Checker、100 帧状态连续数值对比和既有延迟/文件大小门禁。十个音色域全部发布，
全部本地 MIDI 都会生成对应试听 WAV。

人工盲听在所有模型训练和自动评测结束后进行。页面只收集更喜欢 A、更喜欢 B 或无明显差别，不要求
普通评测者进行 1–5 分的音色维度评分。人工评测完成前候选不能上传为正式 HF 标签。

## 已完成结果

release 自动报告中四个 refine/calibrate 候选相对 v1.0 的综合中位数分别为：`gru_ir_96_64 1.0121`、
`film_fdn_128_96 1.1009`、`gru_ir_fullwet_96_64 1.2856`、`film_ir_fullwet_96_64 1.2484`。门槛要求不高于
`0.98`，所以四者全部失败。人工偏好也明显倾向旧版：`gru_ir_96_64` 候选 1 胜、旧版 17 胜、2 平；
`film_fdn_128_96` 候选 0 胜、旧版 17 胜、3 平。

中间 checkpoint 使用以下命令统一导出和筛选：

```bash
python scripts/sweep_stage_checkpoints.py \
  --run-root runs/model-suite-v1.1.0-tb \
  --profile quick \
  --output-root runs/model-suite-v1.1.0-tb/checkpoint-sweep/quick
```

筛选不会默认采用最后一个阶段。候选必须同时满足综合中位数不高于 `0.98`、无分组回归、wet/dry
比例相对旧版不漂移超过 25%、P95 延迟不回归超过 5%、PyTorch/ONNX 数值一致且无部署硬失败；
否则明确保留 v1.0。同一输出目录可断点重跑，已有 ONNX 和报告不会重复计算。

## v1.1.1 恢复路线

分析确认 learned IR 正则原先对 24,000 个 IR 样本求和，正则强度随 IR 长度放大，是混响能量塌缩
的重要风险。代码保留 `sum_per_sample` 以复现旧实验，同时新增长度无关的 `mean` 模式。恢复训练
不再随机初始化：从四个 v1.0 checkpoint 开始，以低学习率训练 `controls` 路径并冻结混响。这里
保持与稳定版相同的 phase-1 ONNX 图；启用 detune 的 pitch/refine 图会新增 `Tanh`，在确认目标
CANN 版本支持并更新部署算子合同之前不得晋级。

```bash
python scripts/train_quality_recovery.py
```

参数在 `configs/quality-recovery.json`。每个模型先训练 2,000 step pilot；综合中位数低于 1.0、
无分组回归且通过混响、延迟、数值和部署门禁的模型才自动继续一次完整 coverage。最终发布门槛仍为
`0.98`，报告在 baseline、pilot 和 full checkpoint 中重新选择，不能仅凭训练 loss 发布。
TensorBoard 位于
`runs/model-suite-v1.1.1-recovery/training/<model>/<pilot|full>/tensorboard`。

2026-07-30 pilot quick 结果：

| 模型 | 综合中位数 | 最坏分组 | 回归分组 | wet/dry 旧版 -> pilot | 后续 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gru_ir_96_64` | 0.9435 | 1.1045 | 3 | 2.0561 -> 2.1327 | 完整 coverage |
| `film_fdn_128_96` | 0.9892 | 1.0323 | 0 | 0.9970 -> 0.9971 | 完整 coverage |
| `gru_ir_fullwet_96_64` | 1.0046 | 1.1284 | 2 | 6.3038 -> 6.3787 | 保留 v1.0 |
| `film_ir_fullwet_96_64` | 1.0079 | 1.1513 | 3 | 1.8393 -> 1.8319 | 保留 v1.0 |

四个 pilot 均通过 FP32 opset 13 导出和 100 帧 PyTorch/ONNX 状态对比，且没有部署硬失败。结果
证明冻结 IR 已修复混响能量塌缩，但只有 `gru_ir_96_64`、`film_fdn_128_96` 值得继续。完整训练完成前这两个
checkpoint 仍只是实验候选，不能替换稳定发布。

本仓库不转换或验证 OM。CANN 算子、内存、数值、延迟及 Ascend 310B 实时音频测试由下游部署
仓库完成，ONNX 验证成功不能表述为 Ascend 部署就绪。
