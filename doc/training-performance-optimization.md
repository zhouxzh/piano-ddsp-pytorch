# 训练性能优化与验证记录

本文记录 2026-07-25 完成的训练路径优化、基准结果和后续启用方式。优化只作用于
NVIDIA GPU 训练路径，不改变 Ascend 310B 的固定 ONNX 控制模型合同。

## 已实现内容

- `PianoModel` 增加 `serial` 参考路径和 `vectorized` 训练路径。后者把 16 个复音槽位与
  batch 轴合并，一次执行谐波和滤波噪声 DSP，再按槽位求和。
- 多尺度频谱损失缓存 Hann 窗，并把 target/prediction 合并为一次 batched STFT。
- 固定谐波编号注册为非持久 buffer，避免每个声部、每步重复创建。
- 力度反事实损失把 low/high 条件合并为一次 `predict_controls` 调用。
- 训练 loss 保持在 GPU 上累积，只在日志与 epoch 结束时同步；日志频率可配置。
- 训练和验证 DataLoader worker 分离，验证可只在本次训练调用结束时执行。
- 增加可选 `torch.compile`、fused Adam 和独立的训练基准程序。
- checkpoint schema 升级为 `ddsp-piano-training-checkpoint/v2`，记录
  `examples_seen`、Python/NumPy/PyTorch/CUDA RNG、shuffle generator 和最近一次吞吐/显存。
  旧 checkpoint 仍可恢复，缺失字段使用兼容默认值。

为了防止正在执行的旧质量周期改变轨迹，`train.py` 的兼容默认值仍是
`serial + standard Adam + no compile`。优化路径必须由参数或新配置显式开启。

## RTX 5090 D 基准

统一条件：v2A 96/64 IR、3 秒、16 复音、AMP、20 次预热、200 次计时。原始 JSON 位于
`runs/benchmarks/training/`。

| 配置 | step/s | examples/s | 峰值分配显存 | 峰值保留显存 |
| --- | ---: | ---: | ---: | ---: |
| batch 1, serial | 3.94 | 3.94 | 1.06 GB | 1.15 GB |
| batch 1, vectorized | 13.60 | 13.60 | 2.50 GB | 2.64 GB |
| batch 2, vectorized | 13.00 | 26.00 | 4.95 GB | 5.24 GB |
| batch 4, vectorized | 8.78 | 35.11 | 9.89 GB | 10.33 GB |
| batch 8, vectorized | 6.95 | 55.63 | 19.75 GB | 20.62 GB |
| batch 4, vectorized, fused Adam | 9.35 | 37.40 | 9.89 GB | 10.33 GB |
| batch 4, vectorized, fused Adam, compile | 19.45 | 77.80 | 1.46 GB | 3.49 GB |

`torch.compile` 会提示 TorchInductor 不能为全部 complex FFT 算子生成代码，但本机运行稳定，
并对 FFT 之外的图产生收益。它仍保持显式开关；PyTorch/CUDA 版本变化后必须重新基准。

## 真实数据 pilot

使用 MAESTRO 完整训练缓存和固定 200 片段平衡验证集，batch 4 优化路径训练 500 step，
即 2,000 个样本：

- `61.94 examples/s`，训练峰值分配显存 `3.04 GB`；
- 训练 loss `2.5285`；
- 平衡验证 loss `2.6546`，当前 v2A 40k checkpoint 为 `2.6507`；
- checkpoint：`runs/training_optimization/pilot-v2a-b4/checkpoints/last.pt`；
- ONNX：`exports/training_optimization/pilot-v2a-b4.onnx`。

pilot 只说明短程数值稳定，不能证明 batch 4 的长期优化轨迹和听感等价。batch 变化时必须按
`examples_seen` 比较，并重新选择学习率/步数；因此正式保守配置仍使用 batch 1。

## 使用方式

重复基准：

```bash
conda run -n torch python scripts/benchmark_training.py \
  --architecture v2a --batch-size 1 \
  --warmup-steps 20 --timed-steps 200 \
  --synthesis-layout vectorized
```

当前质量周期结束后，启动保持 batch/step 轨迹的优化周期：

```bash
conda run -n torch python scripts/run_quality_cycle.py \
  --config configs/v2_optimized_quality_cycle.json \
  --device cuda
```

该配置在 8k/20k/40k 做同一套 dev/release 门禁与人工盲听，使用 batch 1、矢量化 DSP、
fused Adam、`torch.compile`、训练 worker 4、验证 worker 0，并只在每个质量阶段结束时验证。
任何自动指标或盲听回退都应阻止替换现有正式模型。

batch 4/8 只能作为第二阶段实验。至少要求固定 `examples_seen`、相同校准集、2k 样本 pilot、
完整 40k 对照、标准 MIDI 盲听和 release 报告后，才能提升为默认训练配置。

`v2-quality-q1-20260725` 是一个显式命名的 batch-4 专项微调实验，不会把 batch 4 改成全量训练
默认值。它从固定 v2A 权重开始，所有里程碑按 `examples_seen` 定义，并同时训练 uniform 与
curriculum 两个 controls 候选；任何阶段回退都会保留父检查点。配置和验收标准见
[`v2-quality-q1.md`](v2-quality-q1.md)。

## ONNX 与 Ascend 边界

pilot 使用 `scripts/export_onnx.py --verify-steps 100` 导出并通过 ONNX checker 与 100 个连续
状态帧的 CPU 数值比较，最大绝对误差 `8.5831e-6`。合同仍为 FP32、opset 13、batch 1、
单个 250 Hz 控制帧、16 复音槽位和显式 GRU 状态。矢量化谐波/噪声、STFT、compile 和 fused
Adam 均为训练路径，不进入 ONNX 图。

当前服务器没有执行 OM/ATC/CANN 验证。ONNX 成功不能据此标记为 Ascend 310B 部署就绪。
