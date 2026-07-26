# v2 Q1 音质专项微调

`v2-quality-q1-20260725` 以已经完成 40,000 step 的
`v2a_calibrated_v1` 为父模型，解决当前 release 报告中力度约束过弱、响度离群、频谱质心和
尾音 p95 偏高的问题。它不是新的正式版本；人工盲听完成前，产物状态最多为
`objective_candidate`，不会覆盖 `exports/piano_v1.onnx` 或 `exports/piano_v2.onnx`。

## 启动与恢复

```bash
conda run -n torch python scripts/run_quality_finetune.py \
  --config configs/v2_quality_q1.json \
  --device cuda
```

编排器使用 `runs/quality_finetune/v2-quality-q1-20260725/.lock` 防止重复启动，并把状态写入
同目录的 `state.json`。进程中断后运行同一命令即可恢复：阶段内使用 `last.pt` 恢复优化器、
AMP scaler、随机数和 curriculum sampler；进入新阶段时使用 `--finetune-from` 严格加载父权重，
并重置 optimizer、step、examples 和 RNG。父检查点 SHA256 会写入初始化记录。

第一次启动会生成 `cache/quality/v2-quality-q1-train.json`。该文件只读取 MAESTRO train split，
按曲目单次扫描缓存，并记录训练索引哈希、力度响应斜率和每个片段的采样层。亮度分层使用音频
一阶差分的 RMS 谱矩代理，避免对 348,657 个重叠片段逐个执行 FFT；实际训练和评测中的
频谱质心损失仍使用 1024 点 STFT。

## 阶段

| 阶段 | 候选或策略 | 样本里程碑 | 学习率 | 可训练参数 |
| --- | --- | ---: | ---: | --- |
| controls | uniform 与 curriculum A/B | 4,000、8,000 | `1e-4` | 控制网络，混响冻结 |
| reverb | controls 胜者 | 2,000、4,000 | `3e-5` | 仅 IR 混响 |
| joint | 接受的 reverb，否则回滚 controls | 2,000、4,000 | `2e-5` | controls 与 IR 混响 |

所有阶段使用 batch 4、AMP、vectorized synthesis、combined STFT、fused Adam 和
`torch.compile(reduce-overhead)`。controls/joint 权重为 wet 0.60、energy 0.10、onset 0.10、
centroid 0.08、tail 0.07、velocity 0.05；reverb 权重为 0.70、0.08、0.04、0.06、0.12、0。
固定 512 个 train 片段标定各损失，标定缩放限制为 `[0.1, 100]`。

力度目标不再只检查单调性。构建清单时，从单音起音的 32-160 ms 目标 RMS 中，按钢琴 ID 与
21-47、48-71、72-108 三个音区稳健拟合 `log(RMS)` 对 `log(velocity)` 的斜率；训练时对
low/high 反事实输入后 125 ms 内的振幅比使用 Smooth L1。少于 32 个样本的分组回退到训练集
全局斜率，并将斜率限制在 `[0.5, 2.0]`。

## 自动门槛与产物

controls 和 joint 相对当前 v2A 要求 composite median 不高于 0.98、年份 median 不高于 1.02、
MR-STFT median 回退不超过 0.5%，同时要求 loudness/centroid/tail p95 分别不高于
`5.5 LU`、`0.0165` 和 `8.10 dB/s`。reverb 必须让 tail 或 centroid p95 至少改善 5%，
composite 回退不超过 0.5%，loudness p95 回退不超过 0.25 LU，否则自动回滚。

最终候选执行 release 评测，并与正式 v1 再比较一次；相对 v1 的 composite median 不得高于
0.9339。人工评测材料会生成但不激活计时窗口。

- checkpoints：`runs/quality_finetune/v2-quality-q1-20260725/`
- ONNX：`exports/candidates/v2-quality-q1-20260725/`
- 标准报告：`evaluations/quality_finetune/v2-quality-q1-20260725/`
- 全部 `midi/` WAV：`outputs/listening/v2-quality-q1-20260725/`

每次导出均执行 FP32 opset 13 ONNX checker 和连续 100 帧 PyTorch/ONNX Runtime 数值比较。
部署合同仍是 batch 1、单个 250 Hz 控制帧、16 复音、16 kHz/64 samples 和显式 GRU 状态；
相位、噪声和 IR 卷积继续留在宿主端。当前服务器不执行 ATC、OM 或 CANN 实机测试，因此
Q1 通过不能单独作为 Ascend 310B 部署就绪结论。
