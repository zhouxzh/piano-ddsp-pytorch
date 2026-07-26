# v3 Candidate 结构升级与质量周期

记录日期：2026-07-26

## 背景与结论

`v2-quality-q1-20260725` 已结束，状态为 `no_improvement`。uniform 与 curriculum
controls 候选在 4,000 examples 后已经开始回退，8,000 examples 的退化更明显；内部
validation loss 下降，但标准报告中的 composite、响度 p95、频谱质心 p95 及 2017/2018
年度分组没有同步改善。因此 v3 不继续延长旧 controls 微调，也不重复已经失败的
`residual_film`、`residual_deep`、`residual_joint` 或完整 FiLM/deep 路线。

本周期名为 `v3-candidate-20260726`。它是结构候选，不是正式 v3，不覆盖
`exports/piano_v1.onnx` 或 `exports/piano_v2.onnx`。自动门禁通过后的最高状态是
`objective_candidate`；只有全部训练结束后的人工盲听通过，才可以执行独立的正式发布动作。

## 模型结构

v3 保留 v2A 的 legacy context GRU、192 维 monophonic GRU、学习型 IR 和 96/64 控制维度，
将单个 161 维输出投影拆成 amplitude、harmonic 与 noise 三个独立 head。父 checkpoint 的
`dense_out` 按 `[1, 96, 64]` 无损切分到三个 head。可选的 `velocity_onset` gate 使用
Linear、LeakyReLU 和 Tanh，根据标准化 MIDI conditioning 与 context 分别产生三个残差门控；
末层零初始化。

每个候选训练前执行连续 100 帧父模型等价性预检，允许 FP32 多 GEMM 带来的最大绝对误差
`1e-5`。训练加入冻结 v2A teacher，对后处理后的 log-amplitude、归一化 harmonic distribution
和 log-noise controls 使用权重 `0.10` 的 Smooth L1 一致性损失。teacher 与一致性损失均不进入
ONNX 图。

两个筛选候选为：

| 候选 | 独立输出 head | 力度/起音 gate |
| --- | --- | --- |
| `v3a_factorized_heads` | 是 | 否 |
| `v3b_velocity_onset_gate` | 是 | 是 |

## 训练、筛选与恢复

启动或恢复完整周期：

```bash
conda run -n torch python scripts/run_v3_quality_cycle.py \
  --config configs/v3_quality_cycle.json --device cuda
```

训练使用 batch 4、学习率 `3e-5`、AMP、fused Adam、vectorized synthesis、combined spectral
layout 和 `torch.compile(reduce-overhead)`。损失权重为 wet 0.70、energy 0.12、onset 0.10、
centroid 0.03、velocity 0.05、tail 0；固定 512 个训练片段的损失标定上限由 100 降为 20。

两个候选依次评测 1k、2k、4k examples。连续两个里程碑出现 composite `>1.02` 或任一年度
分组 `>1.08` 时自动淘汰。4k 候选必须满足 composite `<=1.00`、MR-STFT `<=1.005`、
年度分组 `<=1.02`，并使 loudness/centroid p95 各改善至少 5%。胜者继续训练和导出
12k、24k、40k checkpoints；若没有筛选候选通过，周期以 `no_improvement` 结束。

40k release 候选相对 v2A 必须满足 composite 不退化、loudness/centroid p95 各改善至少
10%、tail p95 回退不超过 5%，同时相对 v1 的 composite 不高于 `0.9339`。所有命令、哈希、
报告和失败原因写入可恢复的 `state.json`。

产物目录：

- checkpoints：`runs/quality_cycle/v3-candidate-20260726/`
- ONNX：`exports/candidates/v3-candidate-20260726/`
- 自动报告：`evaluations/quality_cycles/v3-candidate-20260726/`
- MIDI 试听：`outputs/listening/v3-candidate-20260726/`

全部训练和 release 评测结束后，程序才准备人工材料，并为 `midi/` 中全部曲目生成 wet gain
`0.25`、`0.50`、`0.75`、`1.00` 四组 WAV。人工状态保留为 `prepared`，不会启动倒计时或阻塞
无人值守训练。

## ONNX 部署合同

最终 milestone 的导出命令为：

```bash
conda run -n torch python scripts/export_onnx.py \
  --checkpoint runs/quality_cycle/v3-candidate-20260726/candidates/<winner>/checkpoints/epoch_0040.pt \
  --output exports/candidates/v3-candidate-20260726/<winner>/examples_40000.onnx \
  --model-variant v3 --verify-steps 100
```

导出保持 FP32、opset 13、batch 1、单个 250 Hz 帧、16 复音及 16 kHz/64 samples。输入为
`conditioning [1,1,16,2]`、`pedal [1,1,4]`、`piano_model [1]`、
`extended_pitch [1,1,16,1]`、`context_state [1,1,64]` 和
`monophonic_state [1,16,192]`。控制输出仍为 amplitude 1、harmonic 96、inharmonicity 1、
f0、noise 64、IR 24000 以及两个同尺寸 next state。

谐波相位、filtered-noise FFT、IR 卷积和一秒 MIDI release state 继续位于宿主端。gated 候选
新增 ONNX `Tanh`，必须在下游指定 CANN 版本中继续验证；当前服务器只完成 PyTorch CPU、
ONNX checker 和 ONNX Runtime 数值验证，不执行 OM/ATC，也不据此宣称 Ascend 310B 已部署。
