# 标准评测与 MIDI 测试集

先锁定 MAESTRO 评测语料，再比较任意两个正式模型：

```bash
python scripts/evaluate_model.py prepare \
  --profile release \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0

python scripts/evaluate_model.py run \
  --profile release \
  --baseline-id gru_ir_96_64 \
  --candidate-id gru_ir_fullwet_96_64
```

报告包含逐片段指标、响度匹配后的音色指标、动态与尾音指标、ONNX 延迟、固定增益与响度匹配盲听
页面。人工页面只收集 A/B/无明显差别偏好，不收集维度评分。评测超时后只标记 deferred，不阻塞
其他训练；后续仍可导入偏好结果完成报告。

`quick` 和 `dev` 用于 checkpoint 筛选，只渲染每个目标片段之前 2.5 秒的真实 MIDI 上下文及目标
窗口，以保留释放态、GRU 状态和混响预卷。`release` 始终从曲目开头连续渲染，候选晋级不得以
quick/dev 的近似结果代替 release 报告。

训练阶段结束后用 `scripts/sweep_stage_checkpoints.py` 比较所有 `best.pt`，并生成
`summary.json`、`summary.md` 及逐模型 `promotions`。晋级结果包含 baseline 回退，因此 refine
阶段不会再无条件覆盖更好的 controls/pilot checkpoint。

通过支持 WAV 分段请求的专用服务打开人工评测，根地址只列出已经生成试听页面的模型：

```bash
python scripts/serve_listening.py \
  --root runs/model-suite-v1.1.0-tb/evaluation \
  --bind 0.0.0.0 \
  --port 8766
```

本机 `midi/` 中的 MuseScore 曲谱只用于本地试听，不进入 Git 或 Release，因为未确认逐首曲谱的
再分发许可。来源链接记录在 [本地 MIDI 试听曲目来源](midi-sources.md)。公开测试使用
`scripts/make_smoke_maestro.py` 生成的合成 fixture。

批量渲染四个正式模型：

```bash
python scripts/render_all_onnx_models.py \
  --midi-dir midi \
  --model-dir artifacts/model-suite-v1.0.1 \
  --output-root artifacts/listening/all_models
```
