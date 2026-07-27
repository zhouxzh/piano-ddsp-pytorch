# 标准评测与 MIDI 测试集

先锁定 MAESTRO 评测语料，再比较任意两个正式模型：

```bash
python scripts/evaluate_model.py prepare \
  --profile release \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0

python scripts/evaluate_model.py run \
  --profile release \
  --baseline-id paper_ir \
  --candidate-id calibrated_ir
```

报告包含逐片段指标、响度匹配后的音色指标、动态与尾音指标、ONNX 延迟、固定增益与响度匹配盲听
页面。人工评测超时后只标记 deferred，不阻塞其他训练；后续仍可导入评分完成报告。

本机 `midi/` 中的 MuseScore 曲谱只用于本地试听，不进入 Git 或 Release，因为未确认逐首曲谱的
再分发许可。来源链接记录在 [本地 MIDI 试听曲目来源](midi-sources.md)。公开测试使用
`scripts/make_smoke_maestro.py` 生成的合成 fixture。

批量渲染四个正式模型：

```bash
python scripts/render_all_onnx_models.py \
  --midi-dir midi \
  --model-dir artifacts/model-suite-v1.0.0 \
  --output-root artifacts/listening/all_models
```
