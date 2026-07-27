# v1.1 质量优先重训

本轮目标是在不改变 Ascend 310B 固定 ONNX 合同的前提下，重新训练四个正式架构。旧的
`model-suite-v1.0.0` 永久保留，新结果先进入 `model-suite-v1.1.0-rc1`。

## 训练日程

| 模型 | Controls | Pitch | Refine / Calibrate |
| --- | --- | --- | --- |
| `paper_ir` | 2 次覆盖，legacy，`1e-3` | 1 次，`1e-5` | 1 次，`1e-4` |
| `film_fdn` | 2 次覆盖，legacy，`1e-3` | 1 次，`1e-5` | 1 次，`1e-4` |
| `calibrated_ir` | 继承 `paper_ir` | 继承 pitch | 1 次 perceptual_v2，`3e-4`，冻结 IR |
| `calibrated_film_ir` | 2 次覆盖，perceptual_v2，`3e-4` | 1 次，`1e-5` | 1 次，`1e-4` |

每次覆盖保证所有训练片段恰好出现一次，随后追加 20% 逆分层权重样本。默认 batch 8、FP16 AMP、
vectorized synthesis、fused Adam、2% warmup、cosine decay 和梯度裁剪 1.0。预计总耗时为
48 到 72 GPU 小时。

## 产物与门禁

候选 checkpoint 记录 stage、detune 状态、数据集哈希、十音色样本数量、可训练参数、覆盖率和精确
sampler 位置。最终导出仍为 FP32 ONNX opset 13、batch 1、单个 250 Hz 控制帧、16 复音、16 kHz、
每次 64 个音频样本和显式 GRU 状态。

自动比较要求综合中位数不高于旧版 0.98、任一分组不高于 1.05、响度 P95 回归不超过 1 LU，且
通过 CPU、ONNX Checker、100 帧状态连续数值对比和既有延迟/文件大小门禁。十个音色域全部发布，
全部本地 MIDI 都会生成对应试听 WAV。

人工盲听在所有模型训练和自动评测结束后进行。评分页面最初显示的 3 分只有在用户操作滑杆后才
计入维度统计；仅选择 A、B 或平局也可以提交。人工评测完成前候选不能上传为正式 HF 标签。

本仓库不转换或验证 OM。CANN 算子、内存、数值、延迟及 Ascend 310B 实时音频测试由下游部署
仓库完成，ONNX 验证成功不能表述为 Ascend 部署就绪。
