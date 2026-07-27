# 训练与 ONNX 导出

MAESTRO 数据保存在 `data/maestro-v3.0.0`，预处理缓存保存在
`cache/maestro-v3.0.0`。可通过 `scripts/download_maestro_dataset.sh` 使用 HF Mirror 下载，并用
`scripts/validate_maestro.py` 验证。

训练命令中的 `--model-id` 必须是四个正式 ID 之一。模型结构和损失来自模型注册表；硬件与运行
参数可显式覆盖：

```bash
python train.py \
  --model-id calibrated_film_ir \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0 \
  --experiment-dir runs/calibrated_film_ir \
  --prepare --device cuda --amp \
  --epochs 20 --steps-per-epoch 2000 --batch-size 1
```

使用 Release checkpoint 继续训练时，设置新的总 epoch 数并传入 `--resume`。不要把不同模型 ID
的 checkpoint 互相恢复。

发布打包程序需要四个源 checkpoint：

```bash
python scripts/prepare_release.py \
  --source paper_ir=path/to/paper.pt \
  --source film_fdn=path/to/film.pt \
  --source calibrated_ir=path/to/calibrated-ir.pt \
  --source calibrated_film_ir=path/to/calibrated-film-ir.pt
```

该程序清理绝对路径和旧实验名称，保留模型、优化器和续训状态，记录权重张量哈希，再对每个模型
执行 CPU 构建、FP32 opset 13 导出、`onnx.checker` 及 100 帧状态连续的 PyTorch/ORT 对比。

成功标准：每个输出均满足 `atol=1e-4, rtol=1e-4`，合同不含服务器绝对路径，所有发布附件的
SHA-256 写入统一清单。这里只能标记为“ONNX 已验证，可交给下游转换”，不能标记为 OM 已验证。
