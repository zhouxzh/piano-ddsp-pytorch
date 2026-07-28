# 训练与 ONNX 导出

MAESTRO 数据保存在 `data/maestro-v3.0.0`，预处理缓存保存在
`cache/maestro-v3.0.0`。可通过 `scripts/download_maestro_dataset.sh` 使用 HF Mirror 下载，并用
`scripts/validate_maestro.py` 验证。

稳定版注册表保持不可变。质量优先重训使用候选注册表和可恢复编排器：

```bash
python scripts/train_model_suite.py
```

程序先构建 train-only 质量 manifest，再对四个架构做 batch 8 基准；显存预留超过 26 GiB 时
自动降为 batch 6。每个 coverage epoch 遍历全部 348,657 个训练片段一次，再追加 20% 困难样本。
每处理约四分之一训练集执行一次 200 片段平衡验证并保存可精确恢复的 sampler 偏移。

显式训练阶段为：

- `controls`：关闭 detune，训练控制网络和混响。
- `pitch`：打开 detune，只训练失谐、非谐性及对应 embedding。
- `refine`：保持 detune 生效，冻结 pitch 参数并微调控制网络。
- `calibrate`：从 `paper_ir/refine` 初始化 `calibrated_ir`，使用感知损失并冻结 IR。

编排状态位于当前 `--run-root` 下的 `pipeline-state.json`（本轮为
`runs/model-suite-v1.1.0-tb/pipeline-state.json`）。进程中断后执行同一命令即可从
`last.pt` 的 epoch、样本偏移、优化器、LR scheduler、AMP scaler 和 RNG 状态继续。

每个模型阶段同时写入 `<experiment-dir>/tensorboard/`。在服务器上查看新的 TensorBoard 训练记录：

```bash
tensorboard --logdir runs/model-suite-v1.1.0-tb --bind_all --port 6006
```

通过 VSCode 转发 `6006` 端口即可查看。`metrics.jsonl` 仍是机器可读的权威记录，TensorBoard
用于观察 loss、验证损失、学习率、覆盖率、吞吐和显存。

旧训练如果没有 event 文件，可以将保留的 `metrics.jsonl` 和 `pipeline.log` 导入当前 TensorBoard：

```bash
python scripts/import_training_metrics_to_tensorboard.py \
  --source-root runs/model-suite-v1.1.0-rc1 \
  --output-root runs/model-suite-v1.1.0-tb/history/model-suite-v1.1.0-rc1 \
  --pipeline-log runs/model-suite-v1.1.0-rc1/pipeline.log
```

导入程序不会覆盖已有 event 文件；需要重新导入时请指定新的输出目录。

发布打包程序需要四个源 checkpoint：

```bash
python scripts/prepare_release.py \
  --registry ddsp_piano/model-suite-v1.1.0-rc1.json \
  --source paper_ir=path/to/paper.pt \
  --source film_fdn=path/to/film.pt \
  --source calibrated_ir=path/to/calibrated-ir.pt \
  --source calibrated_film_ir=path/to/calibrated-film-ir.pt
```

编排器在全部阶段结束后自动调用该程序。它清理绝对路径和旧实验名称，保留模型、优化器和续训
状态，记录权重张量哈希，再对每个模型
执行 CPU 构建、FP32 opset 13 导出、`onnx.checker` 及 100 帧状态连续的 PyTorch/ORT 对比。

成功标准：每个输出均满足 `atol=1e-4, rtol=1e-4`，合同不含服务器绝对路径，所有发布附件的
SHA-256 写入统一清单。这里只能标记为“ONNX 已验证，可交给下游转换”，不能标记为 OM 已验证。

候选还会对 `midi/` 的全部曲目渲染四模型乘十个 MAESTRO 录音域，并分别与 `v1.0.0` 同名模型
生成 release 自动报告和未启动截止时间的盲听包。未操作的默认 3 分不进入维度统计；只选偏好仍是
有效人工结果。人工评测完成前不得把 `v1.1.0-rc1` 提升为正式版本。
