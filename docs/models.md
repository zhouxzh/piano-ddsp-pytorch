# 四模型定义

`model-suite-v1.0.0` 固定使用 `paper_ir`、`film_fdn`、`calibrated_ir` 和
`calibrated_film_ir`。旧名称 `v1`、`v2`、`v2a`、`v2b` 及其文件名不再是公开接口；程序遇到
旧 ID 会报错并提示迁移目标。

`paper_ir` 和 `film_fdn` 分别对应上游 `dafx22.gin` 与 `maestro-v2.gin` 的本地 PyTorch
训练实现，不是上游 TensorFlow checkpoint 的格式转换。两个 calibrated 模型是本仓库的训练
修改版本。四者的权重均来自本仓库 MAESTRO 训练。

每个模型在 GitHub Release 中包含：

- `ddsp_piano_<id>.onnx`：供 ONNX Runtime 验证和后续 OM 转换的控制模型。
- `ddsp_piano_<id>.json`：输入输出 shape、dtype、算子、哈希、DSP 边界和数值对比合同。
- `ddsp_piano_<id>.pt`：清理过路径的 PyTorch 可续训 checkpoint。

Release 另含 `model-suite.json`、`VALIDATION.md` 和 `SHA256SUMS`。所有模型状态固定为
`onnx_status=verified`、`om_status=pending`、`quality_status=quality_selection_pending`。

`paper_ir`、`calibrated_ir` 和 `calibrated_film_ir` 输出 `reverb_ir`；`film_fdn` 输出九维
`reverb_controls`。这个差异由相邻 JSON 明确记录，宿主不能把两种输出互换。
