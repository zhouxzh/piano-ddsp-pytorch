# 来源与许可证

模型结构主要参考：

- [lrenault/ddsp-piano](https://github.com/lrenault/ddsp-piano)，本地核查 revision
  `e868b7ccd3fe`，提供 DAFx22 与 MAESTRO v2 结构，Apache-2.0。
- [ytsrt66589/ddsp-piano-pytorch](https://github.com/ytsrt66589/ddsp-piano-pytorch)，本地核查
  revision `2c9e17aa0c17`，是本仓库早期 PyTorch 基线。
- [Google DDSP](https://github.com/magenta/ddsp)，用于 DDSP 算法和宿主 DSP 边界参考。

发布模型是本仓库使用 MAESTRO 训练得到的 PyTorch checkpoint，不包含上游 TensorFlow 权重。
MAESTRO v3.0.0 由 Google LLC 按 CC BY-NC-SA 4.0 提供，因此发布的 checkpoint、ONNX、参数和
未来 OM 衍生物也按 CC BY-NC-SA 4.0 提供，仅限非商业用途。模型仓库不再分发 MAESTRO 音频或
MIDI。

第三方代码的许可证和修改声明见根目录 `LICENSE`、`LICENSES/` 和
`THIRD_PARTY_NOTICES.md`。本地 `references/` 不随 GitHub 仓库发布。
