# DDSP Piano PyTorch

[English](README.md)

本仓库用于训练、导出和测试专门面向 Ascend 310B 实时 MIDI 钢琴合成的 ONNX 神经控制模型。
PyTorch CPU 和 ONNX Runtime 只是数值参考路径，不是额外的部署目标。首个正式发布为
`model-suite-v1.0.0`，同时保留四个结构差异明显的模型：

| 模型 ID | 结构 | 谐波/噪声 | 宿主混响 |
| --- | --- | ---: | --- |
| `paper_ir` | DAFx22 论文结构 | 96 / 64 | 学习型 IR，wet 0.25 |
| `film_fdn` | 后期 MAESTRO v2 FiLM/深层结构 | 128 / 96 | FDN 控制 |
| `calibrated_ir` | 旧控制网络加感知损失标定 | 96 / 64 | 学习型 IR，wet 1.0 |
| `calibrated_film_ir` | FiLM/深层/联合非谐性加感知标定 | 96 / 64 | 学习型 IR，wet 1.0 |

这些名称只描述结构，不表示音质排名。现有自动指标和人工试听不足以确定唯一胜者，因此四个模型都
作为正式的 Ascend 310B 移植候选发布，待 OM 实机测试后再决定最终保留方案。

## 快速开始

CPU 推理和验证使用 Python 3.11：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -v
```

从 Hugging Face 模型仓库 `zhouxzh/piano-ddsp-ascend310` 下载
`model-suite-v1.0.0` 版本，然后检查：

```bash
HF_ENDPOINT=https://huggingface.co hf download zhouxzh/piano-ddsp-ascend310 \
  --revision model-suite-v1.0.0 \
  --local-dir artifacts/model-suite-v1.0.0
cd artifacts/model-suite-v1.0.0
sha256sum -c SHA256SUMS
```

渲染本地 MIDI 或启动网页实时试听：

```bash
python scripts/render_onnx.py --model-id paper_ir --midi path/to/input.mid --output output.wav
python scripts/realtime_midi_server.py --host 0.0.0.0 --port 8765
```

网页会列出已安装的四个模型。切换模型时，服务会停止旧音频并清空 MIDI、GRU、相位、噪声和混响
状态，避免旧模型残留声音。

## 训练与发布

CUDA 训练环境使用：

```bash
pip install -r requirements-cuda.txt
python scripts/check_environment.py --require-cuda
python train.py \
  --model-id paper_ir \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0 \
  --experiment-dir runs/paper_ir \
  --prepare --device cuda --amp
```

四个 ID 的结构、损失和标准训练默认值固定在
`ddsp_piano/model-suite-v1.0.0.json`。`--epochs`、`--steps-per-epoch`、`--batch-size`、
`--lr` 和 `--seed` 可显式覆盖；checkpoint 会保存最终有效配置。

单个模型导出：

```bash
python scripts/export_onnx.py \
  --model-id paper_ir \
  --checkpoint runs/paper_ir/checkpoints/best.pt \
  --output artifacts/model-suite-v1.0.0/ddsp_piano_paper_ir.onnx \
  --verify-steps 100
```

完整发布和验证流程见 [训练与 ONNX 导出](docs/training-and-export.md)。

## 部署边界

固定 ONNX 合同为 FP32、opset 13、batch 1、单个 250 Hz 控制帧、16 复音、16 kHz 和显式 GRU
状态。谐波相位、滤波噪声、IR/FDN 混响及一秒 MIDI release 状态位于宿主端。

本仓库只验证 PyTorch CPU 和 ONNX Runtime，不进行 OM 转换和昇腾实机验证。Ascend 310B/CANN
上的 ATC 转换、算子、内存、数值、延迟和实时音频测试必须在下游部署仓库完成。

## 文档

- [模型与发布附件](docs/models.md)
- [GitHub 与 Hugging Face 发布流程](docs/publishing.md)
- [训练与 ONNX 导出](docs/training-and-export.md)
- [标准评测与本地 MIDI 测试集](docs/evaluation.md)
- [实时网页播放器](docs/realtime.md)
- [Ascend 310B 交接合同](docs/ascend-310b.md)
- [上游来源与许可证](docs/provenance.md)

本仓库新增代码使用 MIT，改编的 DDSP-Piano 代码保留 Apache-2.0。发布的 checkpoint、ONNX、
模型参数和后续 OM 衍生物使用 CC BY-NC-SA 4.0，仅限非商业用途。
