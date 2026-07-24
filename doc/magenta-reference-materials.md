# Magenta 参考代码、论文与数据集

本文记录本项目本地保存的 Magenta/DDSP 参考材料、固定版本、数据下载方式和取舍。
参考代码与论文位于 `references/`，训练数据统一位于 `/data/dataset`。下载数据时只使用
`https://hf-mirror.com`，并显式清除 Clash 代理环境变量。

## 参考代码

| 本地目录 | 上游仓库 | 固定提交 | 对本项目的用途 |
|---|---|---|---|
| `references/magenta` | `magenta/magenta` | `c15687ebd1c1` | 历史模型、数据处理和 Onsets and Frames；仓库已归档 |
| `references/note-seq` | `magenta/note-seq` | `358255088853` | MIDI/NoteSequence 解析、对齐和量化 |
| `references/mt3` | `magenta/mt3` | `fa53e12321ac` | 多任务转录的数据合同与评测拆分 |
| `references/music-spectrogram-diffusion` | `magenta/music-spectrogram-diffusion` | `24cd4b0df1b1` | MIDI 条件频谱生成的高音质上限参考 |
| `references/magenta-js` | `magenta/magenta-js` | `96bad8837da4` | 浏览器 MIDI/音频交互参考 |
| `references/magenta-realtime` | `magenta/magenta-realtime` | `2a854047691f` | MRT2 流式生成、状态、缓冲和实时推理接口 |
| `references/magenta-studio` | `magenta/magenta-studio` | `30bca2674f34` | MIDI 工具的产品化接口参考 |
| `references/chamber-ensemble-generator` | `lukewys/chamber-ensemble-generator` | `4647edbce671` | CocoChorales 生成、合成控制和 note-expression 数据 |

所有仓库都是浅克隆，但保留各自 `.git`。GitHub 直连失败时只对这些小型代码仓库使用
`http://127.0.0.1:7890`；数据集不会使用该代理。

`references/magenta-realtime` 的 `main` 已是 Magenta RealTime 2，第一版代码位于上游
`v1_legacy` 分支。当前只保存源码，没有下载 MRT2 权重；其架构、模型合同、官方运行时与
Ascend 适用性详见 [Magenta RealTime 2 技术参考](magenta-realtime-2.md)。

## 本地论文

所有 PDF 统一保存在 `references/papers/`，不再按项目建立论文子目录。该目录下的
`SHA256SUMS` 覆盖全部论文，用于完整性复核。当前集合包括：

- NSynth、GANsynth、Onsets and Frames、MAESTRO、Music Transformer、MusicVAE、
  Piano Genie、Coconet、Groove MIDI Dataset、E-GMD。
- DDSP、MIDI-DDSP、实时 DDSP、DDSP-Piano、CocoChorales、Spectrogram Diffusion。
- CREPE、频谱音频距离局限和 Live Music Models。

《Self-supervised Pitch Detection by Inverse Audio Synthesis》的 OpenReview PDF 入口在本机持续
返回 HTTP 403，因此没有把错误页保存成 PDF；官方论文页仍记录在来源清单中，待站点恢复后补下。

### 人工下载待办

| 论文 | 手动下载链接 | 下载后的目标文件 | 自动下载结果 |
|---|---|---|---|
| Self-supervised Pitch Detection by Inverse Audio Synthesis | https://openreview.net/pdf?id=RlVTYWhsky7 | `references/papers/self-supervised-pitch-inverse-audio-synthesis.pdf` | HTTP 403 |

手动放入后执行以下命令校验，并重新生成清单：

```bash
file references/papers/self-supervised-pitch-inverse-audio-synthesis.pdf
pdfinfo references/papers/self-supervised-pitch-inverse-audio-synthesis.pdf
(cd references/papers && sha256sum *.pdf > SHA256SUMS)
```

`references/papers/ddsp-2001.04643.pdf` 继续作为原始 DDSP 论文的固定副本。下载后使用
`file`、`pdfinfo` 和 SHA-256 检查，不以 HTTP 成功状态代替 PDF 内容校验。

### 已下载论文与来源

| 本地文件 | 论文 | 来源 |
|---|---|---|
| `references/papers/ddsp-2001.04643.pdf` | DDSP: Differentiable Digital Signal Processing | https://arxiv.org/abs/2001.04643 |
| `1704.01279-nsynth.pdf` | Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders | https://arxiv.org/abs/1704.01279 |
| `1710.11153-onsets-and-frames.pdf` | Onsets and Frames: Dual-Objective Piano Transcription | https://arxiv.org/abs/1710.11153 |
| `1802.06182-crepe.pdf` | CREPE: A Convolutional Representation for Pitch Estimation | https://arxiv.org/abs/1802.06182 |
| `1803.05428-musicvae.pdf` | A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music | https://arxiv.org/abs/1803.05428 |
| `1809.04281-music-transformer.pdf` | Music Transformer: Generating Music with Long-Term Structure | https://arxiv.org/abs/1809.04281 |
| `1810.05246-piano-genie.pdf` | Piano Genie | https://arxiv.org/abs/1810.05246 |
| `1810.12247-maestro.pdf` | Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset | https://arxiv.org/abs/1810.12247 |
| `1902.08710-gansynth.pdf` | GANsynth: Adversarial Neural Audio Synthesis | https://arxiv.org/abs/1902.08710 |
| `1903.07227-coconet.pdf` | Counterpoint by Convolution | https://arxiv.org/abs/1903.07227 |
| `1905.06118-groove.pdf` | Learning to Groove with Inverse Sequence Transformations | https://arxiv.org/abs/1905.06118 |
| `1907.06637-bach-doodle.pdf` | The Bach Doodle: Approachable Music Composition with Machine Learning at Scale | https://arxiv.org/abs/1907.06637 |
| `2004.00188-e-gmd.pdf` | Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset | https://arxiv.org/abs/2004.00188 |
| `2012.04572-spectral-audio-distances.pdf` | I'm Sorry for Your Loss: Spectrally-Based Audio Distances Are Bad at Pitch | https://arxiv.org/abs/2012.04572 |
| `2103.07220-realtime-ddsp.pdf` | Real-time Timbre Transfer and Sound Synthesis using DDSP | https://arxiv.org/abs/2103.07220 |
| `2111.03017-mt3.pdf` | MT3: Multi-Task Multitrack Music Transcription | https://arxiv.org/abs/2111.03017 |
| `2112.09312-midi-ddsp.pdf` | MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling | https://arxiv.org/abs/2112.09312 |
| `2206.05408-spectrogram-diffusion.pdf` | Multi-Instrument Music Synthesis with Spectrogram Diffusion | https://arxiv.org/abs/2206.05408 |
| `2209.14458-cocochorales.pdf` | The Chamber Ensemble Generator: Limitless High-Quality MIR Data via Generative Modeling | https://arxiv.org/abs/2209.14458 |
| `2508.04651-live-music-models.pdf` | Live Music Models | https://arxiv.org/abs/2508.04651 |
| `ddsp-piano-dafx.pdf` | Differentiable Piano Model for MIDI-to-Audio Performance Synthesis | https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf |
| `ddsp-piano-frontiers-2023.pdf` | A Review of Differentiable Digital Signal Processing for Music and Speech Synthesis | https://doi.org/10.3389/frsip.2023.1284100 |

## 数据集决策

| 数据集 | 官方规模与用途 | 本机状态 | 决策 |
|---|---|---|---|
| [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) | 约 200 小时配对钢琴 MIDI/音频；完整压缩包约 101 GB | `data/maestro-v3.0.0`，121 GB | 当前钢琴训练的主数据，不重复下载；在 `/data/dataset` 建立统一符号链接入口 |
| [NSynth](https://magenta.tensorflow.org/datasets/nsynth) | 305,979 个 16 kHz、4 秒单音样本 | [HF Parquet 镜像](https://hf-mirror.com/datasets/jg583/NSynth) 29.3 GB | 完整下载；用于键盘子集、起音包络和音色正则研究，不直接替代 MAESTRO |
| [Groove](https://magenta.tensorflow.org/datasets/groove) | 约 13.6 小时鼓演奏；官方 MIDI-only 约 3.1 MB | [HF 完整镜像](https://hf-mirror.com/datasets/schism-audio/groove-midi-dataset)约 13.0 GB | 只下载 MIDI 和元数据，用于时值/力度建模参考 |
| [E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) | 约 444.5 小时鼓演奏；官方 MIDI-only 约 103 MB | [HF 完整镜像](https://hf-mirror.com/datasets/schism-audio/e-gmd)约 141.2 GB | 只下载 MIDI 和元数据，不下载鼓音频 |
| [CocoChorales](https://magenta.tensorflow.org/datasets/cocochorales) | 240k 合成室内乐样本；完整压缩数据约 2.9 TB | 未下载 | 完整体量超过合理预算；HF 没有可信的官方镜像，只保留生成代码和来源 |
| [Bach Doodle](https://magenta.tensorflow.org/datasets/bach-doodle) | 约 21.6M 用户和声化结果；JSONL 约 7.5 GB | 未下载 | 只有符号和声，当前优先级低；HF 无可信镜像 |
| [URMP / MIDI-DDSP TFRecord](https://github.com/magenta/midi-ddsp#prepare-dataset) | 复调室内乐分轨、F0 和表达控制 | 未下载 | HF 搜索结果不是 MIDI-DDSP 官方处理版本，避免混用数据合同 |
| [MT3 数据集合](https://github.com/magenta/mt3/blob/main/mt3/datasets.py) | GuitarSet、URMP、MusicNet、Cerberus4、Slakh 等 | 未下载 | 主要服务多乐器转录，不作为当前钢琴合成训练的默认输入 |

HF 中上述镜像均为第三方镜像，不是 Google/Magenta 官方账号发布。为保证可复现性，脚本固定
仓库提交；论文和数据定义仍以 Magenta 官方页面为准。CocoChorales、Bach Doodle 和
MIDI-DDSP URMP 没有合适 HF 镜像时不会自动回退到 Google Storage，也不会偷偷使用 Clash。

## 下载命令

下载全部已批准数据：

```bash
scripts/download_magenta_datasets.sh all
```

也可以单独执行 `nsynth`、`groove-midi` 或 `e-gmd-midi`。脚本默认写入
`/data/dataset`，可通过 `DATASET_ROOT` 改目录，通过 `HF_MAX_WORKERS` 调整大文件并发。Groove
和 E-GMD 包含大量小文件，固定使用较保守的 2 并发，可用 `HF_SMALL_FILE_WORKERS` 调整。端点固定为
`https://hf-mirror.com`，HF 缓存位于 `/data/dataset/.hf-cache`。
针对镜像链路较慢的 HEAD 和大文件请求，脚本分别设置 60 秒元数据超时和 600 秒下载超时；
HF CLI 会复用 `.incomplete` 分块进行断点续传。
下载结束后还会检查 NSynth 的 37 个 Parquet、Groove 的 1,150 个 MIDI 和 E-GMD 的
45,537 个 MIDI；两个 MIDI-only 目录只要出现 WAV 就判定失败。

本机长任务由 `piano-ddsp-reference-datasets.service` 顺序执行，失败 120 秒后自动断点重试：

```bash
systemctl --user status piano-ddsp-reference-datasets.service
journalctl --user -u piano-ddsp-reference-datasets.service -f
```

## 数据读取合同

NSynth 镜像保存为 Parquet，字段包括 `pitch`、`velocity`、`sample_rate`、
`instrument_family_str`、`instrument_source_str`、`qualities_str` 和内嵌 `audio`。原始拆分按
instrument 隔离，不能重新随机拆分造成同一音源泄漏。当前最相关的过滤条件是
`instrument_family_str == "keyboard"`；官方统计为 54,991 个键盘单音，其中 8,508 个为
acoustic。读取元数据时避免无意解码全部音频：

```python
import pyarrow.dataset as ds

dataset = ds.dataset("/data/dataset/nsynth-hf-jg583/data", format="parquet")
keyboard_metadata = dataset.to_table(
    columns=["id", "pitch", "velocity", "instrument_source_str", "qualities_str"],
    filter=ds.field("instrument_family_str") == "keyboard",
)
```

Groove 使用根目录 `info.csv` 作为完整 1,150 条 MIDI 清单，其中已有官方
train/validation/test 列。E-GMD 同样保留镜像中的原始 CSV 划分。它们只用于时序代码研究，
不会加入钢琴音频训练集。

## 对当前模型的优先级

1. MAESTRO 仍是 v1/v2 公平训练和测试的唯一主数据，避免数据变化掩盖结构变化。
2. NSynth 先按 instrument family/source 过滤键盘类样本，用于研究起音、谐波包络和力度条件，
   验证有效后再进入辅助损失或预训练，不直接混入现有训练。
3. CocoChorales 和 MIDI-DDSP 用于借鉴逐音表达、分层控制与合成参数监督；不把其乐器域
   直接解释为钢琴音质收益。
4. Groove/E-GMD 只用于表达时序代码验证。Spectrogram Diffusion 仅作为离线音质上限，
   不进入 Ascend 310B 的实时 ONNX 路径。

任何新数据进入模型前都必须固定训练/验证/测试划分，并保持当前 ONNX 输入输出合同不变。
数据研究属于训练路径，不改变 16 kHz、250 Hz 控制帧、固定 batch 1 和显式 GRU 状态的
Ascend 310B 推理合同。
