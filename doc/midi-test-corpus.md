# MIDI 测试曲目与来源

这些 MIDI 用于人工试听 ONNX 模型的音色、混响、动态和长序列稳定性。文件由用户于
2026-07-23 从 MuseScore 下载，保存在仓库的 `midi/` 目录中。

## 曲目清单

| 曲目 | 本地文件 | 时长 | MuseScore 检索链接 | 原下载页 |
| --- | --- | ---: | --- | --- |
| Canon in D - Johann Pachelbel | `midi/canon-in-d-johann-pachelbel.mid` | 04:04.68 | [检索](https://musescore.com/sheetmusic?text=Canon%20in%20D%20Johann%20Pachelbel) | [原始乐谱页](https://musescore.com/user/1809056/scores/1019991) |
| Prelude I in C major, BWV 846 - J. S. Bach | `midi/prelude-i-in-c-major-bwv-846-well-tempered-clavier-first-book.mid` | 02:10.62 | [检索](https://musescore.com/sheetmusic?text=Prelude%20I%20in%20C%20major%20BWV%20846) | [原始乐谱页](https://musescore.com/user/101554/scores/117279) |
| Nocturne Op. 9 No. 2 in E-flat major - Frederic Chopin | `midi/chopin-nocturne-op-9-no-2-e-flat-major.mid` | 03:39.82 | [检索](https://musescore.com/sheetmusic?text=Chopin%20Nocturne%20Op%209%20No%202%20E%20Flat%20Major) | [原始乐谱页](https://musescore.com/user/6662591/scores/4383881) |
| Gymnopedie No. 1 - Erik Satie | `midi/gymnopedie-no-1-satie.mid` | 04:22.28 | [检索](https://musescore.com/sheetmusic?text=Gymnopedie%20No%201%20Satie) | [原始乐谱页](https://musescore.com/user/19710/scores/4766391) |
| Flight of the Bumblebee - Nikolai Rimsky-Korsakov | `midi/flight-of-the-bumblebee.mid` | 01:23.44 | [检索](https://musescore.com/sheetmusic?text=Flight%20of%20the%20Bumblebee) | [原始乐谱页](https://musescore.com/nicolas/scores/437) |
| Ode to Joy - easy variation | `midi/ode-to-joy-easy-variation.mid` | 00:29.01 | [检索](https://musescore.com/sheetmusic?text=Ode%20to%20Joy%20easy%20variation) | 待补充 |
| Ode to Joy - violin | `midi/ode-to-joy-violin.mid` | 00:32.00 | [检索](https://musescore.com/sheetmusic?text=Ode%20to%20Joy%20violin) | 待补充 |
| Passacaglia - Handel/Halvorsen, easy version | `midi/passacaglia-handelhalvorsen-easy-version.mid` | 02:16.52 | [检索](https://musescore.com/sheetmusic?text=Passacaglia%20Handel%20Halvorsen%20Piano) | [原始乐谱页](https://musescore.com/user/37309912/scores/6790392) |
| 12 Variations on "Ah vous dirai-je, Maman", K. 265 - W. A. Mozart | `midi/variations-on-ah-vous-dirai-je-maman-k265300e-1781-2-french-folk-song-wolfgang-amadeus-mozart.mid` | 09:09.27 | [检索](https://musescore.com/sheetmusic?text=Mozart%2012%20Variations%20Ah%20vous%20dirai-je%20Maman%20K265) | [原始乐谱页](https://musescore.com/user/11152751/scores/9366064) |
| Prelude in C-sharp minor, Op. 3 No. 2 - Sergei Rachmaninoff | `midi/prelude-in-c-sharp-minor-opus-3-no-2-sergei-rachmaninoff.mid` | 03:30.99 | [检索](https://musescore.com/sheetmusic?text=Prelude%20in%20C%20sharp%20minor%20Opus%203%20No%202%20Rachmaninoff) | [原始乐谱页](https://musescore.com/user/2660886/scores/2101171) |

## 来源说明

- 上表的来源平台由下载者确认为 MuseScore。
- MIDI 文件本身没有保存 MuseScore 的作者账号、乐谱 ID、下载页 URL 或版权文本。表中已有的原始乐谱页由下载者另行提供，并已按页面标题与本地 MIDI 文件名逐项核对；两首 Ode to Joy 的原始页仍待补充。
- MuseScore 上同一曲目通常有多个编配版本，因此测试来源应使用表中的原始乐谱页，不要用同名搜索结果替换。
- MIDI 文件被仓库的 `*.mid` 规则忽略，不会随 Git 提交。用于复现实验时需要单独保留或分发这些输入文件。

## ONNX 试听输出

所有 ONNX 版本的音质测试统一使用本目录列出的 MIDI。本次 10 首测试集的内容签名和输出位置为：

- 测试集：`midi-1f2337644fb7`
- 总索引：`exports/midi_tests/all_models/midi-1f2337644fb7/index.json`
- v1：`exports/midi_tests/all_models/midi-1f2337644fb7/v1/`
- v2：`exports/midi_tests/all_models/midi-1f2337644fb7/v2/`

9 个兼容模型目录中的同名 WAV 使用相同的 MIDI conditioning、warm-up、释放尾音和噪声种子，可直接进行人工 A/B 试听。正式模型使用钢琴音色索引 9；只有单音色 embedding 的 smoke 基线使用索引 0。每个目录的 `manifest.json` 记录模型路径、输出时长、峰值、RMS 和复音溢出帧数。

`smoke_piano_controls.onnx` 没有固定单帧输入、显式循环状态和完整部署 JSON，不满足当前实时 ONNX 合同，因此不生成 WAV。总索引会记录该排除原因。

`current-fixed` 从 2026-07-23 起正式命名为 **v1**。为兼容既有导出和部署脚本，模型文件仍为
`exports/piano_current_fixed.onnx`，但文档、试听目录和比较报告统一使用 `v1`。版本定义和 v2
当前音质问题见[模型版本命名与 v2 音质分析](model-versions.md)。
