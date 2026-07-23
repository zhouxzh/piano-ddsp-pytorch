# `_upstream` 仓库审查与借鉴建议

审查日期：2026-07-21

## 优先级概览

| 优先级 | 参考 | 最值得借鉴的内容 | 不能直接照搬的部分 |
|---|---|---|---|
| A | `ddsp-piano` | 钢琴音色结构、FiLM 上下文、联合调律/非谐性、FDN、warm-up、单阶段训练结论 | TensorFlow 图、动态/复数 FFT 路径不能直接用于 310B |
| A | `ddsp-vst` | 后台推理、环形缓冲、持续相位、C++ 谐波/噪声合成、重采样；模型已由项目实测可直接转 OM | MIDI synth 路径基本是单音；仍需为钢琴实现复音宿主 |
| A | `ddsp_pytorch`（ACIDS） | GRU cache、相位 cache、块式推理、导出时分离混响 | 模块内部可变 buffer 不适合当前显式 ONNX 状态合同 |
| A | `realtimeDDSP` | 谐波前值、噪声 overlap、混响 tail 的完整流式状态思路 | Python 控制、FFT/YIN、模块状态突变不能直接进入 Ascend 图 |
| B | `ddsp-realtime` | 可复用 C++ core、CMake、推理与 DSP 解耦 | 性能声明需本机复现；仍围绕 TFLite 和单音模型设计 |
| B | `midi-ddsp` | 音符级表情控制和分层 MIDI 到合成参数思路 | 双向和自回归生成使用未来音符，不满足零前视实时要求，也不是 DDSP-VST 同构网络 |
| B | `Flute.onnx/json` | 扁平显式状态、基础算子分解、TFLite/ONNX 数值对齐记录 | 是 50 Hz 单音长笛控制器，不是钢琴；只作为已验证 OM 结构基线 |
| C | `ddsp-piano-pytorch` | 当前项目的历史基线和形状对照 | 移植不完整、基于旧版架构，不应继续作为实时设计来源 |
| C | `ddsp-pytorch`（sweetcocoa） | 简单 PyTorch DDSP 教学参考 | 旧、单音、离线，实时与钢琴针对性弱 |
| C | `ascend-cann-samples` | 将来可补充 ACL/ATC 工程模板 | 当前是稀疏检出，只含媒体音频样例，没有模型推理样例 |

## 1. 官方 `ddsp-piano`

- 路径：`_upstream/ddsp-piano`
- 仓库：`https://github.com/lrenault/ddsp-piano.git`
- 修订：`e868b7ccd3fe31b39132048a72561d7fcf1b465f`

这是音色架构最重要的参考。当前项目主要对应它的旧 `dafx22.gin` 结构，而上游 `maestro-v2.gin` 已有明显升级：

- 采样率从旧版 16 kHz 提升为 24 kHz。
- 96 个谐波/64 个噪声系数提升为 128/96。
- `OneHotZEncoder + ContextNetwork` 改为按钢琴类别调制的 `FiLMContextNetwork`。
- 分离的旧调律模块改为带预训练物理参数的 `JointParametricInharmTuning`。
- 单音网络改为更深的 `MonophonicDeepNetwork`。
- 静态长 FIR 混响可替换为 `MultiInstrumentFeedbackDelayReverb`，用 8 条 delay line 和早期反射描述环境。

最值得立即采纳的不是直接升级全部模型，而是两条训练经验：

1. 离线合成会在输入前加约 0.5 秒 warm-up，说明循环网络零状态启动存在可听风险，实时宿主也必须定义预热和重置。
2. 上游明确指出第二阶段频率优化通常不改善听感，推荐单阶段训练。当前项目应保留 phase 1 最佳模型并与 phase 2 做盲听，而不是默认后者获胜。

v2 的 FiLM、联合物理调律和 FDN 值得作为第二代模型实验，但每项都会改变 ONNX 合同、算子集合或宿主 DSP；必须单独做消融、ONNX 导出和 310B 内存/时延检查。

## 2. `ddsp-vst`

- 路径：`_upstream/ddsp-vst`
- 仓库：`https://github.com/magenta/ddsp-vst.git`
- 修订：`f2996e97f9469f3956a6b8e9d2d9b50b6555e1e9`

这是宿主端架构最直接的参考。关键设计包括：

- 默认不在音频线程执行模型，而由 20 ms 高精度 timer 执行推理。
- 输入和输出都经过环形缓冲，音频回调只消费已生成的块。
- 模型以 16 kHz、320 采样 hop 运行，再与宿主 44.1/48 kHz 相互重采样。
- 谐波合成器保存 `previousPhase`、上一帧 F0、幅度和谐波分布。
- 噪声使用频率采样生成 FIR，再过滤持续白噪声。
- 插件混响放在模型后由 JUCE 实时效果器处理，而不是每块运行神经 FFT 混响。

项目已经实测 DDSP-VST 模型可以直接转换为 OM，因此在已测试模型和转换配置范围内，模型转换适配不再视为风险。该结论是当前 Ascend 路径的重要基线：固定输入、单个显式 512 维状态以及 DDSP-VST 的基础算子图已经有成功证据。它不自动证明当前钢琴 ONNX 或 MIDI-DDSP 可转换，因为两者不是同一网络图。

当前项目应借鉴其线程和缓冲边界，但不能照搬其 MIDI 处理器。该实现只维护一个 `currentMidiNote` 和一个 ADSR，本质是单音；本项目需要 16 声部、延音踏板、重复击键和声部窃取。

## 3. ACIDS `ddsp_pytorch`

- 路径：`_upstream/ddsp_pytorch`
- 仓库：`https://github.com/acids-ircam/ddsp_pytorch.git`
- 修订：`9db246f48dba66e9b2133691d7abf4af6ede0279`

它在 `realtime_forward()` 中保存 GRU cache 和振荡器 phase，并在导出时把混响脉冲单独写出。这三点已经直接验证了当前项目选择“神经控制模型 + 宿主 DSP”边界的合理性。

需要注意，它通过 `copy_()` 修改模块内部 buffer。当前项目面向 ONNX/Ascend 时采用显式状态输入输出更可控，也更方便宿主决定 reset、双缓冲和多实例状态，因此应借鉴算法，不应恢复隐式可变状态。

## 4. `realtimeDDSP`

- 路径：`_upstream/realtimeDDSP`
- 仓库：`https://github.com/hyakuchiki/realtimeDDSP.git`
- 修订：`3f2f79039413fb01c1a00164b4429539c7db358e`

这是跨块 DSP 状态最完整的 Python 参考：

- `StreamHarmonic` 保存每个部分音的相位、上一帧频率和上一帧幅度，并在块内插值。
- `StreamFilteredNoise` 保留 FIR 卷积超出当前块的 overlap。
- `StreamIRReverb` 保留长 IR 卷积尾音。
- `CachedStreamEstimatorFLSynth` 同时管理分析窗口输入 cache 和输出延迟 cache。

这些状态类型应逐项映射到本项目的 C++ 宿主状态。不要直接导出这些 Python 模块：其中存在 `torch.rand`、FFT、YIN、数据依赖 Python 分支和模块 buffer 突变，不符合保守的 Ascend 310B 图要求。

## 5. `ddsp-realtime`

- 路径：`_upstream/ddsp-realtime`
- 仓库：`https://github.com/woosukji/ddsp-realtime.git`
- 修订：`6cdfb583e5e99acf02cd47dd0a327679d968242a`

该项目把 DDSP-VST 的核心拆成框架无关的 C++17 库，提供 CMake、Python binding 和 Unity 示例。它适合借鉴代码目录、公共 API、模型加载失败处理和 DSP 模块接口。

README 中的 M1 性能数据和多声部声明不是本项目的证据。模型仍是 TFLite 长笛/小提琴式控制器，固定 60 谐波、65 噪声频带和 512 状态；需先复现测试，再决定是否复用具体实现。

## 6. `midi-ddsp`

- 路径：`_upstream/midi-ddsp`
- 仓库：`https://github.com/magenta/midi-ddsp.git`
- 修订：`d7af42704a63b47267ae6a1bc0fee1ed7dc5c855`

它把 MIDI 先转换为 6 个可解释的音符级表情：

```text
volume, vol_fluc, vibrato, brightness, attack, vol_peak_pos
```

再把表情解码为逐帧 DDSP 参数。这种分层结构比仅依赖起音速度更容易控制力度、明亮度和击弦瞬态，对提高钢琴表现力有价值。

但是它的 expression generator 使用双向 GRU，并按完整音符序列自回归生成，天然读取未来信息。钢琴实时版若引入表情层，应重新训练因果模型，只使用当前/过去 MIDI、起音速度、踏板和可选的有限 lookahead。`vibrato` 对钢琴意义弱，可考虑替换为 una-corda、共鸣或击弦硬度等钢琴特定控制。

它与 DDSP-VST 只共享“预测 DDSP 合成参数”的大框架，神经网络并不相同。MIDI-DDSP 默认还包含表情 BiGRU、两层自回归 GRU、F0 BiGRU/两层自回归 GRU 和扩张卷积；完整逐层对比见 [DDSP 网络结构对比](ddsp-network-structure-comparison.md)。

## 7. `Flute.onnx` 与 `Flute.json`

这两个文件不是仓库，但对 Ascend 导出合同有参考价值：

- 固定 FP32 输入：`state[512]`、`f0_scaled[1]`、`pw_scaled[1]`。
- 固定输出：幅度、60 谐波、65 噪声和下一状态。
- opset 11，92 个节点，没有 ONNX `GRU` 节点；循环计算已拆成 `MatMul/Sigmoid/Tanh` 等基础算子。
- JSON 记录了 3 步状态测试和 TFLite/ONNX 最大误差。

与之相比，当前钢琴 ONNX 保留 2 个 `GRU` 节点。项目已确认 DDSP-VST 模型能够直接转换为 OM，所以 `Flute.onnx` 这类显式门控分解不只是理论参考，而是当前的已验证结构基线。当前钢琴图仍需独立转换和对齐；若其 ONNX `GRU` 不被目标 CANN 接受或执行效率不理想，应优先复用这一基础算子分解方式。

## 8. 两个旧 PyTorch 参考

`_upstream/ddsp-piano-pytorch` 是当前代码的直接历史来源，修订为 `2c9e17aa0c179e2c5dd6e9bdf2d78ab7cb0b9ee5`。它适合核对旧 TensorFlow 模型的层尺寸，但原项目的数据、训练和推理路径不完整，也没有可靠的实时部署设计。

`_upstream/ddsp-pytorch` 是 sweetcocoa 的早期通用 PyTorch DDSP，修订为 `ea5f25318dd4cd22c601dd405ebc2bac8e3f4cb6`。它适合阅读基础谐波加噪声实现，实时性和钢琴针对性均低于 ACIDS 与 `realtimeDDSP`。

## 9. Ascend CANN 样例

`_upstream/ascend-cann-samples` 当前修订为 `6511a5f4a45a1f68bd5e617989e68560f2f35cd6`，但工作树使用 sparse checkout，只检出了 `cplusplus/level1_single_api/6_media/1_audio/audio_gitee`。该目录是媒体音频接口样例，不包含 ONNX 转 OM、`aclmdl` 加载执行、异步 stream、静态内存复用或性能测试范例。

因此当前内容不能支持“已参考 CANN 推理实现”的结论。后续应补齐与实际 CANN 版本匹配的模型推理、动态/静态 shape、异步执行和性能样例，再单独记录来源修订；不要用媒体音频样例代替模型部署验证。

## 建议的采纳顺序

1. 先按 `ddsp-vst` 建立音频线程、推理线程和环形缓冲边界。
2. 按 ACIDS 与 `realtimeDDSP` 实现全部显式连续状态。
3. 用当前小模型完成 ATC 和 310B 真机闭环，不先扩大网络。
4. 修正验证集和噪声评估后，对 phase 1/phase 2 做指标与盲听比较。
5. 再从官方 DDSP-Piano v2 逐项引入 FiLM、联合物理调律、FDN 或 24 kHz，每次只改变一个变量。
6. 最后评估因果的钢琴表情层，而不是直接移植 MIDI-DDSP 的双向生成器。
