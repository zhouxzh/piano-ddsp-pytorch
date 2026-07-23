# Ascend 310B 实时落地路线与验收标准

## 推荐运行时边界

```mermaid
flowchart LR
    MIDI[MIDI callback] --> Q1[无锁事件队列]
    Q1 --> SCH[250 Hz 调度器]
    SCH --> VA[16 声部/踏板/release 状态机]
    VA --> Q2[固定输入双缓冲]
    Q2 --> NPU[Ascend 控制模型]
    NPU --> Q3[控制量环形缓冲]
    Q3 --> DSP[C++ 流式谐波 + 噪声 + 混响]
    DSP --> RS[16 kHz 到宿主采样率重采样]
    RS --> OUT[音频 callback]
```

音频 callback 不应加载文件、申请内存、等待 mutex、执行网络同步调用或重建 FFT plan。模型切换和 IR 更新也应在非实时线程完成，然后通过版本化指针或双缓冲原子切换。

## 必须持久化的状态

| 状态 | 建议形状/内容 | Reset 条件 |
|---|---|---|
| Context GRU | `[1,1,64]` | 启动、panic、模型重载 |
| Monophonic GRU | `[1,16,192]` | 对应声部重置或全局 reset |
| Release | 每声部 held pitch + released frame count | 声部重置、panic |
| Voice allocator | pitch、key-down、pedal-held、age、generation | panic、MIDI 设备切换 |
| Harmonic phase | 至少 `[16,2,96]` | 单声部重新分配、采样率变化 |
| Previous controls | 每声部前一帧 F0、幅度、谐波分布 | 单声部重新分配 |
| Noise generator | 每声部 PRNG state | 可复现 seed 或声部 reset |
| Noise FIR | 每声部 filter/overlap tail | filter 大小变化、reset |
| Reverb | 分区卷积或 FDN delay state | 音色/IR 切换策略决定 |
| Resampler | 输入/输出滤波器历史 | 采样率或设备变化 |

声部重新分配时必须使用 generation 标识，防止迟到的 NPU 输出写回已经分配给新音符的槽位。

## 实施阶段

### 阶段 1：修正离线验收基础

- 建立跨曲目、跨年份、跨复音区间的固定验证清单。
- 修复 eval 噪声在声部间相同的问题。
- 新增“完整 750 帧前向”与“750 次单帧显式状态前向”的控制量对齐测试。
- 固定一组短 MIDI，保存 dry harmonic、noise、wet 和最终音频回归结果。
- 同时评估 phase 1 与 phase 2，不让流水线自动替代听感判断。

### 阶段 2：实现 CPU 流式参考

- 在 C++ 或严格固定状态的 Python reference 中实现持续相位谐波合成。
- 实现独立声部 PRNG 和噪声 FIR overlap-add/save。
- 实现可持续尾音的分区卷积，或将官方 FDN 作为另一候选。
- 实现与 [`_pack_polyphony`](../ddsp_piano/maestro.py#L209) 训练编码一致的实时声部分配器，并定义大于 16 复音的窃取策略。
- 明确 MIDI sample offset 到 250 Hz 控制帧的量化规则。

此阶段先用 ONNX Runtime CPU；只有 CPU 参考连续、可听且可复现，才进入 NPU 集成。

### 阶段 3：收紧 ONNX 合同

- 将 `reverb_ir` 从每帧模型拆出，改为预设初始化时查询一次。
- 保持 batch 1、frame 1、polyphony 16 和全部静态维度。
- 将项目已经成功转换为 OM 的 DDSP-VST 模型、转换命令和工件作为 golden baseline，保留其输入输出及算子清单。
- 测试目标 CANN 是否接受当前钢琴 ONNX 的两个 `GRU`；若不接受或性能不达标，沿用 DDSP-VST 已验证思路，导出显式 `MatMul/Add/Sigmoid/Tanh/Mul` GRU cell。
- 记录每个输入输出的 dtype、字节数、对齐要求和所有算子。
- FP32 先通过，再选择性评估 FP16；不增加 BF16 路径。

### 阶段 4：CANN/ATC 与真机

- 固定并记录 CANN、驱动、固件、ATC 和 opset 版本。
- 将当前钢琴 ONNX 独立转换为 OM，保存完整 ATC 日志和实际落图信息；不能用 DDSP-VST 的转换成功代替本模型验证。
- 逐输出比较 PyTorch CPU、ONNX Runtime CPU 和 Ascend 310B，至少连续运行 750 帧。
- 使用预分配 device/host buffer，避免每帧 malloc/free。
- 在推理线程使用异步执行和双缓冲；音频线程只读取已完成的控制块。
- 测量 H2D、NPU、D2H、DSP 和重采样各阶段耗时，不只测模型 kernel。

### 阶段 5：插件/应用集成

- 支持 44.1 kHz 和 48 kHz 宿主采样率及常见 64/128/256/512 buffer。
- 支持 note on/off、重复击键、sustain、sostenuto、soft pedal、panic 和设备热切换。
- 定义 NPU 超时策略：保持上一控制帧、平滑衰减或旁路，不能阻塞音频线程。
- 音色切换时对 GRU、IR 和 DSP 状态做可控 crossfade。

## 性能预算

当前控制周期是 4 ms。建议在满 16 声部、踏板开启和宿主并发负载下使用以下门槛：

| 项目 | 建议门槛 |
|---|---:|
| 神经推理 + 传输 P99 | 小于 2 ms |
| 完整控制与 DSP P99 | 小于 3.2 ms |
| 单次最坏耗时 | 不超过 4 ms；长测期间不得持续超期 |
| 音频 callback | 0 次阻塞锁、0 次堆分配、0 次文件/网络 I/O |
| 连续压力测试 | 至少 30 分钟、0 underrun、0 NaN/Inf |

这些是工程目标，不是当前实测结果。若 310B 单帧启动开销过大，可以评估一次处理固定 2 或 4 个控制帧，但会引入 8/16 ms 调度粒度，必须同时重新评估演奏延迟和 ONNX 固定合同。

## 数值验收

1. PyTorch CPU 与单帧状态 PyTorch：控制量 `allclose`。
2. PyTorch CPU 与 ONNX Runtime CPU：连续多帧最大误差在记录阈值内。
3. ONNX Runtime CPU 与 Ascend FP32：逐输出和逐状态比较。
4. 若使用 FP16，单独给出控制误差、合成音频 SI-SDR/频谱误差和盲听结果。
5. 连续 DSP 与离线 DSP 对比时忽略明确记录的启动/尾部边界，不得忽略块边界点击。

## 音质验收

- 完整 MAESTRO test 的多尺度频谱损失，按年份和复音分桶报告。
- 至少覆盖弱音、强音、快速重复音、密集和弦、长 sustain 和高音区。
- 固定增益的 phase 1/phase 2 双盲 A/B，不做逐文件峰值归一化掩盖响度问题。
- 单独试听 harmonic、noise、dry 和 wet，便于定位噪声相干、相位点击和混响尾音问题。
- 16 kHz 与候选 24 kHz 模型必须在相同 MIDI、相同主观响度下比较。

## “可部署”完成定义

只有同时满足以下条件，才可以将模型标记为 Ascend 310B 实时可部署：

- 固定合同文档与实际 OM 输入输出一致。
- CPU、ONNX、310B 连续状态数值验证通过。
- 所有 DSP 跨块状态实现并通过连续性测试。
- 满载长测达到时延和 underrun 门槛。
- 模型 reset、声部窃取、踏板、超时和音色切换行为已定义并测试。
- 最终 checkpoint 通过跨曲目指标与听感验收。

当前项目尚未满足以上完成定义，主要缺口是宿主实时引擎与 CANN/真机验证。
