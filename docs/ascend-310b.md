# Ascend 310B 交接合同

本仓库的终点是经过 CPU/ONNX Runtime 验证的 ONNX 和相邻 JSON，不是 OM。固定合同：

- FP32、opset 13、batch 1、每次一个 250 Hz 控制帧。
- 16 kHz，每帧对应 64 个音频采样，最多 16 个复音槽。
- 输入包含 conditioning、pedal、piano_model、extended_pitch 和两个显式 GRU state。
- 输出包含合成控制、IR 或 FDN 控制及下一帧 GRU state。
- 一秒 MIDI release、谐波相位、滤波噪声 FFT 和混响在宿主端维护。
- 不使用 BF16；目标部署只考虑 FP16 或 FP32。

下游 Ascend 仓库必须针对实际 CANN 版本完成 ATC 转换、算子支持、内存、CPU/OM 数值对比、
P95/P99/P99.9 延迟、声部分配、踏板、跨块连续性和音频 underrun 测试。ONNX Runtime 成功不代表
Ascend 部署完成。
