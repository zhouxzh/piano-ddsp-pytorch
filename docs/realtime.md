# 实时网页 MIDI 播放器

服务端运行 ONNX 神经控制模型和宿主 DDSP，将连续 PCM16 WAV 块通过 WebSocket 发送到浏览器，
由本地电脑的 Web Audio 播放。启动命令：

```bash
python scripts/realtime_midi_server.py \
  --artifacts-dir artifacts/model-suite-v1.0.1 \
  --model-id gru_ir_96_64 \
  --midi-dir midi \
  --host 0.0.0.0 --port 8765
```

浏览器支持电脑键盘、屏幕键盘、Web MIDI 力度与延音、MIDI 文件播放、循环、变速、seek、录音和
panic。模型选择器只显示实际存在且合同有效的正式模型。切换模型会停止曲谱和音频任务，清空活动
音符、循环状态、振荡器相位、噪声、混响和 GRU 状态，然后完成 warm-up 后恢复流式输出。

公网使用时应配置 HTTPS/WSS、`--access-token` 和防火墙。`/healthz` 返回已安装模型、活动连接数
和线程配置，但不暴露本地路径。
