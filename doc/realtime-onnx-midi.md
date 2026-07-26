# ONNX 实时 MIDI 网络试听

## 用途与边界

`scripts/realtime_midi_server.py` 用于在远程训练服务器上实时验证 ONNX
模型。浏览器采集本地 MIDI/电脑键盘事件，服务器执行 ONNX Runtime CPU
推理和宿主 DDSP，随后把每个音频块作为独立的 mono PCM16 WAV 二进制消息
发回浏览器播放。它不转换或验证 OM，也不代表 Ascend 310B 真机已经通过。

实现参考了 GitHub 最新
[`zhouxzh/Ascend310/samples/case3`](https://github.com/zhouxzh/Ascend310/tree/main/samples/case3)
的 Note On/Off、力度、F3 到 E5 两组八度和电脑键位映射。本地固定参考位于
`references/ascend310-case3/samples/case3/`，提交为 `fb17de9dd50b`。远程
case3 不包含网络服务器或神经音频合成，因此 WebSocket、连续 ONNX 状态和
浏览器音频链路由本仓库实现。

播放器交互还参考了 `references/piano-keyboard-topic/` 中保留的四个仓库：
`Calbabreaker/piano` 的输入源去重、延音和 WebSocket 事件语义，
`sightread/sightread` 的曲谱传输与 MIDI 设备生命周期，
`scottroot/Musical-Dynamics-Training-Software` 的力度动态分级，以及
`dy/piano-keyboard` 的 active-note 与 Note Off 清理。这里只复用设计思路，
音频始终来自服务器 ONNX/DDSP 路径。

```text
local MIDI / screen keyboard
  -> JSON MIDI events over WebSocket
repository midi/*.mid
  -> server-side tempo-aware MIDI event scheduler
  -> stable 16-slot allocation + pedals + one-second release state
  -> stateful one-frame ONNX Runtime inference
  -> per-voice KeyOff envelope + stateful harmonic/noise + partitioned reverb
  -> 32 ms mono PCM16 WAV blocks over WebSocket
  -> browser queue/resampler -> local speakers / WAV recording
```

所有神经网络输入输出仍使用相邻 JSON 中的固定合同：batch 1、250 Hz
控制帧、16 个复音槽、16 kHz，以及显式 context/monophonic GRU 状态。谐波
相位、噪声 overlap、混响卷积和网络访问均在 ONNX 图外的 CPU 主机路径。

`note_off` 后，宿主对对应声部的直接谐波和噪声默认在 60 ms 内线性衰减，
其他仍按住的声部不受影响。CC64 延音踏板按下时推迟该衰减，直到踏板释放。
当所有声部门控都关闭时，总输出（包括已有混响尾音）再在 120 ms 内淡出，
保证单音松键后不会继续长时间发声；仍有其他按键时不会触发总输出淡出。
ONNX 输入侧仍保留一秒 release pitch，以保持既定导出合同和以后 Ascend
宿主实现的一致性。可用 `--keyoff-fade-ms` 和
`--all-notes-off-fade-ms` 分别调整两个时间。

## 推荐启动方式

服务器只监听 loopback，浏览器通过 SSH 转发访问：

```bash
cd /home/zhong/Documents/piano-ddsp-pytorch
conda run -n torch python scripts/realtime_midi_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --onnx-intra-op-threads 1 \
  --onnx-inter-op-threads 1 \
  --torch-threads 1 \
  --torch-interop-threads 1 \
  --keyoff-fade-ms 60 \
  --all-notes-off-fade-ms 120 \
  --model exports/piano_v1.onnx \
  --metadata exports/piano_v1.json
```

在本地电脑另开终端：

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

浏览器打开 `http://localhost:8765`。`localhost` 同时可以满足 Chromium 的
Web MIDI 和 AudioWorklet 安全上下文要求。点击“启动音频”后可选择本地 MIDI
输入，也可直接使用 88 键页面键盘或可移调的两组电脑键盘。硬件 MIDI 力度会
同时显示原始值、映射后值和 `ppp` 至 `fff` 动态级别；力度响应可选均衡、线性、
轻触、重触或固定，其中均衡模式把极端输入压缩到 32 至 112。页面中的“钢琴
音色（MAESTRO 年份）”是 ONNX
内部 `piano_model` embedding，不是 v1/v2 版本选择；实际 ONNX 版本单独显示在
页面上方。曲谱栏会列出 `midi/` 中的 `.mid`/`.midi`
文件；选择曲目并点击“播放”后，服务器根据原始 tempo、velocity、Note
On/Off 和 CC64 至 67 事件实时驱动同一个 ONNX 合成器。播放器支持暂停、继续、
拖动定位、0.5 至 2 倍速度和循环。定位或变速时，服务器重置递归/DSP 状态并从
曲谱起点快速重放控制事件到目标位置，以恢复当时的按键和踏板状态。页面的录音
功能会把收到的连续音频块合并为一个 16 kHz WAV 并在本地下载。

音频运行期间可以直接切换 MAESTRO 年份音色。切换会停止当前曲谱、清空浏览器
音频缓冲，并重新创建和预热 ONNX/DSP 状态；短暂停顿是为避免不同音色之间共享
GRU、振荡器相位或混响历史。

启动时可用 `--midi-dir` 指定其他曲谱目录。服务只接受启动时目录清单中的
相对 ID，不接受浏览器提供的任意文件路径。曲谱超过 16 复音时沿用实时输入
的稳定槽位和声部窃取规则，因此高复音曲目的听感应同时检查页面的声部状态。

默认运行正式 v1。测试 v2 时使用：

```bash
conda run -n torch python scripts/realtime_midi_server.py \
  --model exports/piano_v2.onnx \
  --metadata exports/piano_v2.json
```

两个版本不能在同一端口同时启动。使用不同端口即可并行人工 A/B，但每个
进程默认只允许一个浏览器实时会话。

## 直接网络访问

只有在已配置防火墙时才应监听公网/局域网地址。至少设置随机访问令牌：

```bash
conda run -n torch python scripts/realtime_midi_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --access-token REPLACE_WITH_A_RANDOM_TOKEN
```

访问 `http://SERVER_IP:8765/?token=REPLACE_WITH_A_RANDOM_TOKEN`。远程普通
HTTP 通常不能授权 Web MIDI；页面键盘和兼容 Web Audio 回退仍可使用。需要
Web MIDI 时优先使用 SSH 转发，或通过 `--ssl-cert` 和 `--ssl-key` 提供 HTTPS。

## 协议

浏览器到服务器的 WebSocket 文本消息为 JSON：

- `start`: `piano_model`（MAESTRO 年份音色 embedding）和 `server_gain`。
- `note_on`: MIDI `pitch` 与 `velocity`。
- `note_off`: MIDI `pitch`。
- `control_change`: CC 64 至 67 与 `value`。
- `play_midi`: `midi_id` 必须来自 `hello.midi_files`；可带
  `position_seconds`、`tempo_scale`（0.5 至 2.0）和布尔 `loop`。
- `pause_midi`、`resume_midi`：暂停或继续当前曲谱。
- `seek_midi`: 使用 `position_seconds` 定位当前曲谱。
- `set_midi_transport`: 在播放中更新 `tempo_scale` 和 `loop`。
- `stop_midi`: 停止当前曲谱并重置 ONNX/DSP 递归状态。
- `panic`, `stop`, `ping`, `set_server_gain`。

服务器到浏览器的文本消息为 `hello`、`status`、`metrics`、`pong` 或
`midi_playback`、`error`。`hello.midi_files` 提供曲谱 ID、显示名、时长和
事件数；当前 `hello.protocol_version` 为 2。`metrics.midi_playback` 和
`midi_playback` 事件提供 `state`、当前位置、速度和循环状态。每条二进制消息是一个
完整 RIFF/WAVE 文件，格式为单声道、16-bit PCM、16 kHz；默认 8 个控制帧
即 512 个采样/32 ms。浏览器初始保留约 80 ms 缓冲，并在设备采样率不是
16 kHz 时连续线性重采样。发生欠载后会依次使用 120、180、240 ms 重新蓄水；
停止、定位或重启会话会恢复 80 ms，以兼顾现场键盘延迟和曲谱连续播放。

## 性能判定

页面中的“实时系数”是服务器合成耗时除以 32 ms 块时长。持续小于 1 才能
稳定按时产生音频；“迟到块”持续增加或浏览器报告缓冲不足，表示当前 CPU
路径无法满足该配置。端到端按键延迟还包括最多一个块的事件等待、约 80 ms
浏览器缓冲、网络往返和音频设备缓冲，因此该工具用于远程音质与连续性验证，
不是最终 310B 延迟验收。

实时服务默认把 ONNX Runtime intra/inter-op 与 PyTorch intra/inter-op 线程
都限制为 1。该模型每帧张量较小，默认的全核线程池会与宿主谐波、噪声 FFT
线程池互相抢占；在 32 核开发服务器上实测可让 16 声部、8 帧块从无法实时
降到约 6 ms。可通过 `--onnx-intra-op-threads`、
`--onnx-inter-op-threads`、`--torch-threads` 和
`--torch-interop-threads` 覆盖，但必须重新检查 P95/P99、CPU 占用和迟到块。
线程设置只影响本仓库的 CPU ONNX 试听进程，不改变 ONNX 图或 Ascend 310B
部署合同。

`--chunk-frames` 可调整调度折衷。更小的值降低块等待但增加 ONNX/网络调度
开销，更大的值提高吞吐但增加演奏延迟。任何用于后续 Ascend 宿主实现的
配置都应重新测量 P95/P99 延迟、迟到块、削波和跨块连续性。
