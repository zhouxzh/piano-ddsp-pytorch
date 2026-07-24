# DDSP-Piano 标准化测试与自动优化

## 目标与版本基线

`quality-v1` 固定以 `exports/piano_current_fixed.onnx`（公开版本名 v1）为基线，候选模型只能写入
`exports/candidates/`。客观门禁和人工门禁都通过之后，候选才具备晋级资格；自动流程不会覆盖 v1。

本服务器只进行 PyTorch CPU、ONNX Checker、ONNX Runtime 数值一致性和 CPU 性能验证，不进行
OM 转换或 CANN 设备验证。因此报告不能单独证明 Ascend 310B 已部署就绪。ONNX 仍严格使用
FP32、opset 13、固定 batch 1、单个 250 Hz 控制帧、16 个复音槽和显式 GRU 状态；FFT 合成、
相位累积、噪声和混响继续位于宿主后处理边界。

## 固定测试集

测试集锁文件由 `scripts/evaluate_model.py prepare` 从 MAESTRO 缓存生成，包含预处理参数、每个片段
的绝对定位和内容哈希。除非明确使用 `--refresh`，已有锁文件与重新计算结果不一致时程序会失败，
避免候选之间使用不同数据。

- `quick`：validation，每年份 1 首曲目，每类 1 个三秒窗口，用于冒烟检查。
- `dev`：validation，每年份 2 首曲目；按 quiet、loud、dense、onset、sustain 分层，每类每曲 2 个
  三秒窗口，正常 MAESTRO 数据上共 200 个窗口。
- `release`：test 的全部不重叠三秒窗口，只用于最终候选。

预处理固定为 16 kHz、250 Hz、3 秒、50% 训练重叠和 16 复音。release 缓存生成前要求至少
20 GiB 可用空间。

## 客观指标与门禁

每个片段同时记录 MR-STFT、响度误差、起音包络误差、频谱质心误差、尾音衰减误差、峰值、RMS、
crest factor，以及 harmonic/noise/dry/wet 四个信号的有限性、峰值和 RMS。综合比值的权重为
50%、20%、15%、10%、5%，小于 1 表示优于 v1。

硬门禁要求：ONNX 无动态维度；只含白名单算子；PyTorch/ONNX 逐输出一致；导出时至少连续验证
100 个有状态调用；输出和各 stem 有限且非静音；1 秒与 4 秒渲染分块的最大误差不超过 `1e-4`。
ONNX 文件不得超过 16 MiB。这些检查同时应用于 v1 基线和候选，且固定部署形状、输入输出类型、
元数据和实际图必须一致。

音质门禁要求：综合比值中位数不高于 0.98；任一年份或声音类别的中位数不高于 1.05；响度误差
P95 相对 v1 增幅不超过 1 LU；ONNX CPU 延迟 P95 不高于 v1 的 1.25 倍。所有阈值均集中在
`configs/evaluation_v1.json`。

## 单次标准评测

先固定 dev 测试集，再运行候选：

```bash
conda run -n torch python scripts/evaluate_model.py prepare \
  --profile dev \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0

conda run -n torch python scripts/evaluate_model.py run \
  --baseline exports/piano_current_fixed.onnx \
  --candidate exports/candidates/example.onnx \
  --profile dev \
  --midi-dir midi
```

输出采用语料、基线和候选 SHA-256 寻址，包含 `report.json`、`report.md`、`metrics.csv`、盲听页面、
PCM16 音频和四类诊断 stem。`report.json` 是机器判定依据，`report.md` 用于快速阅读。
逐片段指标另按模型、语料、渲染和指标配置缓存；因此同一固定 v1 基线不会为每个候选重复渲染。
报告目录和缓存统一位于 Git 忽略的 `evaluations/`。

## 盲听与人工超时

dev/release 会从 `midi/` 中每个 MIDI 自动选择 30 秒信息量较高的片段，并各自产生固定增益和
-23 LUFS 响度匹配两组 A/B。A/B 身份由稳定哈希随机化，映射只保存在 `private/blind_mapping.json`。
人工需要评价偏好、音色、起音、力度、延音、混响、纯净度和严重缺陷。

打开报告目录中的 `listening/index.html`，完成后页面下载 `listening_scores.json`。将该文件放回同一
`listening/` 目录。候选需要至少 60% 偏好率，各维度平均回退不超过 0.25，且严重缺陷少于 2 次。

默认人工窗口为 30 分钟，可在 `configs/evaluation_v1.json` 修改，也可在质量循环命令中传入
`--review-timeout-minutes`。到期仍无评分时：

1. 报告状态从 `pending` 变为 `deferred`，盲听页面、音频和私有映射全部保留。
2. 候选不能晋级，但自动流程立即继续训练和评测下一候选。
3. 之后仍可将评分文件放回并执行下面的命令；迟到评分会标记在报告中并正常重新计算门禁。

```bash
conda run -n torch python scripts/evaluate_model.py finalize \
  --report-dir evaluations/path/to/report \
  --scores evaluations/path/to/report/listening/listening_scores.json
```

重新运行质量循环会读取已有检查点和报告，并把迟到的合格评分同步到循环汇总。

## 自动训练循环

`configs/v2_quality_cycle.json` 定义最多四个 v2 消融候选：FDN/IR 混响、128/96 或 96/64 谐波与
噪声维度，并加入 0.10 能量损失和 0.05 起音损失。流程可中断恢复：

1. 四个候选训练到 8,000 步，用 dev 客观报告筛选前两名。
2. 前两名训练到 20,000 步，再次用 dev 报告排序。
3. 第一名训练到 40,000 步，运行完整 release 报告并等待人工评分。
4. 第一名客观失败、人工失败或人工超时后，第二名继续到 40,000 步并执行相同流程。

启动完整循环：

```bash
conda run -n torch python scripts/run_quality_cycle.py --device cuda
```

如需交给 user systemd 自动重启：

```bash
systemd-run --user --unit=ddsp-piano-quality-cycle \
  --property=Restart=on-failure --property=RestartSec=60s \
  --working-directory="$PWD" \
  conda run -n torch python scripts/run_quality_cycle.py --device cuda
```

循环状态原子写入 `runs/quality_cycle/<cycle-id>/state.json`，日志位于同目录 `logs/`，最终汇总为
`cycle_summary.json` 和 `cycle_summary.md`。每个候选的 ONNX、JSON 合约与测试报告都有独立路径。
同一循环通过文件锁避免重复进程；命令中断后原样重跑即可恢复。

## 晋级规则

只有 release 客观门禁和人工门禁同时通过时，循环状态才是 `promotion_ready`。自动程序只记录
`promotion_candidate`，不会复制、重命名或覆盖官方 v1。人工超时结束时使用
`awaiting_deferred_review`，明确区别于模型失败。正式更换版本仍需审阅报告后单独执行。
