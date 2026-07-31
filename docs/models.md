# 四模型定义

`model-suite-v1.0.1` 固定使用 `gru_ir_96_64`、`film_fdn_128_96`、`gru_ir_fullwet_96_64` 和
`film_ir_fullwet_96_64`。名称只编码部署相关结构：控制网络类型、宿主混响类型、谐波数和噪声带数；
`fullwet` 表示训练合同中的 IR wet gain 为 1.0。

正式默认模型为 `gru_ir_96_64`。这是此前人工试听中较稳定的 `current_fixed/v1` 所对应的结构；
现有评测尚不足以宣称它在所有曲目和所有指标上绝对优于另外三种结构。

已经发布的 `model-suite-v1.0.0` 标签保持不可变；`model-suite-v1.0.1` 使用完全相同的权重张量，
只迁移公开 ID、文件名、checkpoint 身份字段和 ONNX 相邻合同。

`paper_ir`、`film_fdn`、`calibrated_ir`、`calibrated_film_ir` 以及更早的 `v1`、`v2`、`v2a`、
`v2b` 均不再是公开接口；程序遇到旧 ID 会报错并提示迁移目标：

| 历史 ID | 正式 ID |
| --- | --- |
| `paper_ir` / `v1` | `gru_ir_96_64` |
| `film_fdn` / `v2` | `film_fdn_128_96` |
| `calibrated_ir` / `v2a` | `gru_ir_fullwet_96_64` |
| `calibrated_film_ir` / `v2b` | `film_ir_fullwet_96_64` |

`gru_ir_96_64` 和 `film_fdn_128_96` 分别对应上游 `dafx22.gin` 与 `maestro-v2.gin` 的本地 PyTorch
训练实现，不是上游 TensorFlow checkpoint 的格式转换。两个 `fullwet` 模型是本仓库修改并训练的
结构。四者的权重均来自本仓库 MAESTRO 训练。

已经完成的历史训练目录和评测报告仍可能包含旧 ID；这些内容作为实验来源记录保留，不是可发布的
模型接口。发布目录、Hugging Face 暂存目录、ONNX 相邻 JSON 及新生成的试听输出必须使用正式 ID。

每个模型在 Hugging Face 模型仓库中包含：

- `ddsp_piano_<id>.onnx`：供 ONNX Runtime 验证和后续 OM 转换的控制模型。
- `ddsp_piano_<id>.json`：输入输出 shape、dtype、算子、哈希、DSP 边界和数值对比合同。
- `ddsp_piano_<id>.pt`：清理过路径的 PyTorch 可续训 checkpoint。

同一 HF 版本另含 `model-suite.json`、`VALIDATION.md` 和 `SHA256SUMS`。所有模型状态固定为
`onnx_status=verified`、`om_status=pending`、`quality_status=quality_selection_pending`。

`gru_ir_96_64`、`gru_ir_fullwet_96_64` 和 `film_ir_fullwet_96_64` 输出 `reverb_ir`；`film_fdn_128_96` 输出九维
`reverb_controls`。这个差异由相邻 JSON 明确记录，宿主不能把两种输出互换。

模型二进制只以 HF 仓库的带标签版本为权威发布源。GitHub 仓库发布源码、训练和验证程序，并使用
相同的 `model-suite-v1.0.1` Git 标签指向对应代码，不重复托管一套模型附件。

checkpoint、ONNX、模型参数和未来 OM 衍生物统一使用 CC BY-NC-SA 4.0，仅限非商业用途。
上游 DDSP-Piano 实现的 Apache-2.0 全文和修改声明随模型发布包保留。
