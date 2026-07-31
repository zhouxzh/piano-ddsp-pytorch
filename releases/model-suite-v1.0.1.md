# model-suite-v1.0.1

结构化命名补丁版本，同时发布 `gru_ir_96_64`、`film_fdn_128_96`、`gru_ir_fullwet_96_64` 和
`film_ir_fullwet_96_64` 的 ONNX、合同 JSON 和可续训 PyTorch checkpoint。

权重张量与 `model-suite-v1.0.0` 相同；本版本只迁移公开模型 ID、文件名和机器可读合同。
旧标签保持不可变。模型附件发布在 Hugging Face 模型仓库 `zhouxzh/piano-ddsp-ascend310` 的同名标签
`model-suite-v1.0.1`。GitHub 的同名标签只固定源码和发布说明。

checkpoint、ONNX、模型参数和未来 OM 衍生物使用 CC BY-NC-SA 4.0，仅限非商业用途；发布包
同时保留上游实现所需的 Apache-2.0 许可证与第三方声明。

四个模型均通过 PyTorch CPU、ONNX checker 和 100 帧状态连续的 ONNX Runtime 数值对比。
模型名称描述结构，不表示质量排序。OM/CANN 转换和 Ascend 310B 实机测试尚未执行。

下载全部附件后运行：

```bash
sha256sum -c SHA256SUMS
```

详细合同和各附件哈希见 `model-suite.json`。
