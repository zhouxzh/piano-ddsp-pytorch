# model-suite-v1.0.0

首个稳定模型套件，同时发布 `paper_ir`、`film_fdn`、`calibrated_ir` 和
`calibrated_film_ir` 的 ONNX、合同 JSON 和可续训 PyTorch checkpoint。

四个模型均通过 PyTorch CPU、ONNX checker 和 100 帧状态连续的 ONNX Runtime 数值对比。
模型名称描述结构，不表示质量排序。OM/CANN 转换和 Ascend 310B 实机测试尚未执行。

下载全部附件后运行：

```bash
sha256sum -c SHA256SUMS
```

详细合同和各附件哈希见 `model-suite.json`。
