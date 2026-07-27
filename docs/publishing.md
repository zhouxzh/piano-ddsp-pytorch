# GitHub 与 Hugging Face 发布流程

## 仓库职责

- GitHub `zhouxzh/piano-ddsp-pytorch`：源码、训练和评测程序、文档、CI、Issue 及源码标签。
- Hugging Face `zhouxzh/piano-ddsp-ascend310`：经过验证的 ONNX、未来经过下游实机验证的
  Ascend 310B OM、相邻合同 JSON、可续训 checkpoint、模型卡、验证报告和校验清单。

模型文件只在 HF 维护权威副本。两个仓库使用相同的版本名，例如
`model-suite-v1.0.0`。发布后禁止覆盖标签；修正模型或合同必须发布新版本。

## 首次建仓

公开的 **Model** 仓库为 `zhouxzh/piano-ddsp-ascend310`。checkpoint、ONNX、模型参数和
未来 OM 衍生物使用 CC BY-NC-SA 4.0，仅限非商业用途。发布包同时保留上游实现的 Apache-2.0
全文和 `THIRD_PARTY_NOTICES.md`；本代码仓库新增代码仍按根目录的 MIT 声明发布。

仓库名绑定 Ascend 310 产品系列，但不绑定中间或最终文件格式。当前
`model-suite-v1.0.0` 只发布本仓库完成 CPU/ONNX Runtime 验证的 ONNX 转换输入，
硬件合同仍明确限定为 Ascend 310B，`om_status` 保持 `pending`。未来 OM 必须由下游部署仓库
完成指定 CANN 版本、Ascend 310B 算子、内存、数值和实时性能验证后，以新标签和独立机器可读
合同发布；不得覆盖现有 ONNX 标签，也不得沿用 ONNX 验证结果宣称 OM 已验证。

后续 Ascend 310 新型号可以继续使用同一 HF 仓库，但必须使用新的发布标签，并独立记录芯片型号、
CANN 版本、输入输出合同、算子、内存、数值误差和实时性能；不能根据仓库名称推断兼容性。

本机使用当前 `hf` CLI，而不是已弃用的 `huggingface-cli`：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf auth login
hf auth whoami
```

服务器的数据集下载始终使用 `HF_ENDPOINT=https://hf-mirror.com`，不经过 Clash。模型仓库写入、
标签创建和发布后回读必须使用 `HF_ENDPOINT=https://huggingface.co`；官方端点直连超时时，仅让
约 30 MB 的模型发布流量使用本机 `127.0.0.1:7890` 代理。

## 每次发布

1. 运行 `scripts/prepare_release.py`，生成 `artifacts/model-suite-vX.Y.Z/`。
2. 运行单元测试，检查 `VALIDATION.md`，并执行 `sha256sum -c SHA256SUMS`。
3. 提交 GitHub 源码并建立 `model-suite-vX.Y.Z` 注释标签。
4. 运行 `scripts/stage_hf_release.py` 生成上传目录。它会复制完整模型发布目录，将
   `releases/huggingface-model-card.md` 转为模型仓库的 `README.md`，并加入许可证和第三方声明。
5. 将生成的目录上传到 HF `main`。
6. 在 HF 当前提交上创建与 GitHub 相同的不可变标签。
7. 从 HF 标签下载到一个空目录，重新运行 SHA-256 检查和四模型 ONNX Runtime 冒烟测试。
8. 推送 GitHub 标签并发布说明，其中只链接 HF 标签，不再重复上传模型附件。

创建仓库并确认完整 Repo ID 后，可执行：

```bash
python scripts/stage_hf_release.py \
  --repo-id zhouxzh/piano-ddsp-ascend310
HF_ENDPOINT=https://huggingface.co hf upload zhouxzh/piano-ddsp-ascend310 \
  artifacts/hf-upload/model-suite-v1.0.0 \
  --commit-message "Release model-suite-v1.0.0"
HF_ENDPOINT=https://huggingface.co hf repos tag create \
  zhouxzh/piano-ddsp-ascend310 model-suite-v1.0.0 \
  --message "Verified four-model ONNX suite"
HF_ENDPOINT=https://huggingface.co hf download zhouxzh/piano-ddsp-ascend310 \
  --revision model-suite-v1.0.0 \
  --local-dir artifacts/hf-verify/model-suite-v1.0.0
```

上传目录只能由 staging 脚本从已验证发布目录生成。不要把 MAESTRO 数据、本地 MIDI、训练日志、
试听 WAV、缓存或服务器绝对路径上传到模型仓库。
