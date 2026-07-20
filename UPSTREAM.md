# Upstream Provenance

The model architecture in `ddsp_piano/` began from the PyTorch DDSP-Piano
reference implementation:

- Repository: https://github.com/ytsrt66589/ddsp-piano-pytorch
- Source revision: `2c9e17aa0c179e2c5dd6e9bdf2d78ab7cb0b9ee5`

This package adds a TensorFlow-free MAESTRO loader, CUDA-aware training entry
point, checkpoint handling, and device-placement fixes. The original upstream
checkout remains under the parent project's `_upstream/` directory.
