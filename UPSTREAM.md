# Upstream Provenance

The model architecture in `ddsp_piano/` began from the PyTorch DDSP-Piano
reference implementation:

- Repository: https://github.com/ytsrt66589/ddsp-piano-pytorch
- Source revision: `2c9e17aa0c179e2c5dd6e9bdf2d78ab7cb0b9ee5`

This package adds a TensorFlow-free MAESTRO loader, CUDA-aware training entry
point, checkpoint handling, and device-placement fixes. The original upstream
checkout is stored locally as `references/ddsp-piano-pytorch/`.

All third-party source repositories, papers, and reference models are stored
under the ignored `references/` directory. The committed
`references/README.md` is the canonical inventory. Key references include:

- ACIDS IRCAM PyTorch DDSP: `acids-ircam/ddsp_pytorch`, revision
  `9db246f48dba66e9b2133691d7abf4af6ede0279`.
- Google DDSP: `magenta/ddsp`, revision
  `cf5e62dfe5d5c80aa14761832233a2e68e840e53`.
- Google Magenta archive: `magenta/magenta`, revision
  `c15687ebd1c1cc12658cd468c737d3c24c02b918`.
- Magenta RealTime: `magenta/magenta-realtime`, revision
  `2a854047691fb96ac024aa01650a702b6cfc5f2e`.
- DDSP paper: *DDSP: Differentiable Digital Signal Processing*,
  `arXiv:2001.04643`, saved as `references/papers/ddsp-2001.04643.pdf`
  (SHA-256 `41aa87d9710abd0c66837502684d5a5f58e0cb1e34a62eb2d14e25944dbb0856`).

The ACIDS realtime model informed the explicit recurrent-state interface. The
Google VST configuration and paper informed the fixed frame-rate neural-control
plus host-DSP boundary. No TensorFlow code is used by this training package.
The expanded Magenta code, paper, and dataset inventory is recorded in
`doc/magenta-reference-materials.md`; data downloads use pinned Hugging Face
mirror revisions and never enter the deployable ONNX graph.
