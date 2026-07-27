# Third-Party Notices

Portions of the model, synthesis, and loss implementation were adapted from
`lrenault/ddsp-piano`, licensed under Apache-2.0. This repository replaces the
TensorFlow training path with PyTorch, adds explicit recurrent ONNX state,
separates host DSP from the deployable graph, and adds MAESTRO preprocessing,
evaluation, release packaging, and browser streaming.

The early PyTorch baseline came from `ytsrt66589/ddsp-piano-pytorch` at
revision `2c9e17aa0c179e2c5dd6e9bdf2d78ab7cb0b9ee5`. Upstream architecture
comparison used `lrenault/ddsp-piano` at revision `e868b7ccd3fe`.

DDSP concepts and host processing follow Google Magenta DDSP. No TensorFlow
checkpoint or third-party training dataset is redistributed in this source
repository. MAESTRO and local MuseScore MIDI files must be obtained separately
under their respective terms.

The published model weights were trained on MAESTRO v3.0.0, made available by
Google LLC under CC BY-NC-SA 4.0. Published checkpoints, ONNX graphs, parameters,
and later OM derivatives are consequently offered under CC BY-NC-SA 4.0 for
non-commercial use. The model release does not redistribute MAESTRO recordings
or MIDI files.
