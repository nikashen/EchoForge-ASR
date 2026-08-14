# Third-Party Boundaries

EchoForge-ASR keeps model code, model weights, public audio, and evaluation
reports separate. No model weight or raw evaluation recording is committed to
this repository or bundled into a release.

The optional runtime integrations are independently licensed and must be
installed and audited in the environment where they are used:

- `sherpa-onnx`: Apache-2.0 code. A model's weight license and training-data
  provenance are separate claims and must be recorded in its model manifest.
- `faster-whisper`: MIT code; its CTranslate2 and model artifacts retain their
  own notices and licenses.
- AISHELL-1/OpenSLR: downloaded by the opt-in evaluation script only. The
  downloaded archive, manifest, and license text remain outside the repository.

The default CI and Pages demo use a deterministic test backend and publish no
real speech-recognition score.
