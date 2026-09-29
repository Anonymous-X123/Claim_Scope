# Third-party notices

This repository does not vendor model weights or third-party package source
code. Optional local inference uses separately installed packages listed in
`requirements-gpu.txt`; those packages remain subject to their respective
licenses.

The frozen experimental outputs in `results/runs/` were produced with
`Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218`. The model card identifies the
model license as Apache 2.0:

https://huggingface.co/Qwen/Qwen3-8B

The model weights are not included in this repository. Run records retain the
model identifier and revision so that their provenance is explicit.
