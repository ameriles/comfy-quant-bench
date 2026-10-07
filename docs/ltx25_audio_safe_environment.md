# LTX 2.5 audio-safe environment

This environment is intentionally separate from the working ComfyUI virtual
environment. It is the conversion environment for the `ltx25-audio-safe`
preset in `tools/quant_misto_w4a8_int8.py`.

## Pinned stack

- Python: `3.10.12`
- PyTorch: `2.9.1+rocm6.4`
- HIP runtime reported by PyTorch: `6.4.43484-123eb5128`
- comfy-kitchen: `0.2.35`
- safetensors: `0.8.0`
- NumPy: `2.2.6`
- packaging: `26.3`
- ComfyUI reference commit: `73c9bad4d21e7addbe1d13bc92eee0f1431b017d`

The full direct and transitive package set is recorded in
`tools/requirements_ltx25_audio_safe.lock.txt`.

## Node-local setup

From `/home/agustin/Projects/comfy-quant-bench`:

```bash
/home/agustin/Projects/ComfyUI/.venv/bin/uv venv \
  --python /usr/bin/python3 \
  .venv

/home/agustin/Projects/ComfyUI/.venv/bin/uv pip install \
  --python .venv/bin/python \
  -r tools/requirements_ltx25_audio_safe.lock.txt
```

The converter imports ComfyUI's `comfy.quant_ops`. On this node it is exposed
to the isolated checkout through an ignored symlink:

```bash
ln -s /home/agustin/Projects/ComfyUI ComfyUI
```

This does not install into, write to, or otherwise modify the existing ComfyUI
checkout or its `.venv`. The repository's allowlist-style `.gitignore` keeps
both `.venv/` and the `ComfyUI` symlink out of commits.

## Validation before conversion

```bash
.venv/bin/python - <<'PY'
import torch
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(TensorWiseINT8Layout)
PY

(cd tools && ../.venv/bin/python -m unittest test_quant_misto_w4a8_int8.py)
```

Run the converter with `--dry-run` before any real build. The dry run must
report exactly 912 audio INT8 layers and 528 W4A8 layers for this preset.
