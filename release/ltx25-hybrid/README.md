---
license: other
license_name: LTX-2.x Community License Agreement
license_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
base_model: Lightricks/LTX-2.5
pipeline_tag: text-to-video
library_name: comfyui
tags:
  - ltx-video
  - ltx-2.5
  - comfyui
  - quantization
  - w4a8
  - int8
  - convrot
  - rocm
  - amd
---

# LTX 2.5 22B Distilled — Hybrid W4A8 / INT8 ConvRot for ComfyUI

Community quantization pack of the official LTX 2.5 22B Distilled transformer and Gemma 4 12B text encoder, prepared by **Agustín Meriles (ameriles)** for ComfyUI.

The transformer uses a mixed recipe designed to retain audio/lip-sync behavior and visually sensitive projections while keeping most of the speed and memory advantages of W4A8:

- **900 Linear layers:** INT8 tensorwise + ConvRot
- **540 Linear layers:** asymmetric W4A8
- all other tensors: preserved from the official BF16 source

The text encoder uses asymmetric W4A8 for 328 tensors and preserves 358 tensors.

## Files

| File | Destination | Size |
| --- | --- | ---: |
| `ltx-2.5-22b-distilled-hybrid-w4a8-int8-convrot.safetensors` | `ComfyUI/models/diffusion_models/` or `ComfyUI/models/unet/` | 15.88 GiB |
| `gemma4-12b-ltx-2.5-w4a8.safetensors` | `ComfyUI/models/text_encoders/` | 9.88 GiB |
| `*.quant.json` | Optional provenance sidecars; keep beside the matching model if desired | — |
| `workflows/*.json` | ComfyUI workflows | — |

The VAE, Audio VAE and latent upscalers are **not redistributed here**. Download them from the official [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) repository.

## Why this hybrid recipe?

An all-W4A8 transformer was very fast on RDNA2, but testing showed weaker speech/lip sync and occasional visual degradation. The final recipe starts from an audio-balanced hybrid and promotes four visually sensitive families to INT8:

- video blocks: `attn1.to_v`, `attn2.to_v`, `ff.net.2`
- video connector: `attn1.to_v`, `ff.net.2`

The audio-balanced base keeps the validated audio and cross-modal families in INT8. The exact selector, strict expected layer counts and tests are published in [`ameriles/comfy-quant-bench`](https://github.com/ameriles/comfy-quant-bench).

This is an empirical quality/performance trade-off, not a claim that one precision is universally best.

## Compatibility

- ComfyUI-only prequantized Safetensors format
- tested with ComfyUI `v0.37.0` (`73c9bad`)
- tested with PyTorch `2.9.1+rocm6.4`
- converted with Comfy Kitchen `0.2.35`
- validated on 3 × AMD Radeon RX 6700 XT (RDNA2)

The formats are not exclusive to RDNA2. Comfy Kitchen provides multiple runtime backends, but performance and kernel availability depend on the installed ComfyUI, PyTorch, backend and GPU. This pack has only been performance-tested on the setup above.

### Important text-encoder note

The W4A8 text encoder materially reduces file size and model storage. In stock ComfyUI, text-encoder execution may dequantize the weights because the CLIP path requests full-precision matrix multiplication and forced weight casting. Therefore, treat the text encoder as a **memory/storage optimization first**, not as a guaranteed native W4A8 speed-up.

## Workflows

Two cleaned workflows are included:

- `LTX2.5_Hybrid_W4A8_INT8_MultiGPU.json`: the measured 3-GPU layout, using ComfyUI-MultiGPU plus the included Free VRAM and Auto Tiled VAE helper nodes.
- `LTX2.5_Hybrid_W4A8_INT8_Simple.json`: standard ComfyUI loaders and tiled VAE decode, with no MultiGPU or memory-helper dependency.

The Simple workflow is portable, but it does not imply that the full pipeline will fit entirely in a 12 GB GPU. ComfyUI may rely on CPU offload, and speed depends heavily on VRAM and system RAM.

The MultiGPU reference allocation is:

- transformer: `cuda:1,34%;cuda:2,66%`
- text encoder: `cuda:0,50%;cpu,50%`
- video VAE: `cuda:0`

The public workflows remove the experimental QwenTTS branch, inactive GGUF loaders and inactive LoRAs from the development workflow.

## Setup

1. Install/update ComfyUI with native quantized Safetensors support and Comfy Kitchen.
2. Place the transformer and text encoder in the directories shown above.
3. Download these official LTX 2.5 components as required by the workflow:
   - `ltx-2.5-video-vae-conv-bf16.safetensors`
   - `ltx-2.5-audio-vae-bf16.safetensors`
   - `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`
   - `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`
4. For the MultiGPU workflow, install [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) and copy the included `ComfyUI-LTX25-Hybrid-Helpers` folder into `ComfyUI/custom_nodes/`.
5. Load one of the workflows, select your own image/audio inputs and adjust device allocations for your hardware.

## Measured result on the reference setup

For the tested 5-second, two-stage talking-head workflow, four warm runs completed in **461.16–465.90 seconds** (about **7:43 average**). Earlier Q4_K_M runs on the same evolving workflow averaged about **13:13**. This is a setup-specific observation, not a universal benchmark: workflow revisions, thermal state, PCIe topology, prompts and input media all affect end-to-end time.

Qualitative acceptance covered prompt following, Spanish speech, lip sync, teeth/facial detail and repeated 5-second generations. Some prompt/seed combinations can still produce motion or facial artifacts, as with the BF16/GGUF baselines.

## Quantization provenance

Source files from `Lightricks/LTX-2.5`:

- `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
- `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`

Transformer conversion:

- preset: `ltx25-q4km-audio-balanced-visual-sensitive`
- output: 17,045,068,544 bytes
- INT8 vs BF16 sampled relative-error median: 0.0094466
- INT8 vs BF16 sampled relative-error maximum: 0.0110030

Text-encoder conversion:

- format: `asym_w4a8_int8`
- group size: 16
- ConvRot group size: 256
- output: 10,604,318,782 bytes

See the JSON sidecars for machine-readable details and `SHA256SUMS` for integrity hashes.

## Limitations

- ComfyUI-only; these files are not intended for the standard `ltx-pipelines` PyTorch loader.
- Performance is validated on ROCm/RDNA2 only.
- The final release recipe was selected through qualitative testing rather than a formal perceptual benchmark.
- Image-to-video can preserve identity well but may show more mouth/body deformation than pure text-to-video, especially with distant subjects or strong motion.
- Ten-second generation exceeded VRAM during the second sampler on the 3 × 12 GB reference system; five-second generation is the validated target.

## License and modification notice

This is a quantized derivative of LTX 2.5 and is distributed under the **LTX-2.x Community License Agreement**. Read the complete included [`LICENSE-LTX-2.x`](LICENSE-LTX-2.x), including its commercial-use conditions and use restrictions, before downloading or using the files.

The original BF16 files were modified by quantization in October 2026 by Agustín Meriles. Lightricks did not produce or endorse this quantization.

## Credits

- Base model: [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
- Quantization tooling and research base: [JoaoZaokk/comfy-quant-bench](https://github.com/JoaoZaokk/comfy-quant-bench)
- Quantization, testing and release: **Agustín Meriles / [ameriles](https://github.com/ameriles)**
