# ComfyUI LTX 2.5 Hybrid Helpers

Two small nodes used by the MultiGPU reference workflow:

- **Free VRAM** asks ComfyUI to unload enough managed models to make a selected amount of VRAM available before decode.
- **VAE Decode (Auto Tiled)** goes directly to adaptive tiled decoding, avoiding an intentional full-decode OOM before ComfyUI falls back to tiling.

Copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.

Tested against ComfyUI `v0.37.0` (`73c9bad`). The auto-tiled node uses private VAE helpers from that ComfyUI version and may need adjustment after upstream VAE API changes.

These helpers are optional. The Simple workflow uses ComfyUI's standard `VAE Decode (Tiled)` node and does not need them.
