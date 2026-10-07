import torch

import comfy.model_management as model_management


class FreeVRAMForPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return {
            "required": {
                "latent": ("LATENT",),
                "device": (devices,),
                "free_gb": (
                    "FLOAT",
                    {"default": 9.0, "min": 0.0, "max": 48.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "free"
    CATEGORY = "utils/memory"

    def free(self, latent, device, free_gb):
        target_device = torch.device(device)
        bytes_to_free = int(free_gb * 1024**3)
        print(f"[FreeVRAM] Requesting {free_gb:.1f} GB free on {device}")
        model_management.free_memory(bytes_to_free, target_device)
        model_management.soft_empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(target_device)
        print(
            f"[FreeVRAM] {device}: {free_bytes / 1024**3:.2f} GB free / "
            f"{total_bytes / 1024**3:.2f} GB total"
        )
        return (latent,)


class VAEDecodeAutoTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "model/latent"
    DESCRIPTION = (
        "Runs adaptive tiled VAE decoding directly, avoiding a full-decode OOM first."
    )

    def decode(self, vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        vae.throw_exception_if_invalid()
        if vae.latent_dim == 2 and latent.ndim == 5:
            latent = latent[:, :, 0]

        model_management.soft_empty_cache()
        with model_management.cuda_device_context(vae.device):
            dims = latent.ndim - 2
            if dims == 1 or vae.extra_1d_channel is not None:
                pixel_samples = vae.decode_tiled_1d(latent)
            elif dims == 2:
                if vae.handles_tiling:
                    tile = 256 // vae.spacial_compression_decode()
                    overlap = tile // 4
                    pixel_samples = vae._decode_tiled_owned(
                        latent, tile_x=tile, tile_y=tile, overlap=overlap
                    )
                else:
                    pixel_samples = vae.decode_tiled_(latent)
            elif dims == 3:
                tile = 256 // vae.spacial_compression_decode()
                overlap = tile // 4
                if vae.handles_tiling:
                    memory_used = vae.memory_used_decode(
                        vae._tile_bounded_shape(latent.shape, tile, tile, None),
                        vae.vae_dtype,
                    )
                    model_management.load_models_gpu(
                        [vae.patcher],
                        memory_required=memory_used,
                        force_full_load=vae.disable_offload,
                    )
                    pixel_samples = vae._decode_tiled_owned(
                        latent, tile_x=tile, tile_y=tile, overlap=overlap
                    )
                else:
                    memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
                    budget = min(
                        memory_used,
                        int(model_management.get_total_memory(vae.device) * 0.8),
                    )
                    model_management.load_models_gpu(
                        [vae.patcher],
                        memory_required=budget,
                        force_full_load=vae.disable_offload,
                    )

                    tile_t = latent.shape[2]

                    def estimate(tile_time, tile_xy):
                        return vae.memory_used_decode(
                            vae._tile_bounded_shape(
                                latent.shape, tile_xy, tile_xy, tile_time
                            ),
                            vae.vae_dtype,
                        )

                    while tile_t > 2 and estimate(tile_t, tile) > budget:
                        tile_t = -(-tile_t // 2)
                    while (
                        tile * 2 <= max(latent.shape[3], latent.shape[4])
                        and estimate(tile_t, tile * 2) <= budget
                    ):
                        tile *= 2

                    overlap = tile // 4
                    print(
                        "[VAEDecodeAutoTiled] "
                        f"budget={budget / 1024**3:.2f} GiB, "
                        f"tile_t={tile_t}, tile_x={tile}, tile_y={tile}, "
                        f"overlap={overlap}"
                    )
                    pixel_samples = vae.decode_tiled_3d(
                        latent,
                        tile_t=tile_t,
                        tile_x=tile,
                        tile_y=tile,
                        overlap=(1, overlap, overlap),
                    )
            else:
                raise RuntimeError(f"Unsupported VAE latent dimensions: {dims}")

        pixel_samples = pixel_samples.to(vae.output_device).movedim(1, -1)
        if len(pixel_samples.shape) == 5:
            pixel_samples = pixel_samples.reshape(
                -1,
                pixel_samples.shape[-3],
                pixel_samples.shape[-2],
                pixel_samples.shape[-1],
            )
        return (pixel_samples,)


NODE_CLASS_MAPPINGS = {
    "FreeVRAMForPassthrough": FreeVRAMForPassthrough,
    "VAEDecodeAutoTiled": VAEDecodeAutoTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeVRAMForPassthrough": "Free VRAM",
    "VAEDecodeAutoTiled": "VAE Decode (Auto Tiled)",
}
