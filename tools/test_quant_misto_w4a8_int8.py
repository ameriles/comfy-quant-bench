import re
import unittest

from quant_misto_w4a8_int8 import (
    LTX25_AUDIO_FAMILIES,
    LTX25_Q4KM_INT8_FAMILIES,
    LTX25_Q4KM_W4A8_FAMILIES,
    REGEX_LTX25_AUDIO_SAFE,
    is_ltx25_q4km_q4,
    is_ltx25_q4km_video_q6,
    is_ltx25_q4km_sensitive,
    select_ltx25_q4km_audio_balanced,
    select_ltx25_q4km_audio_video_q6,
    select_ltx25_q4km_optimized,
    select_ltx25_q4km_audio_balanced_visual_sensitive,
    validate_ltx25_audio_safe,
)


def ltx25_fixture():
    layers = {}
    prefix = "model.diffusion_model"
    for block in range(8):
        base = f"{prefix}.audio_embeddings_connector.transformer_1d_blocks.{block}"
        for suffix in (
            "attn1.to_k", "attn1.to_out.0", "attn1.to_q", "attn1.to_v",
            "ff.net.0.proj", "ff.net.2",
        ):
            layers[f"{base}.{suffix}"] = {"format": "asym_w4a8_int8"}
    attention_families = (
        "audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"
    )
    for block in range(48):
        base = f"{prefix}.transformer_blocks.{block}"
        for family in attention_families:
            for suffix in ("to_k", "to_out.0", "to_q", "to_v"):
                layers[f"{base}.{family}.{suffix}"] = {"format": "asym_w4a8_int8"}
        for suffix in ("net.0.proj", "net.2"):
            layers[f"{base}.audio_ff.{suffix}"] = {"format": "asym_w4a8_int8"}
    for block in range(8):
        base = f"{prefix}.video_embeddings_connector.transformer_1d_blocks.{block}"
        for suffix in (
            "attn1.to_k", "attn1.to_out.0", "attn1.to_q", "attn1.to_v",
            "ff.net.0.proj", "ff.net.2",
        ):
            layers[f"{base}.{suffix}"] = {"format": "asym_w4a8_int8"}
    for block in range(48):
        base = f"{prefix}.transformer_blocks.{block}"
        for family in ("attn1", "attn2"):
            for suffix in ("to_k", "to_out.0", "to_q", "to_v"):
                layers[f"{base}.{family}.{suffix}"] = {"format": "asym_w4a8_int8"}
        for suffix in ("net.0.proj", "net.2"):
            layers[f"{base}.ff.{suffix}"] = {"format": "asym_w4a8_int8"}
    regex = re.compile(REGEX_LTX25_AUDIO_SAFE)
    targets = sorted(name for name in layers if regex.search(name))
    sidecar = {
        "architecture": "ltx_2_5",
        "quantization": "asym_w4a8_int8",
        "quantized_tensors": 1440,
        "preserved_tensors": 2909,
    }
    return layers, targets, sidecar


class LTX25AudioSafeGuardTests(unittest.TestCase):
    def test_accepts_exact_recipe(self):
        layers, targets, sidecar = ltx25_fixture()

        validate_ltx25_audio_safe(layers, targets, sidecar)

    def test_rejects_wrong_architecture(self):
        layers, targets, sidecar = ltx25_fixture()
        sidecar["architecture"] = "other"

        with self.assertRaisesRegex(SystemExit, "architecture"):
            validate_ltx25_audio_safe(layers, targets, sidecar)

    def test_rejects_missing_audio_layer(self):
        layers, targets, sidecar = ltx25_fixture()
        removed = next(name for name in layers if "audio_embeddings_connector" in name)
        del layers[removed]
        layers["model.diffusion_model.video_path.extra"] = {"format": "asym_w4a8_int8"}
        targets.remove(removed)

        with self.assertRaisesRegex(SystemExit, "familias"):
            validate_ltx25_audio_safe(layers, targets, sidecar)

    def test_rejects_mixed_base(self):
        layers, targets, sidecar = ltx25_fixture()
        layers[targets[0]]["format"] = "int8_tensorwise"

        with self.assertRaisesRegex(SystemExit, "base W4A8 uniforme"):
            validate_ltx25_audio_safe(layers, targets, sidecar)

    def test_q4km_balanced_exact_recipe(self):
        layers, _, sidecar = ltx25_fixture()

        int8 = select_ltx25_q4km_audio_balanced(layers, sidecar)
        audio = [name for name in layers if any(family in name for family in LTX25_AUDIO_FAMILIES)]
        w4a8 = [name for name in audio if is_ltx25_q4km_q4(name)]

        self.assertEqual(len(int8), 740)
        self.assertEqual(len(w4a8), 172)
        self.assertEqual(len(layers) - len(int8), 700)
        self.assertEqual(
            {family: sum(family in name for name in int8) for family in LTX25_AUDIO_FAMILIES},
            LTX25_Q4KM_INT8_FAMILIES,
        )
        self.assertEqual(
            {family: sum(family in name for name in w4a8) for family in LTX25_AUDIO_FAMILIES},
            LTX25_Q4KM_W4A8_FAMILIES,
        )

    def test_q4km_balanced_rejects_changed_map(self):
        layers, _, sidecar = ltx25_fixture()
        removed = "model.diffusion_model.transformer_blocks.17.audio_ff.net.2"
        del layers[removed]
        layers["model.diffusion_model.video_path.replacement"] = {"format": "asym_w4a8_int8"}

        with self.assertRaisesRegex(SystemExit, "familias"):
            select_ltx25_q4km_audio_balanced(layers, sidecar)

    def test_q4km_audio_video_q6_exact_recipe(self):
        layers, _, sidecar = ltx25_fixture()

        int8 = select_ltx25_q4km_audio_video_q6(layers, sidecar)
        video_q6 = [name for name in layers if is_ltx25_q4km_video_q6(name)]

        self.assertEqual(len(video_q6), 46)
        self.assertEqual(len(int8), 786)
        self.assertEqual(len(layers) - len(int8), 654)

    def test_q4km_audio_video_q6_rejects_changed_map(self):
        layers, _, sidecar = ltx25_fixture()
        removed = "model.diffusion_model.transformer_blocks.17.attn1.to_v"
        del layers[removed]
        layers["model.diffusion_model.video_path.replacement"] = {"format": "asym_w4a8_int8"}

        with self.assertRaisesRegex(SystemExit, "visual Q6_K"):
            select_ltx25_q4km_audio_video_q6(layers, sidecar)

    def test_q4km_optimized_exact_recipe(self):
        layers, _, sidecar = ltx25_fixture()

        int8 = select_ltx25_q4km_optimized(layers, sidecar)

        self.assertEqual(len(int8), 368)
        self.assertEqual(len(layers) - len(int8), 1072)
        self.assertTrue(all(is_ltx25_q4km_sensitive(name) for name in int8))

    def test_q4km_optimized_rejects_changed_map(self):
        layers, _, sidecar = ltx25_fixture()
        removed = "model.diffusion_model.transformer_blocks.47.video_to_audio_attn.to_v"
        del layers[removed]
        layers["model.diffusion_model.video_path.replacement"] = {"format": "asym_w4a8_int8"}

        with self.assertRaisesRegex(SystemExit, "familias"):
            select_ltx25_q4km_optimized(layers, sidecar)

    def test_q4km_audio_balanced_visual_sensitive_exact_recipe(self):
        layers, _, sidecar = ltx25_fixture()

        int8 = select_ltx25_q4km_audio_balanced_visual_sensitive(layers, sidecar)

        self.assertEqual(len(int8), 900)
        self.assertEqual(len(layers) - len(int8), 540)
        self.assertEqual(
            sum(
                is_ltx25_q4km_sensitive(name)
                and not re.search(REGEX_LTX25_AUDIO_SAFE, name)
                for name in int8
            ),
            160,
        )

    def test_q4km_audio_balanced_visual_sensitive_rejects_changed_map(self):
        layers, _, sidecar = ltx25_fixture()
        removed = "model.diffusion_model.transformer_blocks.47.attn2.to_v"
        del layers[removed]
        layers["model.diffusion_model.video_path.replacement"] = {"format": "asym_w4a8_int8"}

        with self.assertRaisesRegex(SystemExit, "visual"):
            select_ltx25_q4km_audio_balanced_visual_sensitive(layers, sidecar)


if __name__ == "__main__":
    unittest.main()
