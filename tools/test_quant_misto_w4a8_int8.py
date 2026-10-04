import re
import unittest

from quant_misto_w4a8_int8 import (
    LTX25_AUDIO_FAMILIES,
    REGEX_LTX25_AUDIO_SAFE,
    validate_ltx25_audio_safe,
)


def ltx25_fixture():
    layers = {}
    for family, count in LTX25_AUDIO_FAMILIES.items():
        for index in range(count):
            layers[f"model.diffusion_model.{family}.{index}"] = {
                "format": "asym_w4a8_int8"
            }
    for index in range(528):
        layers[f"model.diffusion_model.video_path.{index}"] = {
            "format": "asym_w4a8_int8"
        }
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


if __name__ == "__main__":
    unittest.main()
