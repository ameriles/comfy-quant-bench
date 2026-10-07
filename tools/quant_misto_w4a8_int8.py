"""Misto W4A8 + INT8: parte das camadas de um checkpoint W4A8 ja pronto troca para INT8 ConvRot.

Motivo (25/09): o 10Eros v1.5 W4A8 chia no audio e o BF16 nao (julgamento do dono, render no Colab). O palpite e
o 4 bits nas camadas de audio. Em vez de requantizar tudo, este conversor:

    camadas que casam --regex   INT8 ConvRot (int8_tensorwise, o mesmo de tools/quant_int8.py), quantizadas
                                A PARTIR DA FONTE BF16, uma por vez, dentro do laco de escrita (sem acumular)
    todo o resto                copiado BYTE A BYTE do W4A8 existente (pesos 4 bits, escalas, bias, VAE...)

`_quantization_metadata` sai do W4A8 com as entradas dessas camadas trocadas para
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}; o ComfyUI ja le misto por camada.

Recusas: saida/sidecar/partial existentes, fonte que nao e a do W4A8 (tamanho no sidecar do W4A8), tensor fora
das camadas com forma/dtype diferente da fonte, camada INT8 com K nao divisivel pelo grupo do ConvRot.
Cada camada INT8 e conferida pelo dequantizador real do ComfyUI (TensorWiseINT8Layout) contra a fonte; o erro
vai para o sidecar. Isso prova o formato, nao a qualidade -- qualidade e render.

O preset ``ltx25-audio-safe`` e deliberadamente estrito: promove para INT8 as 912 Linears dos caminhos
de audio, audio<->video e ``audio_embeddings_connector`` do LTX 2.5 oficial, deixando 528 Linears de
video em W4A8. Ele recusa outra arquitetura, outro formato-base ou qualquer contagem diferente.

O preset ``ltx25-q4km-audio-balanced`` reproduz, com os dois formatos nativos disponiveis, o mapa
observado no LTX-2.5-Distilled-Q4_K_M.gguf do laboratorio: as 172 Linears de audio que o GGUF guarda
em Q4_K permanecem W4A8; as 684 Q5_K e 56 Q6_K viram INT8 ConvRot. Resultado estrito: 740 INT8 +
700 W4A8. Q5_K nao tem equivalente nativo neste pipeline, por isso ele e promovido para INT8.

O preset ``ltx25-q4km-audio-video-q6`` acrescenta a essa receita somente as 46 Linears visuais que
o mesmo GGUF guarda em Q6_K: 16 do conector de video e 30 projecoes de valor/saida nos blocos. Ele
mantem as 380 visuais Q5_K em W4A8 e produz 786 INT8 + 654 W4A8.

O preset ``ltx25-q4km-optimized`` usa o Q4_K_M como mapa de sensibilidade, sem tentar copiar seus
formatos literalmente: Q4_K e Q5_K permanecem W4A8, salvo as sete familias que o proprio GGUF
promove a Q6_K nos blocos Q4. Essas mesmas familias viram INT8 em todos os blocos, incluindo seus
equivalentes Q5_K. Os conectores preservam em INT8 ``attn1.to_v`` e ``ff.net.2``. Resultado estrito:
102 equivalentes Q6_K + 266 equivalentes Q5_K sensiveis = 368 INT8; as outras 1072 ficam W4A8.

O preset ``ltx25-q4km-audio-balanced-visual-sensitive`` conserva as 740 Linears INT8 do
``audio-balanced`` e soma as 160 visuais sensiveis do ``optimized``: ``attn1.to_v``,
``attn2.to_v`` e ``ff.net.2`` nos blocos, mais ``attn1.to_v`` e ``ff.net.2`` no conector
de video. Resultado estrito: 900 INT8 + 540 W4A8.

    python_embeded\\python.exe -s tools/quant_misto_w4a8_int8.py --fonte P:/ComfyBench/checkpoints/10Eros_v1.5_bf16.safetensors \\
        --w4a8 P:/ComfyBench/checkpoints/10Eros_v1.5_bf16_w4a8.safetensors --dry-run
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import struct
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ComfyUI"))

import _conversion as C  # noqa: E402
from quant_int8 import quantize  # noqa: E402
from _conversion import read_header, read_tensor  # noqa: E402

REGEX_AUDIO = r"(?:^|\.)(?:audio_attn\d+|audio_ff|audio_to_video_attn|video_to_audio_attn)\."
LTX25_AUDIO_SAFE = "ltx25-audio-safe"
LTX25_Q4KM_AUDIO_BALANCED = "ltx25-q4km-audio-balanced"
LTX25_Q4KM_AUDIO_VIDEO_Q6 = "ltx25-q4km-audio-video-q6"
LTX25_Q4KM_OPTIMIZED = "ltx25-q4km-optimized"
LTX25_Q4KM_AUDIO_BALANCED_VISUAL_SENSITIVE = "ltx25-q4km-audio-balanced-visual-sensitive"
LTX25_Q4KM_REFERENCE = {
    "filename": "LTX-2.5-Distilled-Q4_K_M.gguf",
    "sha256": "0f51eb0d82b19bddbfb3b0371a65217844ea03750f27dd733528f22152e0e0d0",
    "same_audio_layers": {"Q4_K": 172, "Q5_K": 684, "Q6_K": 56},
    "mapping": {"Q4_K": "asym_w4a8_int8", "Q5_K": "int8_tensorwise", "Q6_K": "int8_tensorwise"},
}
LTX25_Q4KM_VIDEO_Q6_REFERENCE = {
    **LTX25_Q4KM_REFERENCE,
    "same_video_layers": {"Q4_K": 102, "Q5_K": 380, "Q6_K": 46},
    "video_mapping": {"Q4_K": "asym_w4a8_int8", "Q5_K": "asym_w4a8_int8",
                      "Q6_K": "int8_tensorwise"},
}
LTX25_AUDIO_FAMILIES = {
    "audio_embeddings_connector": 48,
    "audio_attn1": 192,
    "audio_attn2": 192,
    "audio_ff": 96,
    "audio_to_video_attn": 192,
    "video_to_audio_attn": 192,
}
REGEX_LTX25_AUDIO_SAFE = (
    r"(?:^|\.)(?:audio_embeddings_connector|audio_attn\d+|audio_ff|"
    r"audio_to_video_attn|video_to_audio_attn)\."
)
PRESET_REGEX = {
    LTX25_AUDIO_SAFE: REGEX_LTX25_AUDIO_SAFE,
    LTX25_Q4KM_AUDIO_BALANCED: REGEX_LTX25_AUDIO_SAFE,
    LTX25_Q4KM_AUDIO_VIDEO_Q6: REGEX_LTX25_AUDIO_SAFE,
    LTX25_Q4KM_OPTIMIZED: r"(?:^|\.)(?:transformer_blocks|[av]udio_embeddings_connector)\.",
    LTX25_Q4KM_AUDIO_BALANCED_VISUAL_SENSITIVE:
        r"(?:^|\.)(?:transformer_blocks|[av]udio_embeddings_connector)\.",
}
LTX25_Q4KM_Q4_BLOCKS = frozenset((*range(9), 17))
LTX25_Q4KM_W4A8_FAMILIES = {
    "audio_embeddings_connector": 32,
    "audio_attn1": 30,
    "audio_attn2": 30,
    "audio_ff": 20,
    "audio_to_video_attn": 30,
    "video_to_audio_attn": 30,
}
LTX25_Q4KM_INT8_FAMILIES = {
    family: LTX25_AUDIO_FAMILIES[family] - count
    for family, count in LTX25_Q4KM_W4A8_FAMILIES.items()
}


def is_ltx25_q4km_q4(name: str) -> bool:
    """Mapa nominal das 172 Linears que o GGUF de referencia armazena em Q4_K."""
    connector = re.search(
        r"(?:^|\.)audio_embeddings_connector\.transformer_1d_blocks\.(\d+)\."
        r"(?:attn1\.(?:to_k|to_out\.0|to_q)|ff\.net\.0\.proj)$",
        name,
    )
    if connector:
        return 0 <= int(connector.group(1)) < 8

    block = re.search(
        r"(?:^|\.)transformer_blocks\.(\d+)\."
        r"(?:(?:audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\."
        r"(?:to_k|to_out\.0|to_q)|audio_ff\.net\.(?:0\.proj|2))$",
        name,
    )
    return bool(block and int(block.group(1)) in LTX25_Q4KM_Q4_BLOCKS)


def is_ltx25_q4km_video_q6(name: str) -> bool:
    """Mapa nominal das 46 Linears visuais que o GGUF de referencia armazena em Q6_K."""
    connector = re.search(
        r"(?:^|\.)video_embeddings_connector\.transformer_1d_blocks\.(\d+)\."
        r"(?:attn1\.to_v|ff\.net\.2)$",
        name,
    )
    if connector:
        return 0 <= int(connector.group(1)) < 8

    block = re.search(
        r"(?:^|\.)transformer_blocks\.(\d+)\."
        r"(?:attn1\.to_v|attn2\.to_v|ff\.net\.2)$",
        name,
    )
    return bool(block and int(block.group(1)) in LTX25_Q4KM_Q4_BLOCKS)


def is_ltx25_q4km_sensitive(name: str) -> bool:
    """Mapa Q6_K do GGUF estendido as mesmas familias dentro dos blocos Q5_K."""
    connector = re.search(
        r"(?:^|\.)(?:audio|video)_embeddings_connector\.transformer_1d_blocks\.(\d+)\."
        r"(?:attn1\.to_v|ff\.net\.2)$",
        name,
    )
    if connector:
        return 0 <= int(connector.group(1)) < 8

    return re.search(
        r"(?:^|\.)transformer_blocks\.(\d+)\."
        r"(?:(?:attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|"
        r"video_to_audio_attn)\.to_v|ff\.net\.2)$",
        name,
    ) is not None


def validate_ltx25_audio_safe(camadas: dict, alvo: list[str], sidecar: dict) -> None:
    """Recusa silencios perigosos: este preset so vale para o LTX 2.5 oficial medido."""
    expected_sidecar = {
        "architecture": "ltx_2_5",
        "quantization": "asym_w4a8_int8",
        "quantized_tensors": 1440,
        "preserved_tensors": 2909,
    }
    for key, expected in expected_sidecar.items():
        if sidecar.get(key) != expected:
            raise SystemExit(
                f"RECUSADO: preset {LTX25_AUDIO_SAFE} exige sidecar {key}={expected!r}, "
                f"recebeu {sidecar.get(key)!r}"
            )

    if len(camadas) != 1440:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_AUDIO_SAFE} exige 1440 camadas quantizadas, "
            f"recebeu {len(camadas)}"
        )
    formatos = {config.get("format") for config in camadas.values()}
    if formatos != {"asym_w4a8_int8"}:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_AUDIO_SAFE} exige base W4A8 uniforme, "
            f"recebeu {sorted(str(value) for value in formatos)}"
        )

    family_counts = {family: sum(family in name for name in alvo)
                     for family in LTX25_AUDIO_FAMILIES}
    if family_counts != LTX25_AUDIO_FAMILIES:
        raise SystemExit(
            f"RECUSADO: familias do preset {LTX25_AUDIO_SAFE} diferem: "
            f"esperado {LTX25_AUDIO_FAMILIES}, recebeu {family_counts}"
        )
    if len(alvo) != 912 or len(camadas) - len(alvo) != 528:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_AUDIO_SAFE} exige 912 INT8 + 528 W4A8; "
            f"recebeu {len(alvo)} + {len(camadas) - len(alvo)}"
        )


def select_ltx25_q4km_audio_balanced(camadas: dict, sidecar: dict) -> list[str]:
    """Seleciona Q5_K/Q6_K como INT8 e deixa os equivalentes Q4_K em W4A8."""
    regex = re.compile(REGEX_LTX25_AUDIO_SAFE)
    audio = sorted(name for name in camadas if regex.search(name))
    validate_ltx25_audio_safe(camadas, audio, sidecar)

    w4a8 = sorted(name for name in audio if is_ltx25_q4km_q4(name))
    int8 = sorted(set(audio) - set(w4a8))
    w4a8_counts = {family: sum(family in name for name in w4a8)
                   for family in LTX25_AUDIO_FAMILIES}
    int8_counts = {family: sum(family in name for name in int8)
                   for family in LTX25_AUDIO_FAMILIES}
    if w4a8_counts != LTX25_Q4KM_W4A8_FAMILIES:
        raise SystemExit(
            f"RECUSADO: mapa Q4_K do preset {LTX25_Q4KM_AUDIO_BALANCED} difere: "
            f"esperado {LTX25_Q4KM_W4A8_FAMILIES}, recebeu {w4a8_counts}"
        )
    if int8_counts != LTX25_Q4KM_INT8_FAMILIES or len(int8) != 740:
        raise SystemExit(
            f"RECUSADO: mapa Q5_K/Q6_K do preset {LTX25_Q4KM_AUDIO_BALANCED} difere: "
            f"esperado {LTX25_Q4KM_INT8_FAMILIES}, recebeu {int8_counts} ({len(int8)} camadas)"
        )
    if len(w4a8) != 172 or len(camadas) - len(int8) != 700:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_Q4KM_AUDIO_BALANCED} exige 740 INT8 + 700 W4A8; "
            f"recebeu {len(int8)} + {len(camadas) - len(int8)}"
        )
    return int8


def select_ltx25_q4km_audio_video_q6(camadas: dict, sidecar: dict) -> list[str]:
    """Soma as 46 visuais Q6_K ao mapa conservador de audio Q5_K/Q6_K."""
    audio_int8 = select_ltx25_q4km_audio_balanced(camadas, sidecar)
    video_q6 = sorted(name for name in camadas if is_ltx25_q4km_video_q6(name))
    expected = {
        "video_embeddings_connector": 16,
        "attn1": 10,
        "attn2": 10,
        "ff": 10,
    }
    received = {
        "video_embeddings_connector": sum("video_embeddings_connector" in name for name in video_q6),
        "attn1": sum(re.search(r"\.transformer_blocks\.\d+\.attn1\.to_v$", name) is not None
                     for name in video_q6),
        "attn2": sum(re.search(r"\.transformer_blocks\.\d+\.attn2\.to_v$", name) is not None
                     for name in video_q6),
        "ff": sum(re.search(r"\.transformer_blocks\.\d+\.ff\.net\.2$", name) is not None
                  for name in video_q6),
    }
    if len(video_q6) != 46 or received != expected:
        raise SystemExit(
            f"RECUSADO: mapa visual Q6_K do preset {LTX25_Q4KM_AUDIO_VIDEO_Q6} difere: "
            f"esperado {expected} (46 camadas), recebeu {received} ({len(video_q6)} camadas)"
        )
    int8 = sorted(set(audio_int8) | set(video_q6))
    if len(int8) != 786 or len(camadas) - len(int8) != 654:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_Q4KM_AUDIO_VIDEO_Q6} exige 786 INT8 + 654 W4A8; "
            f"recebeu {len(int8)} + {len(camadas) - len(int8)}"
        )
    return int8


def select_ltx25_q4km_optimized(camadas: dict, sidecar: dict) -> list[str]:
    """Protege em INT8 as familias Q6_K e seus homologos nos blocos Q5_K."""
    all_audio = sorted(name for name in camadas if re.search(REGEX_LTX25_AUDIO_SAFE, name))
    validate_ltx25_audio_safe(camadas, all_audio, sidecar)
    int8 = sorted(name for name in camadas if is_ltx25_q4km_sensitive(name))
    received = {
        "audio_connector": sum("audio_embeddings_connector" in name for name in int8),
        "video_connector": sum("video_embeddings_connector" in name for name in int8),
        "video_attn1": sum(re.search(r"\.transformer_blocks\.\d+\.attn1\.to_v$", name) is not None
                           for name in int8),
        "video_attn2": sum(re.search(r"\.transformer_blocks\.\d+\.attn2\.to_v$", name) is not None
                           for name in int8),
        "audio_attn1": sum(re.search(r"\.transformer_blocks\.\d+\.audio_attn1\.to_v$", name) is not None
                           for name in int8),
        "audio_attn2": sum(re.search(r"\.transformer_blocks\.\d+\.audio_attn2\.to_v$", name) is not None
                           for name in int8),
        "audio_to_video": sum(re.search(r"\.transformer_blocks\.\d+\.audio_to_video_attn\.to_v$", name) is not None
                              for name in int8),
        "video_to_audio": sum(re.search(r"\.transformer_blocks\.\d+\.video_to_audio_attn\.to_v$", name) is not None
                              for name in int8),
        "video_ff": sum(re.search(r"\.transformer_blocks\.\d+\.ff\.net\.2$", name) is not None
                        for name in int8),
    }
    expected = {
        "audio_connector": 16, "video_connector": 16,
        "video_attn1": 48, "video_attn2": 48,
        "audio_attn1": 48, "audio_attn2": 48,
        "audio_to_video": 48, "video_to_audio": 48, "video_ff": 48,
    }
    if received != expected or len(int8) != 368 or len(camadas) - len(int8) != 1072:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_Q4KM_OPTIMIZED} exige 368 INT8 + 1072 W4A8 "
            f"e familias {expected}; recebeu {len(int8)} + {len(camadas) - len(int8)}, {received}"
        )
    return int8


def select_ltx25_q4km_audio_balanced_visual_sensitive(
    camadas: dict, sidecar: dict
) -> list[str]:
    """Une o audio validado do balanced as 160 familias visuais sensiveis."""
    audio_int8 = set(select_ltx25_q4km_audio_balanced(camadas, sidecar))
    visual_int8 = sorted(
        name for name in camadas
        if not re.search(REGEX_LTX25_AUDIO_SAFE, name) and is_ltx25_q4km_sensitive(name)
    )
    received = {
        "video_connector": sum("video_embeddings_connector" in name for name in visual_int8),
        "video_attn1": sum(
            re.search(r"\.transformer_blocks\.\d+\.attn1\.to_v$", name) is not None
            for name in visual_int8
        ),
        "video_attn2": sum(
            re.search(r"\.transformer_blocks\.\d+\.attn2\.to_v$", name) is not None
            for name in visual_int8
        ),
        "video_ff": sum(
            re.search(r"\.transformer_blocks\.\d+\.ff\.net\.2$", name) is not None
            for name in visual_int8
        ),
    }
    expected = {
        "video_connector": 16,
        "video_attn1": 48,
        "video_attn2": 48,
        "video_ff": 48,
    }
    int8 = sorted(audio_int8 | set(visual_int8))
    if received != expected or len(visual_int8) != 160:
        raise SystemExit(
            f"RECUSADO: mapa visual do preset "
            f"{LTX25_Q4KM_AUDIO_BALANCED_VISUAL_SENSITIVE} difere: esperado "
            f"{expected} (160 camadas), recebeu {received} ({len(visual_int8)} camadas)"
        )
    if len(int8) != 900 or len(camadas) - len(int8) != 540:
        raise SystemExit(
            f"RECUSADO: preset {LTX25_Q4KM_AUDIO_BALANCED_VISUAL_SENSITIVE} exige "
            f"900 INT8 + 540 W4A8; recebeu {len(int8)} + {len(camadas) - len(int8)}"
        )
    return int8


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", required=True, type=Path, help="checkpoint BF16 de onde o W4A8 saiu")
    p.add_argument("--w4a8", required=True, type=Path, help="checkpoint W4A8 pronto (base da copia)")
    selecao = p.add_mutually_exclusive_group()
    selecao.add_argument("--preset", choices=sorted(PRESET_REGEX),
                         help="receita estrita com contagens e proveniencia conhecidas")
    selecao.add_argument("--regex", help="camadas (nome sem .weight) que viram INT8")
    p.add_argument("--convrot-groupsize", type=int, default=256)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    fonte = a.fonte.resolve()
    # Preserva o caminho fornecido para encontrar o sidecar ao lado de uma referencia simbolica,
    # mas abre os pesos pelo alvo real. Assim o laboratorio nao precisa duplicar o W4A8 de 12,5 GB.
    base_ref = a.w4a8.absolute()
    base = base_ref.resolve()
    output = (a.output or base_ref.with_name(base_ref.stem + "_audioint8.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    conv = C.Conversion(base, output, sidecar)
    conv.refuse_unsafe(allow_quantized_source=True)  # a base E quantizada por construcao
    hb, mb = conv.header, conv.metadata
    hf, _ = read_header(fonte)

    side_base = base_ref.with_suffix(".quant.json")
    if side_base.is_file():
        side_base_data = json.loads(side_base.read_text(encoding="utf-8"))
        tam = side_base_data.get("source_size")
        if tam != fonte.stat().st_size:
            raise SystemExit(f"RECUSADO: o W4A8 veio de uma fonte de {tam} B, esta tem {fonte.stat().st_size} B")
        output_size = side_base_data.get("output_size")
        if output_size != base.stat().st_size:
            raise SystemExit(
                f"RECUSADO: sidecar declara W4A8 de {output_size} B, arquivo tem {base.stat().st_size} B"
            )
    else:
        raise SystemExit(f"RECUSADO: sem sidecar {side_base} para provar de que fonte o W4A8 saiu")

    qmeta = json.loads(mb["_quantization_metadata"])
    camadas = qmeta["layers"]
    regex_int8 = PRESET_REGEX[a.preset] if a.preset else (a.regex or REGEX_AUDIO)
    rx = re.compile(regex_int8)
    alvo = sorted(n for n in camadas if rx.search(n))
    if not alvo:
        raise SystemExit("RECUSADO: --regex nao casou nenhuma camada quantizada")
    if a.preset == LTX25_AUDIO_SAFE:
        validate_ltx25_audio_safe(camadas, alvo, side_base_data)
    elif a.preset == LTX25_Q4KM_AUDIO_BALANCED:
        alvo = select_ltx25_q4km_audio_balanced(camadas, side_base_data)
    elif a.preset == LTX25_Q4KM_AUDIO_VIDEO_Q6:
        alvo = select_ltx25_q4km_audio_video_q6(camadas, side_base_data)
    elif a.preset == LTX25_Q4KM_OPTIMIZED:
        alvo = select_ltx25_q4km_optimized(camadas, side_base_data)
    elif a.preset == LTX25_Q4KM_AUDIO_BALANCED_VISUAL_SENSITIVE:
        alvo = select_ltx25_q4km_audio_balanced_visual_sensitive(camadas, side_base_data)
    for n in alvo:
        info = hf.get(n + ".weight")
        if info is None or len(info["shape"]) != 2:
            raise SystemExit(f"RECUSADO: {n}.weight ausente ou nao 2D na fonte")
        if info["shape"][1] % a.convrot_groupsize:
            raise SystemExit(f"RECUSADO: {n} K={info['shape'][1]} nao divide por {a.convrot_groupsize}")
    # tensores de cada camada alvo no W4A8 (peso 4 bits + escalas) saem; o bias fica
    sai = {k for n in alvo for k in hb if k.startswith(n + ".weight")}
    # fora das camadas quantizadas, a base tem de ser a fonte (mesma forma e dtype)
    prefixos = tuple(n + "." for n in camadas)
    for k, v in hb.items():
        if not k.startswith(prefixos):
            f = hf.get(k)
            if f is None or f["shape"] != v["shape"] or f["dtype"] != v["dtype"]:
                raise SystemExit(f"RECUSADO: {k} difere da fonte ({v['dtype']} {v['shape']} vs "
                                 f"{f and (f['dtype'], f['shape'])})")

    bytes_int8 = sum(hf[n + ".weight"]["shape"][0] * hf[n + ".weight"]["shape"][1] for n in alvo)
    bytes_sai = sum(hb[k]["data_offsets"][1] - hb[k]["data_offsets"][0] for k in sai)
    tam_saida = base.stat().st_size - bytes_sai + bytes_int8 + sum(hf[n + ".weight"]["shape"][0] * 4 for n in alvo)
    print(f"base {base.name}: {len(camadas)} camadas quantizadas; {len(alvo)} viram INT8 ConvRot")
    if a.preset:
        print(f"preset {a.preset}: {len(alvo)} INT8 + {len(camadas) - len(alvo)} W4A8")
    for n in alvo[:6]:
        print(f"  {n}  {hf[n + '.weight']['shape']}")
    print(f"  ... saida estimada {tam_saida / 2**30:.2f} GiB (base {base.stat().st_size / 2**30:.2f})  -> {output}")
    if a.dry_run:
        return 0
    conv.guard(tam_saida)

    # forma da escala: descoberta quantizando a primeira camada (nao presumida)
    fh = open(fonte, "rb")
    fdata = 8 + struct.unpack("<Q", fh.read(8))[0]

    def le(n):
        i = hf[n + ".weight"]
        s, e = i["data_offsets"]
        return read_tensor(fh, fdata + s, e - s, i["dtype"], i["shape"])

    _, esc0 = quantize(le(alvo[0]), True, a.convrot_groupsize)
    forma_esc = lambda n: [hf[n + ".weight"]["shape"][0], *esc0.shape[1:]]  # noqa: E731

    from comfy.quant_ops import QUANT_ALGOS, get_layout_class
    layout = get_layout_class(QUANT_ALGOS["int8_tensorwise"]["comfy_tensor_layout"])
    erros: dict[str, float] = {}
    pendente: dict[str, torch.Tensor] = {}

    def produz_peso(n):
        def f():
            w = le(n)
            q, s = quantize(w, True, a.convrot_groupsize)
            q, s = q.cpu().contiguous(), s.cpu().contiguous().float()
            params = layout.Params(scale=s, orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape),
                                   convrot=True, convrot_groupsize=a.convrot_groupsize)
            d = layout.dequantize(q, params).float()
            erros[n] = float((d - w.float()).norm() / w.float().norm().clamp_min(1e-30))
            pendente[n] = s
            return q
        return f

    def produz_escala(n):
        return lambda: pendente.pop(n)

    entradas, feitos = [], set()
    for k, v in hb.items():
        dono = next((n for n in alvo if k.startswith(n + ".weight")), None)
        if dono is None:
            entradas.append(C.plan_copy(k, v))
        elif dono not in feitos:  # no lugar do primeiro tensor do peso 4 bits, escreve peso INT8 + escala
            feitos.add(dono)
            N, K = hf[dono + ".weight"]["shape"]
            fe = forma_esc(dono)
            entradas.append(C.plan_lazy(dono + ".weight", "I8", [N, K], N * K, produz_peso(dono)))
            entradas.append(C.plan_lazy(dono + ".weight_scale", "F32", fe, 4 * int(torch.tensor(fe).prod()),
                                        produz_escala(dono)))
    assert feitos == set(alvo)

    for n in alvo:
        camadas[n] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": a.convrot_groupsize}
    meta = dict(mb)
    meta["_quantization_metadata"] = json.dumps(qmeta, separators=(",", ":"))
    meta["quantization"] = f"{mb.get('quantization', '?')}+int8_tensorwise_convrot({len(alvo)} camadas)"

    t0 = time.perf_counter()
    ultimo = [0.0]

    def progresso(i, tot, chave):
        if time.perf_counter() - ultimo[0] > 30 or i == tot:
            ultimo[0] = time.perf_counter()
            print(f"[{i}/{tot}] {len(erros)}/{len(alvo)} INT8  {time.perf_counter() - t0:.0f} s  {chave}", flush=True)

    conv.commit(entradas, meta, progress=progresso)
    fh.close()
    vals = sorted(erros.values())
    preset_details = {}
    if a.preset == LTX25_Q4KM_AUDIO_BALANCED:
        preset_details = {"gguf_reference_map": LTX25_Q4KM_REFERENCE}
    elif a.preset == LTX25_Q4KM_AUDIO_VIDEO_Q6:
        preset_details = {"gguf_reference_map": LTX25_Q4KM_VIDEO_Q6_REFERENCE}
    conv.write_sidecar({
        "fonte": str(fonte), "fonte_size": fonte.stat().st_size, "base_w4a8": str(base),
        "base_size": base.stat().st_size, "output": str(output), "output_size": output.stat().st_size,
        "preset": a.preset, "regex_int8": regex_int8,
        "camadas_int8": len(alvo), "camadas_w4a8": len(camadas) - len(alvo),
        **preset_details,
        "int8": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": a.convrot_groupsize},
        "erro_rel_int8_vs_fonte": {"mediana": vals[len(vals) // 2], "max": vals[-1],
                                   "pior": max(erros, key=erros.get)},
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__, "segundos": round(time.perf_counter() - t0, 1),
    })
    print(f"OK {output} ({output.stat().st_size / 2**30:.2f} GiB); erro INT8 mediana {vals[len(vals) // 2]:.4f} "
          f"max {vals[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
