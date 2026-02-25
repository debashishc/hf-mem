import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

SafetensorsDtypes = Literal[
    "F64",
    "I64",
    "U64",
    "F32",
    "I32",
    "U32",
    "F16",
    "BF16",
    "I16",
    "U16",
    "F8_E5M2",  # NOTE: Only CUDA +11.8
    "F8_E4M3",  # NOTE: CUDA +11.8 and AMD ROCm
    "I8",
    "U8",
]


def get_safetensors_dtype_bytes(dtype: SafetensorsDtypes | str) -> int:
    match dtype:
        case "F64" | "I64" | "U64":
            return 8
        case "F32" | "I32" | "U32":
            return 4
        case "F16" | "BF16" | "I16" | "U16":
            return 2
        case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
            return 1
        case _:
            raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")


TorchDtypes = Literal["float32", "float16", "bfloat16", "float8_e4m3", "float8_e4m3fn", "float8_e5m2", "int8"]


def torch_dtype_to_safetensors_dtype(dtype: TorchDtypes | str) -> SafetensorsDtypes:
    if dtype.startswith("torch."):
        dtype = dtype.replace("torch.", "")
    match dtype:
        case "float32":
            return "F32"
        case "float16":
            return "F16"
        case "bfloat16":
            return "BF16"
        case "float8_e4m3" | "float8_e4m3fn":
            return "F8_E4M3"
        case "float8_e5m2":
            return "F8_E5M2"
        # NOTE: `I8` is usally not used for quantizing i.e., the KV cache will never be of type `I8`, hence this
        # case might never be hit
        case "int8":
            return "I8"
        case _:
            return "F16"


@dataclass
class GpuSpec:
    """Specification for a GPU type."""

    name: str
    vram_gib: float
    max_per_node: int | None

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gib * (1024**3))


GPU_REGISTRY: Dict[str, GpuSpec] = {
    "b200":     GpuSpec("B200",     192, 8),
    "h200":     GpuSpec("H200",     141, 8),
    "h100":     GpuSpec("H100",      80, 8),
    "gh200":    GpuSpec("GH200",     96, 1),
    "a100-80":  GpuSpec("A100-80G",  80, 8),
    "a100-40":  GpuSpec("A100-40G",  40, 8),
    "l40s":     GpuSpec("L40S",      48, 8),
    "v100-32":  GpuSpec("V100-32G",  32, 8),
    "v100-16":  GpuSpec("V100-16G",  16, 8),
    "a10":      GpuSpec("A10",       24, None),
    "rtx4090":  GpuSpec("RTX 4090",  24, None),
    "rtx3090":  GpuSpec("RTX 3090",  24, None),
}

COMMON_TP_DEGREES = [1, 2, 4, 8]
SuggestionReasonCode = Literal["head_divisible", "common_tp_degree", "raw_capacity"]
SUGGESTION_REASON_TEXT: Dict[SuggestionReasonCode, str] = {
    "head_divisible": "Rounded up to a head-divisible parallel degree",
    "common_tp_degree": "Rounded up to common parallel degree (1/2/4/8)",
    "raw_capacity": "Used raw capacity count (no higher compatible degree found)",
}


def get_gpu_spec(name: str) -> GpuSpec:
    key = name.lower().strip()
    if key not in GPU_REGISTRY:
        valid = ", ".join(sorted(GPU_REGISTRY.keys()))
        raise RuntimeError(
            f"Unknown GPU '{name}'. Valid options: {valid}\n"
            f"Use `hf-mem --list-gpus` to see all supported GPUs."
        )
    return GPU_REGISTRY[key]


def find_valid_tp_degrees(num_attention_heads: int | None) -> List[int]:
    if num_attention_heads is None:
        return COMMON_TP_DEGREES
    return [d for d in range(1, num_attention_heads + 1) if num_attention_heads % d == 0]


def get_suggestion_reason_text(reason: SuggestionReasonCode) -> str:
    return SUGGESTION_REASON_TEXT[reason]


def compute_gpu_count(
    total_bytes: int,
    gpu: GpuSpec,
    overhead: float = 0.0,
    num_attention_heads: int | None = None,
) -> Tuple[int, int, SuggestionReasonCode]:
    """Returns (raw_count, suggested_count, reason).

    raw_count:       math.ceil(total_bytes / effective_vram)
    suggested_count: smallest valid TP degree >= raw_count, or raw_count if none qualifies
    reason:          "head_divisible", "common_tp_degree", or "raw_capacity"
    """
    effective_vram = gpu.vram_bytes * (1.0 - overhead)
    raw = math.ceil(total_bytes / effective_vram)
    valid_degrees = find_valid_tp_degrees(num_attention_heads)
    suggested = next((d for d in sorted(valid_degrees) if d >= raw), None)
    if suggested is not None:
        reason = "head_divisible" if num_attention_heads is not None else "common_tp_degree"
    else:
        suggested = raw
        reason = "raw_capacity"
    return raw, suggested, reason


def format_gpu_table() -> str:
    lines = [
        f"{'Name':<12} {'VRAM (GiB)':>10}  {'Max/Node':>8}",
        f"{'─' * 12} {'─' * 10}  {'─' * 8}",
    ]
    for key, spec in GPU_REGISTRY.items():
        node_str = str(spec.max_per_node) if spec.max_per_node else "—"
        lines.append(f"{key:<12} {spec.vram_gib:>10.0f}  {node_str:>8}")
    return "\n".join(lines)
