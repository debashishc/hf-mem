<img src="https://github.com/user-attachments/assets/509a8244-8a91-4051-b337-41b7b2fe0e2f" />

---

> [!WARNING]
> `hf-mem` is still experimental and therefore subject to major changes across releases, so please keep in mind that breaking changes may occur until v1.0.0.

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face models, written in Python. `hf-mem` is lightweight, only depends on `httpx`, as it pulls the [Safetensors](https://github.com/huggingface/safetensors) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests). It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers) and [Sentence Transformers](https://github.com/huggingface/sentence-transformers) models, as well as any model that contains [Safetensors](https://github.com/huggingface/safetensors) compatible weights.

Read more information about `hf-mem` in [this short-form post](https://alvarobartt.com/hf-mem).

## Usage

### Transformers

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2
```

<img src="https://github.com/user-attachments/assets/530f8b14-a415-4fd6-9054-bcd81cafae09" width="600" />

### Diffusers

```bash
uvx hf-mem --model-id Qwen/Qwen-Image
```

<img src="https://github.com/user-attachments/assets/cd4234ec-bdcc-4db4-8b01-0ac9b5cd390c" width="600" />

### Sentence Transformers

```bash
uvx hf-mem --model-id google/embeddinggemma-300m
```

<img src="https://github.com/user-attachments/assets/a52c464b-a6c1-446d-9921-68aaefb9df88" width="600" />

## Experimental

By enabling the `--experimental` flag, you can enable the KV Cache memory estimation for LLMs (`...ForCausalLM`) and VLMs (`...ForConditionalGeneration`), even including a custom `--max-model-len` (defaults to the `config.json` default), `--batch-size` (defaults to 1), and the `--kv-cache-dtype` (defaults to `auto` which means it uses the default data type set in `config.json` under `torch_dtype` or `dtype`, or rather from `quantization_config` when applicable).

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2 --experimental
```

<img src="https://github.com/user-attachments/assets/64eaff88-d395-4d8d-849b-78fb86411dc3" width="600" />

## GPU Count Estimation

You can estimate how many GPUs are needed to host the model weights (and optionally the KV cache when `--experimental` is also enabled).

List supported GPU presets:

```bash
uvx hf-mem --list-gpus
```

Estimate GPU count for a model (weights only):

```bash
uvx hf-mem --model-id Qwen/Qwen3.5-397B-A17B-FP8 --gpu h100
```

<img src="https://github.com/user-attachments/assets/f28add99-c8e9-4865-89cb-352ea63cb0ff" width="600" />

Reserve headroom for runtime overhead with `--overhead` (e.g. `0.2` = 20% of VRAM reserved):

```bash
uvx hf-mem --model-id Qwen/Qwen3.5-397B-A17B-FP8 --gpu h100 --overhead 0.2
```

<img src="https://github.com/user-attachments/assets/95fdf2cf-95c8-4ebf-98be-64788ed2d51e" width="600" />

Override the preset VRAM for cluster-specific variants:

```bash
uvx hf-mem --model-id Qwen/Qwen3.5-397B-A17B-FP8 --gpu l40s --gpu-vram-gib 32
```

Combine with KV cache estimation (weights + KV cache):

```bash
uvx hf-mem --model-id Qwen/Qwen3.5-397B-A17B-FP8 --experimental --gpu h200
```

<img src="https://github.com/user-attachments/assets/b134b7c5-c7f9-42ca-b0a7-fe12d6961373" width="600" />

## (Optional) Agent Skills

Optionally, you can add `hf-mem` as an agent skill, which allows the underlying coding agent to discover and use it when provided as a [`SKILL.md`](.skills/hf-mem/SKILL.md).

More information can be found at [Anthropic Agent Skills and how to use them](https://github.com/anthropics/skills).

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
