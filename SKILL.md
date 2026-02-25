---
name: hf-mem
description: CLI to estimate the required VRAM to load Safetensors models for inference from the Hugging Face Hub (Transformers, Diffusers and Sentence Transformers)
license: mit
---

# hf-mem

## What it does

Estimates inference memory requirements for models on the Hugging Face Hub via Safetensors metadata with HTTP Range requests.

## Requirements

- `uv` package manager (for `uvx` command)
- `HF_TOKEN` environment variable (only for gated/private models)

## When to use

- User asks about model VRAM/memory needs
- User wants to check if a model fits in their GPU
- User provides a Hugging Face model URL or model ID

## Usage

```bash
uvx hf-mem --model-id <org/model-name>
```

Add `--experimental` to include KV cache estimations for LLMs and VLMs.

Use GPU estimation flags when the user asks how many GPUs are needed:

- `--list-gpus` to print supported GPU presets (works without `--model-id`)
- `--gpu <name>` to estimate GPU count
- `--overhead <fraction>` to reserve VRAM headroom (for example `0.2`)
- `--gpu-vram-gib <value>` to override preset VRAM for cluster-specific configs

### Examples

- `uvx hf-mem --model-id black-forest-labs/FLUX.1-dev`
- `uvx hf-mem --model-id mistralai/Mistral-7B-v0.1 --experimental`
- `uvx hf-mem --list-gpus`
- `uvx hf-mem --model-id Qwen/Qwen3.5-397B-A17B-FP8 --gpu h100`
- `uvx hf-mem --model-id Qwen/Qwen3.5-397B-A17B-FP8 --gpu l40s --gpu-vram-gib 32`

## When it fails

- HTTP 401, if the model is gated/private, meaning you need to set `HF_TOKEN` with read access to it.
- HTTP 404, if the provided `--model-id` is not available on the Hugging Face Hub.
- RuntimeError, if none of `model.safetensors`, `model.safetensors.index.json`, or `model_index.json` is available.
