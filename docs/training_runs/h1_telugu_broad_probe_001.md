# H1 Telugu Broad Adapter Probe 001

## Purpose

Train a broad Telugu fluency LoRA adapter for `google/gemma-4-E4B-it`.

Hypothesis:

- 16-bit LoRA on all text projection modules can improve Telugu fluency and instruction following without full-model finetuning.
- This adapter can later be merged or stacked with a medical English adapter.

## Model

- Base: `google/gemma-4-E4B-it`
- Precision: bf16 compute on H100
- Base load: not 4-bit, not 8-bit
- Context: 2048 tokens

## Adapter

Target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

LoRA:

- `r`: 16
- `alpha`: 16
- `dropout`: 0.0
- `bias`: none
- `use_gradient_checkpointing`: unsloth
- `use_rslora`: false

## Dataset

- Train dataset: `RohithMidigudla/gemma-health-telugu-sft-balanced`
- Adapter profile: `telugu`
- Local config source: `config.yaml`

## Probe Config

```yaml
training:
  run_name: h1_telugu_broad_probe_001
  batch_size: 4
  eval_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.00005
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.01
  optim: adamw_8bit
  max_steps: 500
  save_steps: 500
  eval_steps: 500
  logging_steps: 5
  skip_eval: false
  resume_from_checkpoint: true
```

Effective batch size:

```text
4 * 8 * 1 GPU = 32
```

## Storage

Modal Volume:

```text
gemma-health-unsloth-checkpoints
```

Adapter path:

```text
/checkpoints/adapters/h1_telugu_broad_probe_001/telugu
```

Checkpoints:

```text
/checkpoints/adapters/h1_telugu_broad_probe_001/telugu/checkpoint-*
```

## Hugging Face Output

Final adapter repo:

```text
RohithMidigudla/gemma-health-telugu-lora-h1
```

Rules:

- On success, push final adapter folder to repo root.
- On failure, push latest checkpoint to `last-checkpoint/`.

## W&B

Project:

```text
gemma-health-adapters
```

Run naming:

```text
<config.training.run_name>-telugu
```

## Smoke Command

```bash
uv run python scripts/modal_train_adapter.py \
  --adapter telugu \
  --hub-dataset-id RohithMidigudla/gemma-health-telugu-sft-balanced \
  --gpu H100 \
  --max-steps 1 \
  --max-examples 5 \
  --output-root /checkpoints/adapters/h1_smoke \
  --hub-model-id RohithMidigudla/gemma-health-telugu-lora-h1-smoke \
  --push-to-hub \
  --hf-secret hf-secret \
  --wandb-secret wandb-secret \
  --execute
```

## Full Probe Command

Use detach so the job continues if local laptop disconnects.

```bash
uv run python scripts/modal_train_adapter.py \
  --adapter telugu \
  --hub-dataset-id RohithMidigudla/gemma-health-telugu-sft-balanced \
  --gpu H100 \
  --max-steps 5500 \
  --output-root /checkpoints/adapters/h1_telugu_broad_probe_001 \
  --hub-model-id RohithMidigudla/gemma-health-telugu-lora-h1 \
  --push-to-hub \
  --hf-secret hf-secret \
  --wandb-secret wandb-secret \
  --detach \
  --execute
```

## 8x H100 DDP Command

Use this to keep the same effective batch size as the stable single-node plan while distributing work across 8 H100s:

```text
4 per GPU * 1 grad accumulation * 8 GPUs = 32 effective batch
```

```bash
uv run python scripts/modal_train_adapter.py \
  --adapter telugu \
  --hub-dataset-id RohithMidigudla/gemma-health-telugu-sft-balanced \
  --gpu H100:8 \
  --max-steps 5500 \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --output-root /checkpoints/adapters/h1_telugu_broad_probe_001 \
  --hub-model-id RohithMidigudla/gemma-health-telugu-lora-h1 \
  --push-to-hub \
  --hf-secret hf-secret \
  --wandb-secret wandb-secret \
  --detach \
  --execute
```

## Success Criteria

- Model loads on H100.
- W&B shows loss, learning rate, runtime, and system metrics.
- Checkpoints appear every 500 steps.
- Final adapter appears in Modal Volume.
- Final adapter is pushed to Hugging Face on success.
- Latest checkpoint is pushed to Hugging Face on failure.
