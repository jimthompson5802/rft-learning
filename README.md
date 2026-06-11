# RFT Learning

This repository contains small experiments for supervised and reinforcement fine-tuning concepts with LoRA, including:

- `sft-lora-lesson.ipynb`: a LoRA + SFT arithmetic lesson
- `rft-lora-lesson.ipynb`: a LoRA + GRPO arithmetic lesson

## Requirements

- `uv`
- Homebrew Python 3.12 at `/opt/homebrew/bin/python3.12`

Note: this project is pinned to the resolved Homebrew interpreter path in `.python-version`:

`/opt/homebrew/opt/python@3.12/bin/python3.12`

## Setup With `uv`

1. Confirm the Python interpreter exists:

```bash
ls -l /opt/homebrew/bin/python3.12
```

2. Create or sync the virtual environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

This will create `.venv` and install the project dependencies from `pyproject.toml` / `uv.lock`.

## Using The Notebook

Open either `sft-lora-lesson.ipynb` or `rft-lora-lesson.ipynb` in your notebook editor and select the local kernel from `.venv`.

If your editor does not detect it automatically, use the Python interpreter at:

`/Users/jim/Desktop/genai/rft-learning/.venv/bin/python`

## Installed Dependencies

The project currently uses:

- `accelerate`
- `datasets`
- `ipykernel`
- `peft`
- `transformers`
- `trl[peft]`

## Notes

- `sft-lora-lesson.ipynb` demonstrates supervised fine-tuning with known target completions.
- The lesson currently depends on the `trl` API version installed in this repo.

## Number of batches for SFT and RFT

The difference is coming from `GRPOTrainer` semantics, not from a different dataset.

Both notebooks build the same raw data:
- `MAX_A = 21`, `MAX_B = 11` gives `20 * 10 = 200` examples total.
- `train_test_split(test_size=0.25, seed=42)` gives `150` train and `50` test in both notebooks.

What changes is how one “training step” is formed.

For `sft-lora-lesson.ipynb`:
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`

So SFT does normal supervised batching:
- micro-batches per epoch: `ceil(150 / 2) = 75`
- optimizer steps per epoch: `ceil(75 / 4) = 19`

That is why SFT typically looks like about `19` training steps.

For `rft-lora-lesson.ipynb`:
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- `num_generations=4`

In GRPO, TRL requires the effective batch size to be divisible by `num_generations`, and it uses those generations to form grouped RL updates. With 1 process, the effective batch size is:

`2 * 4 = 8`

Since `num_generations=4`, each GRPO update uses:

`8 / 4 = 2 prompts`

So steps per epoch become:

`ceil(150 / 2) = 75`

That is why RFT/GRPO shows many more training batches/steps than SFT, even though the train split is identical.

The official TRL docs call this out:
- GRPO `num_generations`: the effective batch size must be evenly divisible by it: https://huggingface.co/docs/trl/en/grpo_trainer
- GRPO generation batching defaults are tied to the effective training batch size / accumulation behavior: https://huggingface.co/docs/trl/en/grpo_trainer

So the short version is: SFT accumulates 4 ordinary mini-batches before 1 optimizer step, while GRPO uses that same effective batch budget to generate grouped rollouts, and with `num_generations=4` that changes the number of prompts consumed per update.

If you want, I can also add a tiny diagnostic cell to both notebooks that prints the expected step count before training so this is visible directly in the lessons.
