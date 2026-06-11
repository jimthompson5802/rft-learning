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

## Reproducibility

Metrics from the out-of-sample test data set.

### RFT

|Model|epochs|Before RFT|After RFT|
|-----|:----:|----------|---------|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-1.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-2.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-3.png)|
|Qwen2.5-0.5B-Instruct|3|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-4.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-5.png)|
|Qwen2.5-1.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-6.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-7.png)|





### SFT

|Model|epochs|Before SFT|After SFT|
|-----|:----:|----------|---------|
|Qwen2.5-0.5B-Instruct|1|![](./images/sft-eval-run-before.png)|![](./images/sft-eval-run-1.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/sft-eval-run-before.png)|![](./images/sft-eval-run-2.png)|



## Number of batches for SFT and RFT

Both notebooks start from the same amount of data. The different batch counts come from what a “batch” means to each trainer.

Here are the shared values first:

- `MAX_A = 21`
  - The code loops over `a` from `1` to `20`.
  - That gives `20` possible `a` values.

- `MAX_B = 11`
  - The code loops over `b` from `1` to `10`.
  - That gives `10` possible `b` values.

- Total examples:
  - `20 * 10 = 200`
  - Each example is one addition problem like `"What is 9 + 3?"`

- `test_size = 0.25`
  - 25% of the 200 examples go to test.
  - `200 * 0.25 = 50` test examples.

- Training examples:
  - `200 - 50 = 150`
  - So both notebooks train on exactly `150` examples.

For the SFT notebook, these values mean:

- `per_device_train_batch_size = 2`
  - Each forward/backward pass uses `2` training examples.

- `gradient_accumulation_steps = 4`
  - The trainer does `4` of those small passes before taking `1` optimizer step.

- So one optimizer step in SFT represents:
  - `2 examples per mini-batch * 4 mini-batches = 8 examples`

- Number of SFT mini-batches in one epoch:
  - `150 training examples / 2 examples per mini-batch = 75 mini-batches`

- Number of SFT optimizer steps in one epoch:
  - `75 mini-batches / 4 accumulation steps = 18.75`
  - Rounded up by the trainer, that is about `19 optimizer steps`

So in SFT:
- `2` = examples processed at once
- `4` = how many of those small batches are accumulated
- `8` = effective examples contributing to one parameter update
- `19` = approximate update steps for the epoch

For the RFT / GRPO notebook, the shared values `2` and `4` still exist, but there is one extra value:

- `num_generations = 4`
  - For each prompt, GRPO generates `4` different candidate completions.
  - Those 4 completions form one comparison group for reward computation.

Now the other values mean:

- `per_device_train_batch_size = 2`
  - GRPO starts with `2` prompts per micro-batch.

- `gradient_accumulation_steps = 4`
  - It accumulates across `4` micro-batches.

- Effective GRPO batch size:
  - `2 prompts per micro-batch * 4 accumulation steps = 8 prompt slots`

- `num_generations = 4`
  - Each actual training prompt needs `4` generated completions.

- So the number of distinct prompts represented in one GRPO update is:
  - `8 total slots / 4 generations per prompt = 2 prompts`

That is the key difference.

In SFT:
- one update covers `8 distinct training examples`

In GRPO:
- one update covers `2 distinct prompts`, because each prompt is expanded into `4` sampled completions

Then the GRPO epoch length becomes:

- `150 training prompts / 2 prompts per update = 75 updates`

So in GRPO:
- `2` = prompts loaded per micro-batch
- `4` = accumulation steps
- `8` = effective prompt-generation slots in one update
- `4` = generations sampled for each prompt
- `2` = distinct prompts consumed per final GRPO update
- `75` = approximate updates for the epoch

That is why the batch or step count is different even though both notebooks start with the same `150` training examples. The SFT trainer uses each example once per update flow, while the GRPO trainer expands each prompt into `4` generated candidates, so each update consumes fewer distinct prompts.