# Reinforcement Fine-Tuning Learning Repo

This repository contains small experiments for supervised and reinforcement fine-tuning concepts with LoRA, including:

- `sft-lora-lesson.ipynb`: a LoRA + SFT arithmetic lesson
- `rft-lora-lesson.ipynb`: a LoRA + GRPO arithmetic lesson
- `rft-lora-early-stopping.ipynb`: a GRPO + LoRA lesson with validation-based moving-average early stopping
- `grpo-completion-exploration.ipynb`: a notebook for inspecting GRPO completion Parquet files

## Fine-Tuning Task

The training notebooks fine-tune a model on a simple arithmetic addition task. Given prompts such as `What is 9 + 3?`, the model is trained to respond in the exact format `<think>...</think><answer>...</answer>`, where the `<answer>` tag contains the correct sum.

The early stopping notebook uses the same task, but splits the data into train, validation, and test subsets so GRPO training can stop when held-out reward stops improving.

For details on how RFT works see [RFT Explanation document](./rft-explanation.md)

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

Open `sft-lora-lesson.ipynb`, `rft-lora-lesson.ipynb`, `rft-lora-early-stopping.ipynb`, or `grpo-completion-exploration.ipynb` in your notebook editor and select the local kernel from `.venv`.

`rft-lora-early-stopping.ipynb` extends the basic GRPO lesson with:

- a train/validation/test split
- periodic evaluation during training
- a custom moving-average early stopping callback
- best-checkpoint reloading based on `smoothed_reward`

`grpo-completion-exploration.ipynb` reads a file from `grpo-arithmetic-lora-demo/completions` into a pandas DataFrame. Set `parquet_filename` at the top of the notebook to choose the file, and optionally change `records_to_show` to control how many records are rendered. The notebook prints the requested count, the total number of records in the file, and the number actually shown, then displays each record with the full `prompt` and `completion` text.

If your editor does not detect it automatically, use the Python interpreter at:

`/Users/jim/Desktop/genai/rft-learning/.venv/bin/python`

## Installed Dependencies

The project currently uses:

- `accelerate`
- `datasets`
- `ipykernel`
- `pandas`
- `peft`
- `pyarrow`
- `transformers`
- `trl[peft]`

## Notes

- `sft-lora-lesson.ipynb` demonstrates supervised fine-tuning with known target completions.
- `rft-lora-early-stopping.ipynb` demonstrates validation-based GRPO training with moving-average early stopping and best-model selection.
- `grpo-completion-exploration.ipynb` is useful for inspecting saved GRPO completion samples after training.
- The lesson currently depends on the `trl` API version installed in this repo.

## Reproducibility

Metrics from the out-of-sample test data set.

### Reinforcement Fine-Tuning (RFT)

#### Initial Test

```
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
training_args = GRPOConfig(
    output_dir="grpo-arithmetic-lora-demo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=64,
    num_train_epochs=1,
    logging_steps=10,
    learning_rate=5e-5,
    log_completions=True,
)
```


|Model|epochs|Before RFT|After RFT|
|-----|:----:|----------|---------|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-1.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-2.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-3.png)|
|Qwen2.5-0.5B-Instruct|3|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-4.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-5.png)|
|Qwen2.5-1.5B-Instruct|1|![](./images/rft-eval-run-6-before.png)|![](./images/rft-eval-run-6.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-7.png)|

#### Increased Batch Size & Epochs

```
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
training_args = GRPOConfig(
    output_dir="grpo-arithmetic-lora-demo",
    per_device_train_batch_size=8,            <==
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=64,
    num_train_epochs=4,                       <==
    logging_steps=10,
    learning_rate=5e-5,
    log_completions=True,
)
```

|Model|epochs|Before RFT|After RFT|
|-----|:----:|----------|---------|
|Qwen2.5-0.5B-Instruct|4|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-8.png)|
|Qwen2.5-0.5B-Instruct|4|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-9.png)|
|Qwen2.5-0.5B-Instruct|4|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-10.png)|


#### Early Stopping

```
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

moving_average_window = 5
early_stopping_patience = 5
early_stopping_threshold = 0.0

training_args = GRPOConfig(
    output_dir="grpo-arithmetic-lora-early-stopping-demo",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=4,
    num_generations_eval=4,
    max_completion_length=64,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    min_p=None,
    repetition_penalty=1.0,
    num_train_epochs=8,
    logging_steps=10,
    learning_rate=5e-5,
    log_completions=False, #True,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="smoothed_reward",
    greater_is_better=True,
    save_total_limit=2,
)
```

|Model|Actual/Max epochs|Before RFT|After RFT|
|-----|:----:|----------|---------|
|Qwen2.5-0.5B-Instruct|3/8|![](./images/rft-eval-run-before.png)|![](./images/rft-eval-run-11.png)|


### Supervised Fine-Tuning (SFT)

|Model|epochs|Before SFT|After SFT|
|-----|:----:|----------|---------|
|Qwen2.5-0.5B-Instruct|1|![](./images/sft-eval-run-before.png)|![](./images/sft-eval-run-1.png)|
|Qwen2.5-0.5B-Instruct|1|![](./images/sft-eval-run-before.png)|![](./images/sft-eval-run-2.png)|
