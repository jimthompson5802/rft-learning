## Learning path

1. **Start with the idea**
   Reinforcement fine-tuning teaches a model by scoring its outputs with a **reward/grader**, rather than only showing it examples of correct answers. OpenAI describes the basic loop as: write a grader, upload prompts, start the RFT job, evaluate checkpoints, then deploy the tuned model. ([OpenAI Developers][1])

2. **Learn the key concepts**
   Focus on:

   * prompt dataset
   * model rollout / generated answer
   * reward function / grader
   * policy optimization
   * validation set
   * checkpoint evaluation

3. **Compare with supervised fine-tuning**
   Use **SFT** when you have good input/output examples. Use **RFT** when success can be scored by a rubric, test, evaluator, or executable check. OpenAI positions SFT around known good outputs, while RFT is useful for objectives that can be graded. ([OpenAI Developers][2])

4. **Use a local toy example first**
   Before using a hosted RFT API, write a tiny program where a “model” generates answers and a reward function scores them. This teaches the coding pattern without GPU cost.

5. **Then study real tooling**

   * OpenAI RFT docs and grader examples for hosted RFT. ([OpenAI Developers][1])
   * Hugging Face TRL for open-source RL fine-tuning, especially GRPOTrainer. TRL describes GRPO as generating completions, computing advantage, estimating KL divergence, and optimizing loss. ([Hugging Face][3])

## Simple learning program

This toy program trains a tiny policy to answer addition questions. The “reinforcement fine-tuning” idea is represented by:

* generate answer
* score answer
* increase probability of better answers

```python
import random
from collections import defaultdict

# Tiny "policy": for each question, store scores for candidate answers
policy = defaultdict(lambda: defaultdict(float))

def question():
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    return f"{a}+{b}", a + b

def generate_answer(q):
    # Candidate answers from 0..20
    candidates = list(range(21))
    scores = [policy[q][c] for c in candidates]

    # Soft preference: mostly exploit, sometimes explore
    if random.random() < 0.2:
        return random.choice(candidates)

    max_score = max(scores)
    best = [c for c in candidates if policy[q][c] == max_score]
    return random.choice(best)

def reward(predicted, correct):
    return 1.0 if predicted == correct else -0.1

def train(steps=5000, lr=0.1):
    for step in range(steps):
        q, correct = question()
        answer = generate_answer(q)
        r = reward(answer, correct)

        # Reinforce good answers, penalize bad ones
        policy[q][answer] += lr * r

def test(n=20):
    correct_count = 0

    for _ in range(n):
        q, correct = question()
        answer = generate_answer(q)

        if answer == correct:
            correct_count += 1

        print(f"{q} -> model: {answer}, correct: {correct}")

    print(f"\nAccuracy: {correct_count}/{n}")

train()
test()
```

## What this teaches

The important coding pattern is:

```text
prompt -> model output -> grader/reward -> training update
```

In real RFT, the “model” is an LLM, the reward might be a rubric or test suite, and the training update is handled by the platform or framework. Your main job is usually to design **good prompts, good validation data, and a reliable grader**.

[1]: https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning?utm_source=chatgpt.com "Reinforcement fine-tuning | OpenAI API"
[2]: https://developers.openai.com/api/docs/guides/supervised-fine-tuning?utm_source=chatgpt.com "Supervised fine-tuning | OpenAI API"
[3]: https://huggingface.co/docs/trl/index?utm_source=chatgpt.com "TRL - Transformers Reinforcement Learning"

--- 

## Lesson 2

Try a **“format + arithmetic correctness” GRPO exercise**.

Goal: fine-tune a small LLM to answer simple arithmetic in a required format:

```text
<think>...</think>
<answer>42</answer>
```

Use **Hugging Face TRL’s `GRPOTrainer`**, which supports GRPO and custom reward functions. TRL’s docs show GRPO reward functions for math/coding correctness and formatting-style rewards. ([Hugging Face][1])

## Exercise idea

Train on prompts like:

```text
What is 17 + 25?
```

Reward the model for:

1. Putting the answer inside `<answer>...</answer>`
2. Returning an integer
3. Returning the correct integer

## Minimal sketch

```python
import re
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

def make_dataset(n=200):
    rows = []
    for a in range(1, 21):
        for b in range(1, 11):
            rows.append({
                "prompt": f"What is {a} + {b}? Respond as <think>...</think><answer>...</answer>",
                "answer": str(a + b)
            })
    return Dataset.from_list(rows[:n])

dataset = make_dataset()

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text)
    return match.group(1).strip() if match else ""

def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        ok = bool(re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", text, re.DOTALL))
        rewards.append(0.5 if ok else 0.0)
    return rewards

def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for c, expected in zip(completions, answer):
        text = c[0]["content"] if isinstance(c, list) else c
        predicted = extract_answer(text)
        rewards.append(1.0 if predicted == expected else 0.0)
    return rewards

training_args = GRPOConfig(
    output_dir="grpo-arithmetic-demo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=128,
    max_completion_length=128,
    num_train_epochs=1,
    logging_steps=10,
)

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[format_reward, correctness_reward],
)

trainer.train()
```

Install:

```bash
pip install -U "trl" "transformers" "datasets" "accelerate"
```

For a MacBook, this may be slow or may require changes depending on memory/GPU support. A small cloud GPU is better.

## Why this is a good exercise

It teaches the real RFT/GRPO coding loop:

```text
prompt → LLM generates multiple completions → reward functions score them → GRPO updates the model
```

The key lesson is that much of GRPO work is not exotic RL code. It is designing **good reward functions** and making sure they cannot be gamed.

[1]: https://huggingface.co/docs/trl/en/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer"

---

## Lesson 3

Use LoRA by adding a `peft_config` to `GRPOTrainer`. TRL supports PEFT/LoRA by wrapping the base model when `peft_config` is provided, so the base model stays mostly frozen and only LoRA adapter weights are trained. ([Hugging Face][1])

```bash
pip install -U "trl[peft]" transformers datasets accelerate peft
```

Then modify the previous example like this:

```python
import re
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

def make_dataset():
    rows = []
    for a in range(1, 21):
        for b in range(1, 11):
            rows.append({
                "prompt": f"What is {a} + {b}? Respond as <think>...</think><answer>...</answer>",
                "answer": str(a + b)
            })
    return Dataset.from_list(rows)

dataset = make_dataset()

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text)
    return match.group(1).strip() if match else ""

def format_reward(completions, **kwargs):
    rewards = []

    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        ok = bool(re.search(
            r"<think>.*?</think>\s*<answer>.*?</answer>",
            text,
            re.DOTALL
        ))
        rewards.append(0.5 if ok else 0.0)

    return rewards

def correctness_reward(completions, answer, **kwargs):
    rewards = []

    for c, expected in zip(completions, answer):
        text = c[0]["content"] if isinstance(c, list) else c
        predicted = extract_answer(text)
        rewards.append(1.0 if predicted == expected else 0.0)

    return rewards

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
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
    max_prompt_length=128,
    max_completion_length=128,
    num_train_epochs=1,
    logging_steps=10,
    learning_rate=5e-5,
)

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[format_reward, correctness_reward],
    peft_config=lora_config,
)

trainer.train()

trainer.save_model("grpo-arithmetic-lora-adapter")
```

The important addition is this part:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "o_proj"],
)
```

and then:

```python
GRPOTrainer(
    ...,
    peft_config=lora_config,
)
```

For the learning exercise, start with only:

```python
target_modules=["q_proj", "v_proj"]
```

Then try the larger list later. The smaller list is easier to reason about; the larger list usually gives the adapter more capacity. PEFT describes LoRA as training low-rank adapter matrices instead of updating all model parameters, which reduces memory and training cost. ([Hugging Face][2])

[1]: https://huggingface.co/docs/trl/en/peft_integration?utm_source=chatgpt.com "PEFT Integration"
[2]: https://huggingface.co/docs/peft/en/developer_guides/lora?utm_source=chatgpt.com "LoRA"
