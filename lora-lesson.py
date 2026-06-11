import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType

MAX_A = 21
MAX_B = 11

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# ----------------------------
# Dataset
# ----------------------------

def make_dataset():
    """Build a small arithmetic dataset with strict output-format instructions."""
    rows = []

    for a in range(1, MAX_A):
        for b in range(1, MAX_B):
            rows.append({
                "prompt": f"What is {a} + {b}? Respond exactly as <think>...</think><answer>...</answer>",
                "answer": str(a + b),
            })

    return Dataset.from_list(rows)

dataset = make_dataset()
split = dataset.train_test_split(test_size=0.25, seed=42)

train_dataset = split["train"]
test_dataset = split["test"]

# ----------------------------
# Reward helpers
# ----------------------------

def extract_answer(text):
    """Return the contents of the first <answer> tag, or an empty string."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def has_required_format(text):
    """Check whether the response contains both think and answer tags."""
    return bool(re.search(
        r"<think>.*?</think>\s*<answer>.*?</answer>",
        text,
        re.DOTALL
    ))

def format_reward(completions, **kwargs):
    """Reward completions that follow the required XML-like response format."""
    rewards = []

    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        rewards.append(0.5 if has_required_format(text) else 0.0)

    return rewards

def correctness_reward(completions, answer, **kwargs):
    """Reward completions whose extracted answer matches the expected answer."""
    rewards = []

    for c, expected in zip(completions, answer):
        text = c[0]["content"] if isinstance(c, list) else c
        predicted = extract_answer(text)
        rewards.append(1.0 if predicted == expected else 0.0)

    return rewards

# ----------------------------
# Evaluation
# ----------------------------

def generate_response(model, tokenizer, prompt):
    """Generate a deterministic response for a single user prompt."""
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def evaluate_model(model, tokenizer, eval_dataset, label):
    """Run evaluation on a dataset and print summary metrics with examples."""
    model.eval()

    total = len(eval_dataset)
    correct = 0
    formatted = 0
    total_reward = 0.0

    examples = []

    for row in eval_dataset:
        prompt = row["prompt"]
        expected = row["answer"]

        text = generate_response(model, tokenizer, prompt)
        predicted = extract_answer(text)

        is_formatted = has_required_format(text)
        is_correct = predicted == expected

        format_score = 0.5 if is_formatted else 0.0
        correctness_score = 1.0 if is_correct else 0.0
        reward = format_score + correctness_score

        formatted += int(is_formatted)
        correct += int(is_correct)
        total_reward += reward

        if len(examples) < 5:
            examples.append({
                "prompt": prompt,
                "expected": expected,
                "generated": text,
                "predicted": predicted,
                "reward": reward,
            })

    print(f"\n=== {label} ===")
    print(f"Answer accuracy:   {correct}/{total} = {correct / total:.2%}")
    print(f"Format compliance: {formatted}/{total} = {formatted / total:.2%}")
    print(f"Average reward:    {total_reward / total:.3f}")

    print("\nSample generations:")
    for ex in examples:
        print("-" * 60)
        print("Prompt:   ", ex["prompt"])
        print("Expected: ", ex["expected"])
        print("Generated:", ex["generated"])
        print("Predicted:", ex["predicted"])
        print("Reward:   ", ex["reward"])

# ----------------------------
# Load base model
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Baseline before RFT
evaluate_model(
    base_model,
    tokenizer,
    test_dataset,
    label="Before GRPO + LoRA"
)

# ----------------------------
# LoRA + GRPO training
# ----------------------------

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
)

trainer = GRPOTrainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[format_reward, correctness_reward],
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("grpo-arithmetic-lora-adapter")

# ----------------------------
# Evaluate after RFT
# ----------------------------

trained_model = trainer.model

evaluate_model(
    trained_model,
    tokenizer,
    test_dataset,
    label="After GRPO + LoRA"
)
