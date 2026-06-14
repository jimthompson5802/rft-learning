# Explanation of Reinforcement Fine-Tuning (RFT)

## What RFT Is

Reinforcement Fine-Tuning, or RFT, is a way to improve a model when we do not want to supervise it with a single fixed target completion for every prompt. Instead of saying "this exact output is correct," we let the model generate answers and then score those answers with a reward function.

The model is updated to produce outputs that earn higher rewards more often.

In that sense, SFT and RFT answer different training questions:

- SFT asks: "Can the model imitate this known good answer?"
- RFT asks: "Can the model discover answers that score well under this reward?"

## How That Looks In This Repo

This project uses a small arithmetic task:

- Prompt example: `What is 9 + 3? Respond exactly as <think>9 + 3</think><answer>...</answer>`
- Desired behavior:
  - the model should produce the correct sum
  - the model should follow the required tagged format
  - the `<think>` text should restate the ordered expression from the prompt

The RFT notebook does not train on a gold assistant completion. Instead, it lets the model generate completions and then scores them with three reward functions:

- `format_reward`
  - gives `0.5` when the response includes both `<think>` and `<answer>` tags
- `correctness_reward`
  - gives `1.0` when the extracted answer matches the expected sum
- `think_reward`
  - gives `1.0` when the first `<think>` block is exactly `x + y` or `What is x + y` for the ordered operands in the prompt

The total reward is:

```text
total_reward = format_reward + correctness_reward + think_reward
```

A completion earns up to `2.5`: `0.5` for the required tags, `1.0` for the correct answer, and `1.0` for valid think content. The components are independent, so a response can still earn one component when another fails.

## Why Use RFT

RFT is useful when exact target completions are either unavailable or too restrictive.

For many tasks, there may be multiple acceptable answers. A reward-based setup lets us define what we care about at a higher level, such as:

- correctness
- format compliance
- safety
- tool-use quality
- brevity or style preferences

Instead of forcing the model to copy one reference answer, we reward behaviors we want to encourage.

## What GRPO Is Doing Here

In this repository, RFT is implemented with GRPO. At a high level:

1. The trainer samples several completions for the same prompt.
2. Each completion is scored with the reward functions.
3. The trainer compares those sampled completions within the group.
4. The model is updated so higher-reward completions become more likely.

This is why the RFT notebook thinks in terms of prompts plus multiple generated candidates, rather than prompt-completion pairs like SFT.

## How Training Is Accomplished With RFT

In this project, RFT training happens as a repeated loop:

1. A batch of prompts is selected from the training set.
2. For each prompt, the model generates multiple candidate completions.
3. Each completion is scored by the reward functions.
4. GRPO compares the completions for the same prompt.
5. The model weights are updated to make higher-reward completions more likely in the future.

The important idea is that the model is not directly told which exact sentence to copy. Instead, it explores several possible answers and learns from which ones score better.

### Step 1: Sample prompts

The trainer starts with prompts from the arithmetic training set, such as:

```text
What is 9 + 3? Respond exactly as <think>9 + 3</think><answer>...</answer>
```

Each prompt also carries the expected numeric answer, which is needed for reward scoring.

### Step 2: Generate candidate completions

For each prompt, the GRPO setup in this repo samples multiple completions. In the notebook, `num_generations = 4`, so each prompt produces four candidate answers.

Those candidates might differ in:

- whether they use the required tags
- whether the math is correct
- how cleanly they structure the response

This gives the trainer several alternatives to compare for the same underlying task.

### Step 3: Score each completion with rewards

Each generated completion is passed through the reward functions:

- `format_reward`
- `correctness_reward`
- `think_reward`

For example:

- a correctly formatted but wrong answer might receive `0.5`
- a correctly formatted answer with valid think content but a wrong answer might receive `1.5`
- a completion satisfying all three reward checks receives `2.5`

At this point, every sampled completion has a scalar reward attached to it.

### Step 4: Compare completions within each prompt group

GRPO does not just look at whether a reward is high or low in isolation. It looks at the group of completions generated for the same prompt and compares them to one another.

That comparison tells the trainer which completions were better relative to the alternatives sampled for that prompt.

This matters because the model is learning a preference inside each group:

- less reward should make a completion less likely
- more reward should make a completion more likely

### Step 5: Update the model

After rewards are computed, the trainer performs gradient-based optimization. The update pushes the model toward generating tokens that appeared in stronger completions and away from tokens that appeared in weaker completions.

In this repo, the update is applied through LoRA adapters rather than by fully rewriting every model weight. That means:

- the base model stays frozen
- the LoRA adapter weights are the part being trained
- the trained adapter can later be saved and reused

So the model improves not by memorizing a single gold response, but by gradually shifting its probabilities toward reward-earning behaviors.

### What the model is really learning

Over many updates, the model learns patterns such as:

- include `<think>` and `<answer>` tags
- place the final numeric result inside `<answer>`
- produce answers that match the arithmetic target

It is not learning from a teacher-written completion at each step. It is learning from the consequences of its own generated outputs under the reward function.

### Short version

RFT training in this repo works like this:

1. generate several answers per prompt
2. score those answers
3. compare them within the prompt group
4. update LoRA weights so better-scoring answers become more probable

## Advantage

In reinforcement learning, reward tells us how good an outcome was. Advantage tells us how much better or worse that outcome was than some baseline expectation.

A simple mental model is:

```text
advantage = reward - baseline
```

Why does that matter? Because the model usually should not be updated based only on raw reward. What matters more is whether one sampled completion was better or worse than the alternatives.

In this repo, GRPO generates multiple completions for the same prompt and compares them as a group. That means the training signal is not just:

- "this completion got reward `2.5`"

It is more like:

- "this completion scored better than the other completions for the same prompt"

That relative comparison is where the idea of advantage enters.

### Advantage in GRPO

In GRPO, the baseline is group-relative rather than coming from a separate value model. Conceptually, the trainer looks at the rewards for the completions sampled for one prompt and determines which completions are above the group and which are below it.

So the signal is roughly:

```text
advantage = completion_reward - group_baseline
```

Where `group_baseline` is derived from the other sampled completions for that prompt.

The effect is:

- completions with positive advantage become more likely
- completions with negative advantage become less likely

### Example

Suppose one prompt produces four completions with total rewards:

```text
[0.0, 0.5, 1.5, 2.5]
```

The `2.5` completion is clearly stronger than the rest of the group, so it would have positive advantage. The `0.0` completion would have negative advantage. The model update would push probability mass toward outputs like the `2.5` example and away from outputs like the `0.0` example.

### Why advantage is helpful

Advantage makes the update more informative than raw reward alone.

It helps answer:

- which sampled completion was best for this prompt
- which sampled completion underperformed compared with the others
- how to shift the model relative to its own current behavior

So in this repo, the reward functions define the quality signal, and GRPO turns that signal into a relative learning signal that behaves like group-based advantage.

## What Is Used To Compute The Gradient Updates

The gradient updates in RFT are not computed directly from the reward number alone.

Instead, the update is based on a combination of:

- the model's log-probabilities for the tokens in each sampled completion
- the advantage for that completion
- GRPO stability terms such as clipping and KL-related control

A simple way to think about it is:

```text
gradient signal ~ advantage * grad(log p(sampled tokens))
```

This means the trainer looks at the tokens the model actually generated, checks how good that completion was relative to the others, and then adjusts the model so that better completions become more likely.

### What role the reward plays

The reward functions provide the scoring signal:

- `format_reward`
- `correctness_reward`
- `think_reward`

Those rewards are combined into a total reward for each sampled completion. GRPO then turns those rewards into a relative signal, which is the advantage.

So the flow is:

1. generate completions
2. compute rewards
3. convert rewards into group-relative advantage
4. use that advantage to weight the policy gradient update

### What role log-probabilities play

The model is a probabilistic generator. For every generated token, it assigns a probability.

During training, the optimizer uses the log-probabilities of the sampled completion tokens to compute how the parameters should move.

In effect:

- positive advantage pushes those sampled tokens to become more likely
- negative advantage pushes those sampled tokens to become less likely

So reward says whether the output was good, while log-probabilities determine how to change the model to encourage or discourage that output.

### What GRPO adds

GRPO does not rely on a separate value model in this setup. Instead, it uses group-relative comparisons among multiple completions sampled for the same prompt.

That means the update is guided by questions like:

- which completion scored best for this prompt
- which completion scored worst
- how far above or below the group a completion was

This relative structure is what makes the advantage signal meaningful.

### What is actually being updated in this repo

The trainer is attached to a LoRA configuration, so the optimization updates the LoRA adapter weights rather than all base-model weights.

That means:

- the base model stays frozen
- the adapter layers receive the gradient updates
- the saved artifact is the trained adapter

So in this repo, the gradient update is best understood as:

- rewards define quality
- GRPO converts quality into relative advantage
- token log-probabilities provide the differentiable learning signal
- LoRA adapter weights are the parameters being changed

## RFT vs. SFT In This Project

The SFT notebook and the RFT notebook use the same arithmetic task, but they teach the model in different ways.

### SFT

- training data includes a canonical assistant completion
- the model learns by imitating that completion
- the signal is token-level supervision

### RFT

- training data includes the prompt and expected answer
- the model generates candidate completions on its own
- the signal is a scalar reward based on output quality

SFT says, "learn this answer."

RFT says, "generate answers that score well."

## Strengths of RFT

- It can optimize for behaviors that are easier to score than to fully write out.
- It can support multiple valid answers.
- It can combine several goals into one reward signal.
- It can improve model behavior even when exact reference completions are not ideal.

## Limitations of RFT

- Reward quality matters a lot. A weak reward function teaches weak behavior.
- Models can learn shortcuts that exploit the reward instead of solving the real task.
- Training is usually more complex than SFT because it depends on repeated generation and scoring.
- Results can be noisy if sampling, reward design, or optimization settings are poor.

## A Simple Mental Model

One helpful way to think about RFT is:

- SFT is like showing worked examples and asking the model to imitate them.
- RFT is like letting the model try answers and then rewarding the better ones.

Both approaches can be useful together. A common pattern is:

1. start with SFT to teach the basic structure
2. use RFT to optimize for the behaviors that matter most

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
