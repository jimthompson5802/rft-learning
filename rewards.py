"""Shared reward computations for the arithmetic fine-tuning notebooks."""

import re
from dataclasses import dataclass
from typing import Any


FORMAT_REWARD = 0.5
CORRECTNESS_REWARD = 1.0
THINK_REWARD = 1.0


@dataclass(frozen=True, slots=True)
class RewardComputation:
    """The parsing and reward result for one completion."""

    predicted_answer: str
    think_text: str
    is_formatted: bool
    is_correct: bool
    is_think_valid: bool
    format_reward: float
    correctness_reward: float
    think_reward: float

    @property
    def total_reward(self) -> float:
        """Return the combined format, correctness, and think reward."""
        return self.format_reward + self.correctness_reward + self.think_reward


def completion_text(completion: Any) -> str:
    """Return text from either a plain or conversational completion."""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list) and completion:
        message = completion[0]
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    raise TypeError("completion must be a string or a non-empty list containing a content message")


def prompt_text(prompt: Any) -> str:
    """Return text from either a plain or conversational prompt."""
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        return "\n".join(
            message["content"]
            for message in prompt
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )

    raise TypeError("prompt must be a string or a list containing content messages")


def extract_answer(text: str) -> str:
    """Return the contents of the first answer tag, or an empty string."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_think(text: str) -> str:
    """Return the contents of the first think tag, or an empty string."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_operands(prompt: Any) -> tuple[str, str] | None:
    """Return the ordered addition operands from an arithmetic prompt."""
    match = re.search(r"What is (-?\d+) \+ (-?\d+)\?", prompt_text(prompt))
    return match.groups() if match else None


def has_valid_think(text: str, prompt: Any) -> bool:
    """Return whether the first think tag exactly restates the prompt expression."""
    operands = extract_operands(prompt)
    if operands is None:
        return False

    x, y = operands
    think_text = extract_think(text)
    return think_text in {f"{x} + {y}", f"What is {x} + {y}"}


def has_required_format(text: str) -> bool:
    """Return whether text contains consecutive think and answer tags."""
    return bool(
        re.search(
            r"<think>.*?</think>\s*<answer>.*?</answer>",
            text,
            re.DOTALL,
        )
    )


def compute_reward(completion: Any, expected_answer: str, prompt: Any) -> RewardComputation:
    """Compute all reward components and the total for one completion."""
    text = completion_text(completion)
    predicted_answer = extract_answer(text)
    think_text = extract_think(text)
    is_formatted = has_required_format(text)
    is_correct = predicted_answer == expected_answer
    is_think_valid = has_valid_think(text, prompt)

    return RewardComputation(
        predicted_answer=predicted_answer,
        think_text=think_text,
        is_formatted=is_formatted,
        is_correct=is_correct,
        is_think_valid=is_think_valid,
        format_reward=FORMAT_REWARD if is_formatted else 0.0,
        correctness_reward=CORRECTNESS_REWARD if is_correct else 0.0,
        think_reward=THINK_REWARD if is_think_valid else 0.0,
    )


def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Return the format reward for each trainer completion."""
    del kwargs
    return [
        FORMAT_REWARD if has_required_format(completion_text(completion)) else 0.0
        for completion in completions
    ]


def correctness_reward(
    completions: list[Any],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """Return the correctness reward for each trainer completion."""
    del kwargs
    return [
        CORRECTNESS_REWARD
        if extract_answer(completion_text(completion)) == expected_answer
        else 0.0
        for completion, expected_answer in zip(completions, answer)
    ]


def think_reward(
    completions: list[Any],
    prompts: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Return the think-content reward for each trainer completion."""
    del kwargs
    return [
        THINK_REWARD if has_valid_think(completion_text(completion), prompt) else 0.0
        for completion, prompt in zip(completions, prompts)
    ]
