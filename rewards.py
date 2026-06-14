"""Shared reward computations for the arithmetic fine-tuning notebooks."""

import re
from dataclasses import dataclass
from typing import Any


FORMAT_REWARD = 0.5
CORRECTNESS_REWARD = 1.0


@dataclass(frozen=True, slots=True)
class RewardComputation:
    """The parsing and reward result for one completion."""

    predicted_answer: str
    is_formatted: bool
    is_correct: bool
    format_reward: float
    correctness_reward: float

    @property
    def total_reward(self) -> float:
        """Return the combined format and correctness reward."""
        return self.format_reward + self.correctness_reward


def completion_text(completion: Any) -> str:
    """Return text from either a plain or conversational completion."""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list) and completion:
        message = completion[0]
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    raise TypeError("completion must be a string or a non-empty list containing a content message")


def extract_answer(text: str) -> str:
    """Return the contents of the first answer tag, or an empty string."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def has_required_format(text: str) -> bool:
    """Return whether text contains consecutive think and answer tags."""
    return bool(
        re.search(
            r"<think>.*?</think>\s*<answer>.*?</answer>",
            text,
            re.DOTALL,
        )
    )


def compute_reward(completion: Any, expected_answer: str) -> RewardComputation:
    """Compute format, correctness, and total reward for one completion."""
    text = completion_text(completion)
    predicted_answer = extract_answer(text)
    is_formatted = has_required_format(text)
    is_correct = predicted_answer == expected_answer

    return RewardComputation(
        predicted_answer=predicted_answer,
        is_formatted=is_formatted,
        is_correct=is_correct,
        format_reward=FORMAT_REWARD if is_formatted else 0.0,
        correctness_reward=CORRECTNESS_REWARD if is_correct else 0.0,
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
