import unittest

from rewards import (
    CORRECTNESS_REWARD,
    FORMAT_REWARD,
    THINK_REWARD,
    compute_reward,
    correctness_reward,
    extract_answer,
    extract_operands,
    extract_think,
    format_reward,
    has_required_format,
    has_valid_think,
    think_reward,
)


class RewardTests(unittest.TestCase):
    def test_compute_reward_for_correct_formatted_completion(self) -> None:
        result = compute_reward(
            "<think>2 + 3</think><answer>5</answer>",
            "5",
            "What is 2 + 3? Respond exactly as requested.",
        )

        self.assertEqual(result.predicted_answer, "5")
        self.assertEqual(result.think_text, "2 + 3")
        self.assertTrue(result.is_formatted)
        self.assertTrue(result.is_correct)
        self.assertTrue(result.is_think_valid)
        self.assertEqual(result.format_reward, FORMAT_REWARD)
        self.assertEqual(result.correctness_reward, CORRECTNESS_REWARD)
        self.assertEqual(result.think_reward, THINK_REWARD)
        self.assertEqual(result.total_reward, 2.5)

    def test_compute_reward_scores_components_independently(self) -> None:
        result = compute_reward(
            "<think>2 + 3 = 5</think><answer>5</answer>",
            "5",
            "What is 2 + 3?",
        )

        self.assertTrue(result.is_formatted)
        self.assertTrue(result.is_correct)
        self.assertFalse(result.is_think_valid)
        self.assertEqual(result.think_reward, 0.0)
        self.assertEqual(result.total_reward, FORMAT_REWARD + CORRECTNESS_REWARD)

        think_only_result = compute_reward(
            "<think>2 + 3</think>",
            "5",
            "What is 2 + 3?",
        )
        self.assertFalse(think_only_result.is_formatted)
        self.assertFalse(think_only_result.is_correct)
        self.assertTrue(think_only_result.is_think_valid)
        self.assertEqual(think_only_result.total_reward, THINK_REWARD)

    def test_parsing_helpers(self) -> None:
        text = "<think> 4 + 8 </think>\n<answer> 12 </answer>"

        self.assertEqual(extract_answer(text), "12")
        self.assertEqual(extract_think(text), "4 + 8")
        self.assertEqual(extract_operands("What is 4 + 8?"), ("4", "8"))
        self.assertTrue(has_required_format(text))

    def test_valid_think_accepts_only_the_two_exact_forms(self) -> None:
        prompt = "What is 2 + 3? Respond exactly as requested."

        self.assertTrue(has_valid_think("<think> 2 + 3 </think>", prompt))
        self.assertTrue(has_valid_think("<think>What is 2 + 3</think>", prompt))

        invalid_values = [
            "2 + 3 = 5",
            "3 + 2",
            "2+3",
            "What is 2 + 3?",
            "The expression is 2 + 3",
            "2 + 4",
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertFalse(has_valid_think(f"<think>{value}</think>", prompt))

    def test_valid_think_rejects_missing_tags_and_malformed_prompts(self) -> None:
        self.assertFalse(has_valid_think("2 + 3", "What is 2 + 3?"))
        self.assertFalse(has_valid_think("<think>2 + 3</think>", "Add 2 and 3."))

    def test_trainer_reward_functions_accept_conversational_completions(self) -> None:
        completions = [
            [{"role": "assistant", "content": "<think>2 + 5</think><answer>7</answer>"}],
            [{"role": "assistant", "content": "7"}],
        ]
        prompts = [
            [{"role": "user", "content": "What is 2 + 5?"}],
            [{"role": "user", "content": "What is 2 + 5?"}],
        ]

        self.assertEqual(format_reward(completions), [FORMAT_REWARD, 0.0])
        self.assertEqual(
            correctness_reward(completions, answer=["7", "7"]),
            [CORRECTNESS_REWARD, 0.0],
        )
        self.assertEqual(
            think_reward(completions, prompts=prompts),
            [THINK_REWARD, 0.0],
        )

    def test_think_reward_uses_the_first_think_block(self) -> None:
        completions = [
            "<think>2 + 3</think><think>wrong</think><answer>5</answer>",
            "<think>wrong</think><think>2 + 3</think><answer>5</answer>",
        ]

        self.assertEqual(
            think_reward(completions, prompts=["What is 2 + 3?"] * 2),
            [THINK_REWARD, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
