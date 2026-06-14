import unittest

from rewards import (
    CORRECTNESS_REWARD,
    FORMAT_REWARD,
    compute_reward,
    correctness_reward,
    extract_answer,
    format_reward,
    has_required_format,
)


class RewardTests(unittest.TestCase):
    def test_compute_reward_for_correct_formatted_completion(self) -> None:
        result = compute_reward(
            "<think>2 + 3 = 5</think><answer>5</answer>",
            "5",
        )

        self.assertEqual(result.predicted_answer, "5")
        self.assertTrue(result.is_formatted)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.format_reward, FORMAT_REWARD)
        self.assertEqual(result.correctness_reward, CORRECTNESS_REWARD)
        self.assertEqual(result.total_reward, 1.5)

    def test_compute_reward_scores_format_and_correctness_independently(self) -> None:
        result = compute_reward("<answer>5</answer>", "5")

        self.assertFalse(result.is_formatted)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.total_reward, CORRECTNESS_REWARD)

    def test_parsing_helpers(self) -> None:
        text = "<think>work</think>\n<answer> 12 </answer>"

        self.assertEqual(extract_answer(text), "12")
        self.assertTrue(has_required_format(text))

    def test_trainer_reward_functions_accept_conversational_completions(self) -> None:
        completions = [
            [{"role": "assistant", "content": "<think>work</think><answer>7</answer>"}],
            [{"role": "assistant", "content": "7"}],
        ]

        self.assertEqual(format_reward(completions), [FORMAT_REWARD, 0.0])
        self.assertEqual(
            correctness_reward(completions, answer=["7", "7"]),
            [CORRECTNESS_REWARD, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
