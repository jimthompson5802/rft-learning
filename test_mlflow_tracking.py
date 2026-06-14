import unittest
from contextlib import contextmanager
from enum import Enum
import json
from unittest.mock import patch

import mlflow_tracking


class FakeRun:
    def __init__(self, run_name, nested, tags):
        self.run_name = run_name
        self.nested = nested
        self.tags = tags


class FakeMlflow:
    def __init__(self):
        self.tracking_uri = None
        self.experiment_name = None
        self.experiments = {}
        self.logged_params = []
        self.param_values = {}
        self.logged_metrics = []
        self.logged_artifacts = []
        self.started_runs = []
        self.tracking = self

    def set_tracking_uri(self, tracking_uri):
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def MlflowClient(self):
        return self

    def get_experiment_by_name(self, experiment_name):
        return self.experiments.get(experiment_name)

    def create_experiment(self, name, artifact_location=None):
        self.experiments[name] = FakeRun(
            run_name=name,
            nested=False,
            tags={"artifact_location": artifact_location},
        )
        return name

    @contextmanager
    def start_run(self, run_name=None, nested=False, tags=None):
        run = FakeRun(run_name=run_name, nested=nested, tags=tags or {})
        self.started_runs.append(run)
        yield run

    def log_params(self, params):
        for key, value in params.items():
            if key in self.param_values and self.param_values[key] != value:
                raise ValueError(f"Cannot change parameter {key}")
            self.param_values[key] = value
        self.logged_params.append(params)

    def log_metric(self, key, value, step=None):
        self.logged_metrics.append((key, value, step))

    def log_artifact(self, path, artifact_path=None):
        self.logged_artifacts.append((path, artifact_path))


class ExampleStrategy(Enum):
    STEPS = "steps"


class MlflowTrackingTests(unittest.TestCase):
    def test_serialize_config_handles_enum_members_and_classes(self):
        self.assertEqual(
            mlflow_tracking.serialize_config(ExampleStrategy.STEPS),
            "steps",
        )
        self.assertEqual(
            mlflow_tracking.serialize_config(ExampleStrategy),
            f"{ExampleStrategy.__module__}.{ExampleStrategy.__qualname__}",
        )

    def test_real_peft_and_trl_configs_are_json_serializable(self):
        from peft import LoraConfig, TaskType
        from trl import SFTConfig

        configs = {
            "lora": LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
            ),
            "training": SFTConfig(
                output_dir="test-output",
                use_cpu=True,
            ),
        }

        serialized = mlflow_tracking.serialize_config(configs)
        flattened = mlflow_tracking.flatten_params(configs)

        json.dumps(serialized, sort_keys=True)
        self.assertEqual(flattened["lora.task_type"], "CAUSAL_LM")
        self.assertEqual(flattened["training.lr_scheduler_type"], "linear")

    def test_flatten_params_serializes_nested_values(self):
        flattened = mlflow_tracking.flatten_params(
            {
                "training": {"epochs": 3, "use_cpu": False},
                "targets": ["q_proj", "v_proj"],
            }
        )

        self.assertEqual(flattened["training.epochs"], "3")
        self.assertEqual(flattened["training.use_cpu"], "false")
        self.assertEqual(flattened["targets"], '["q_proj", "v_proj"]')

    def test_metrics_only_filters_non_numeric_values(self):
        metrics = mlflow_tracking.metrics_only(
            {
                "accuracy": 1.0,
                "count": 5,
                "phase": "baseline",
                "finite": float("inf"),
                "flag": True,
            },
            prefix="baseline",
        )

        self.assertEqual(
            metrics,
            {
                "baseline.accuracy": 1.0,
                "baseline.count": 5.0,
            },
        )

    def test_initialize_tracking_uses_local_store_by_default(self):
        fake_mlflow = FakeMlflow()
        with patch.object(mlflow_tracking, "mlflow", fake_mlflow):
            setup = mlflow_tracking.initialize_tracking()

        self.assertEqual(fake_mlflow.tracking_uri, setup.tracking_uri)
        self.assertEqual(fake_mlflow.experiment_name, mlflow_tracking.DEFAULT_EXPERIMENT_NAME)
        self.assertTrue(setup.tracking_uri.endswith("/mlflow.db"))
        self.assertIsNotNone(setup.artifact_path)
        self.assertTrue(str(setup.artifact_path).endswith("mlartifacts"))
        self.assertIn(mlflow_tracking.DEFAULT_EXPERIMENT_NAME, fake_mlflow.experiments)

    def test_parent_and_child_runs_set_expected_metadata(self):
        fake_mlflow = FakeMlflow()
        with patch.object(mlflow_tracking, "mlflow", fake_mlflow):
            with mlflow_tracking.start_parent_run(
                notebook_name="sft-lora-lesson.ipynb",
                notebook_type="sft",
                tags={"model_name": "Qwen"},
            ):
                with mlflow_tracking.start_child_run("baseline_eval"):
                    pass

        self.assertEqual(len(fake_mlflow.started_runs), 2)
        parent_run, child_run = fake_mlflow.started_runs
        self.assertFalse(parent_run.nested)
        self.assertEqual(parent_run.tags["run_level"], "parent")
        self.assertEqual(parent_run.tags["notebook_type"], "sft")
        self.assertEqual(parent_run.tags["model_name"], "Qwen")
        self.assertTrue(child_run.nested)
        self.assertEqual(child_run.tags["run_level"], "child")
        self.assertEqual(child_run.tags["phase"], "baseline_eval")

    def test_log_eval_result_logs_metrics_and_artifacts(self):
        fake_mlflow = FakeMlflow()
        result = {
            "label": "Before SFT + LoRA",
            "metrics": {"accuracy": 0.5, "avg_reward": 1.25},
            "settings": {"num_generations": 1, "do_sample": False},
            "samples": [{"prompt": "What is 1 + 1?", "generated": "<answer>2</answer>"}],
            "rows": [{"prompt": "What is 1 + 1?", "score": 1.5}],
        }
        with patch.object(mlflow_tracking, "mlflow", fake_mlflow):
            mlflow_tracking.log_eval_result(
                phase="baseline",
                result=result,
                parent_metric_prefix="baseline",
            )

        logged_metric_keys = [key for key, _, _ in fake_mlflow.logged_metrics]
        self.assertIn("accuracy", logged_metric_keys)
        self.assertIn("avg_reward", logged_metric_keys)
        self.assertIn("baseline.accuracy", logged_metric_keys)
        self.assertIn("baseline.avg_reward", logged_metric_keys)
        self.assertEqual(fake_mlflow.param_values["label"], "Before SFT + LoRA")
        self.assertEqual(fake_mlflow.param_values["settings.do_sample"], "false")
        self.assertEqual(fake_mlflow.param_values["settings.num_generations"], "1")
        self.assertEqual(len(fake_mlflow.logged_artifacts), 5)

    def test_log_eval_result_reuses_flat_param_names_across_separate_runs(self):
        greedy_mlflow = FakeMlflow()
        sampled_mlflow = FakeMlflow()
        greedy_result = {
            "label": "Greedy",
            "metrics": {"accuracy": 1.0},
            "settings": {"num_generations": 1, "do_sample": False, "seed": None},
        }
        sampled_result = {
            "label": "Sampled",
            "metrics": {"accuracy": 0.9},
            "settings": {"num_generations": 4, "do_sample": True, "seed": 42},
        }

        with patch.object(mlflow_tracking, "mlflow", greedy_mlflow):
            mlflow_tracking.log_eval_result("fine_tuned_eval_greedy", greedy_result)
        with patch.object(mlflow_tracking, "mlflow", sampled_mlflow):
            mlflow_tracking.log_eval_result("fine_tuned_eval_sampled", sampled_result)

        self.assertEqual(greedy_mlflow.param_values["label"], "Greedy")
        self.assertEqual(greedy_mlflow.param_values["settings.do_sample"], "false")
        self.assertEqual(sampled_mlflow.param_values["label"], "Sampled")
        self.assertEqual(sampled_mlflow.param_values["settings.do_sample"], "true")
        self.assertEqual([key for key, _, _ in greedy_mlflow.logged_metrics], ["accuracy"])
        self.assertEqual([key for key, _, _ in sampled_mlflow.logged_metrics], ["accuracy"])

    def test_log_eval_result_skips_parent_mirroring_when_prefix_is_omitted(self):
        fake_mlflow = FakeMlflow()
        result = {
            "label": "Before SFT + LoRA",
            "metrics": {"accuracy": 0.5, "avg_reward": 1.25},
            "settings": {"num_generations": 1, "do_sample": False},
        }

        with patch.object(mlflow_tracking, "mlflow", fake_mlflow):
            mlflow_tracking.log_eval_result("baseline_eval", result)

        logged_metric_keys = {key for key, _, _ in fake_mlflow.logged_metrics}
        self.assertEqual(logged_metric_keys, {"accuracy", "avg_reward"})
        self.assertEqual(fake_mlflow.param_values["label"], "Before SFT + LoRA")
        self.assertEqual(fake_mlflow.param_values["settings.do_sample"], "false")

    def test_log_eval_result_does_not_prefix_flat_params(self):
        fake_mlflow = FakeMlflow()
        result = {
            "label": "Greedy",
            "metrics": {"accuracy": 1.0},
            "settings": {"num_generations": 1, "do_sample": False},
        }

        with patch.object(mlflow_tracking, "mlflow", fake_mlflow):
            mlflow_tracking.log_eval_result("fine_tuned_eval", result)

        self.assertEqual(fake_mlflow.param_values["label"], "Greedy")
        self.assertEqual(fake_mlflow.param_values["settings.num_generations"], "1")
        self.assertNotIn("eval.fine_tuned_eval.label", fake_mlflow.param_values)


if __name__ == "__main__":
    unittest.main()
