"""
Reinforcement Learning workload engine for NeuroPilots.

This module defines the workload-specific behavior for Reinforcement
Learning experiments using Proximal Policy Optimization (PPO).

Current responsibility of this layer:
    - Identify the Reinforcement Learning workload.
    - Expose PPO as the default model/algorithm.
    - Validate Reinforcement Learning experiment artifacts.
    - Describe RL-specific experiment requirements.

Actual environment interaction, policy training, rollout generation,
reward computation, evaluation, and experiment execution are handled
by later layers.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.constants import WORKLOAD_REINFORCEMENT_LEARNING
from utils.logger import get_logger
from utils.models import ExperimentArtifacts

from workloads.base import BaseWorkloadEngine


logger = get_logger(__name__)


class ReinforcementLearningWorkload(BaseWorkloadEngine):
    """
    Workload engine for Reinforcement Learning experiments.

    NeuroPilots uses PPO (Proximal Policy Optimization) as the default
    Reinforcement Learning algorithm according to the system architecture.

    This class provides workload metadata and artifact validation only.
    It does not create environments, train policies, execute rollouts,
    or perform RL evaluation.
    """

    workload_name = WORKLOAD_REINFORCEMENT_LEARNING

    def get_model_spec(self) -> Dict[str, Any]:
        """
        Return the default Reinforcement Learning model specification.

        Returns:
            Dictionary describing the RL algorithm and expected task types.
        """

        return {
            "workload": self.workload_name,
            "model_name": self.default_model_name,
            "algorithm": "PPO",
            "algorithm_family": "policy_gradient",
            "task_types": [
                "reinforcement_learning",
                "policy_optimization",
                "continuous_control",
                "discrete_control",
            ],
            "input_type": "environment_observation",
            "output_type": "action",
            "framework_agnostic": True,
        }

    def _validate_workload_specific_artifacts(
        self,
        artifacts: ExperimentArtifacts,
    ) -> None:
        """
        Validate artifacts required for Reinforcement Learning.

        RL experiments differ from supervised learning because the
        primary training signal comes from environment interaction
        and rewards rather than a conventional labeled target.

        At this layer we therefore require:
            - Dataset/environment-related artifacts.
            - Model/policy artifacts.

        Detailed environment and policy validation belongs to later
        execution and experiment layers.
        """

        if artifacts.dataset is None:
            raise ValueError(
                "Reinforcement Learning workload requires dataset "
                "or environment artifacts."
            )

        if artifacts.model is None:
            raise ValueError(
                "Reinforcement Learning workload requires model "
                "or policy artifacts."
            )

        logger.debug(
            "Reinforcement Learning artifacts validated successfully."
        )

    def get_experiment_requirements(self) -> Dict[str, Any]:
        """
        Return RL-specific experiment requirements.

        The requirements describe the information NeuroPilots should
        inspect when diagnosing and optimizing PPO experiments.
        """

        return {
            "environment": {
                "required": True,
                "important_signals": [
                    "environment_name",
                    "observation_space",
                    "action_space",
                    "episode_length",
                    "episode_termination_conditions",
                    "environment_version",
                ],
            },
            "dataset": {
                "required": True,
                "expected_input_type": "environment_interactions",
                "important_signals": [
                    "episode_count",
                    "transition_count",
                    "observation_distribution",
                    "action_distribution",
                    "reward_distribution",
                    "episode_length_distribution",
                    "missing_values",
                    "data_quality",
                ],
            },
            "model": {
                "required": True,
                "default_model": self.default_model_name,
                "important_signals": [
                    "policy_architecture",
                    "value_function_architecture",
                    "parameter_count",
                    "configuration",
                    "observation_encoder",
                ],
            },
            "training_logs": {
                "required": True,
                "important_signals": [
                    "episode_reward",
                    "mean_reward",
                    "episode_length",
                    "policy_loss",
                    "value_loss",
                    "entropy_loss",
                    "approx_kl",
                    "clip_fraction",
                    "learning_rate",
                    "explained_variance",
                ],
            },
            "hyperparameters": {
                "important_parameters": [
                    "learning_rate",
                    "batch_size",
                    "n_steps",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "clip_range",
                    "ent_coef",
                    "vf_coef",
                    "max_grad_norm",
                ],
            },
            "policy": {
                "important_parameters": [
                    "policy_type",
                    "network_architecture",
                    "activation_function",
                    "hidden_layers",
                    "observation_normalization",
                    "reward_normalization",
                ],
                "important_signals": [
                    "policy_entropy",
                    "action_distribution",
                    "policy_stability",
                    "policy_loss",
                ],
            },
            "reward": {
                "important_signals": [
                    "mean_reward",
                    "median_reward",
                    "reward_variance",
                    "reward_trend",
                    "episode_return",
                    "success_rate",
                ],
            },
            "stability": {
                "important_signals": [
                    "policy_loss",
                    "value_loss",
                    "approx_kl",
                    "clip_fraction",
                    "gradient_norm",
                    "explained_variance",
                    "reward_variance",
                ],
            },
            "evaluation": {
                "important_signals": [
                    "evaluation_reward",
                    "evaluation_success_rate",
                    "evaluation_episode_length",
                    "training_vs_evaluation_reward",
                ],
            },
        }

    def describe(self) -> Dict[str, Any]:
        """
        Return a complete description of the RL workload.
        """

        description = super().describe()

        description.update(
            {
                "domain": "reinforcement_learning",
                "experiment_requirements": self.get_experiment_requirements(),
            }
        )

        return description


__all__ = ["ReinforcementLearningWorkload"]