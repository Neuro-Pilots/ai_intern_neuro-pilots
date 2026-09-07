"""
Root-cause analysis engine for NeuroPilots.

This module consumes the observations produced by PerformanceAnalyzer
and evaluates them against the supported NeuroPilots failure modes.

Responsibilities:
    - Evaluate evidence for known failure modes.
    - Calculate transparent confidence scores.
    - Rank likely root causes.
    - Produce structured RootCause objects.
    - Produce a structured Diagnosis.

Important design principle:
    This engine does not use an LLM. It provides a deterministic,
    explainable baseline diagnostic layer. The LLM Research Advisor
    will be implemented later and can use this diagnosis as structured
    evidence.

The supported failure modes come from the NeuroPilots architecture:
    - overfitting
    - underfitting
    - data_leakage
    - class_imbalance
    - domain_shift
    - temporal_leakage
    - noise
    - hallucination
    - poor_retrieval
    - training_instability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from diagnostics.failure_modes import (
    get_affected_workloads,
    get_diagnostic_questions,
    get_evidence_signals,
    get_failure_mode_definition,
    get_failure_modes_for_workload,
)
from utils.constants import (
    FAILURE_MODE_CLASS_IMBALANCE,
    FAILURE_MODE_DATA_LEAKAGE,
    FAILURE_MODE_DOMAIN_SHIFT,
    FAILURE_MODE_HALLUCINATION,
    FAILURE_MODE_NOISE,
    FAILURE_MODE_OVERFITTING,
    FAILURE_MODE_POOR_RETRIEVAL,
    FAILURE_MODE_TEMPORAL_LEAKAGE,
    FAILURE_MODE_TRAINING_INSTABILITY,
    FAILURE_MODE_UNDERFITTING,
    SUPPORTED_FAILURE_MODES,
)
from utils.logger import get_logger
from utils.models import Diagnosis, PerformanceResult, RootCause


logger = get_logger(__name__)


@dataclass(frozen=True)
class DiagnosticEvidence:
    """
    Internal representation of one diagnostic observation.

    Attributes:
        signal:
            Name of the observed signal.

        value:
            Observed value.

        strength:
            Evidence strength in the range [0, 1].

        explanation:
            Human-readable explanation of the observation.
    """

    signal: str
    value: Any
    strength: float
    explanation: str


class RootCauseEngine:
    """
    Deterministic and explainable root-cause analysis engine.

    The engine intentionally uses transparent rules instead of opaque
    model predictions. This makes the first diagnostic layer easier to
    validate and provides structured evidence for the later LLM advisor.
    """

    def __init__(
        self,
        minimum_confidence: float = 0.30,
        maximum_root_causes: int = 3,
    ) -> None:
        """
        Initialize the Root-Cause Engine.

        Args:
            minimum_confidence:
                Minimum confidence required for a root cause to be included.

            maximum_root_causes:
                Maximum number of ranked root causes returned.
        """

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0.0 and 1.0."
            )

        if maximum_root_causes < 1:
            raise ValueError(
                "maximum_root_causes must be at least 1."
            )

        self.minimum_confidence = minimum_confidence
        self.maximum_root_causes = maximum_root_causes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        performance: PerformanceResult,
        *,
        workload: Optional[str] = None,
        additional_signals: Optional[Dict[str, Any]] = None,
    ) -> Diagnosis:
        """
        Analyze performance evidence and return a structured diagnosis.

        Args:
            performance:
                PerformanceResult produced by PerformanceAnalyzer.

            workload:
                Optional workload identifier. When provided, only failure
                modes relevant to that workload are evaluated.

            additional_signals:
                Optional externally supplied signals, such as:
                    - class distributions
                    - retrieval metrics
                    - hallucination metrics
                    - temporal leakage checks
                    - deployment shift metrics

        Returns:
            Diagnosis containing the primary root cause, additional causes,
            and the original performance result.

        Raises:
            ValueError:
                If performance is not a PerformanceResult instance.
        """

        if not isinstance(performance, PerformanceResult):
            raise ValueError(
                "performance must be a PerformanceResult instance."
            )

        signals = self._merge_signals(
            performance=performance,
            additional_signals=additional_signals,
        )

        failure_modes = self._get_candidate_failure_modes(workload)

        candidates: List[RootCause] = []

        for failure_mode in failure_modes:
            root_cause = self._evaluate_failure_mode(
                failure_mode=failure_mode,
                performance=performance,
                signals=signals,
            )

            if root_cause is None:
                continue

            if root_cause.confidence < self.minimum_confidence:
                continue

            candidates.append(root_cause)

        candidates.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        candidates = candidates[: self.maximum_root_causes]

        primary_root_cause: Optional[RootCause] = None
        additional_root_causes: List[RootCause] = []

        if candidates:
            primary_root_cause = candidates[0]
            additional_root_causes = candidates[1:]

        diagnosis = Diagnosis(
            primary_root_cause=primary_root_cause,
            additional_root_causes=additional_root_causes,
            performance=performance,
        )

        logger.info(
            "Root-cause analysis completed: primary=%s, candidates=%d.",
            (
                primary_root_cause.name
                if primary_root_cause is not None
                else "none"
            ),
            len(candidates),
        )

        return diagnosis

    def get_candidate_failure_modes(
        self,
        workload: Optional[str] = None,
    ) -> Tuple[str, ...]:
        """
        Return failure modes that will be considered for a workload.
        """

        return self._get_candidate_failure_modes(workload)

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_candidate_failure_modes(
        workload: Optional[str],
    ) -> Tuple[str, ...]:
        """
        Determine which failure modes should be evaluated.
        """

        if workload is None:
            return tuple(SUPPORTED_FAILURE_MODES)

        return get_failure_modes_for_workload(workload)

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _merge_signals(
        self,
        performance: PerformanceResult,
        additional_signals: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Combine metrics, comparisons, and externally supplied signals.

        Explicit additional signals take precedence over derived values.
        """

        signals: Dict[str, Any] = {}

        signals.update(performance.metrics)

        self._flatten_nested_mapping(
            performance.comparisons,
            signals,
        )

        self._flatten_nested_mapping(
            performance.resource_usage,
            signals,
        )

        if additional_signals:
            self._flatten_nested_mapping(
                additional_signals,
                signals,
            )

            signals.update(additional_signals)

        return signals

    def _flatten_nested_mapping(
        self,
        mapping: Dict[str, Any],
        output: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """
        Flatten nested dictionaries into dot-separated signal names.
        """

        for key, value in mapping.items():
            key_string = str(key)

            full_key = (
                f"{prefix}.{key_string}"
                if prefix
                else key_string
            )

            if isinstance(value, dict):
                self._flatten_nested_mapping(
                    value,
                    output,
                    full_key,
                )
            else:
                output[full_key] = value

    # ------------------------------------------------------------------
    # Failure-mode evaluation
    # ------------------------------------------------------------------

    def _evaluate_failure_mode(
        self,
        failure_mode: str,
        performance: PerformanceResult,
        signals: Dict[str, Any],
    ) -> Optional[RootCause]:
        """
        Evaluate one failure mode using transparent diagnostic rules.
        """

        definition = get_failure_mode_definition(failure_mode)

        evidence: List[DiagnosticEvidence] = []

        if failure_mode == FAILURE_MODE_OVERFITTING:
            evidence = self._detect_overfitting(signals)

        elif failure_mode == FAILURE_MODE_UNDERFITTING:
            evidence = self._detect_underfitting(signals)

        elif failure_mode == FAILURE_MODE_DATA_LEAKAGE:
            evidence = self._detect_data_leakage(signals)

        elif failure_mode == FAILURE_MODE_CLASS_IMBALANCE:
            evidence = self._detect_class_imbalance(signals)

        elif failure_mode == FAILURE_MODE_DOMAIN_SHIFT:
            evidence = self._detect_domain_shift(signals)

        elif failure_mode == FAILURE_MODE_TEMPORAL_LEAKAGE:
            evidence = self._detect_temporal_leakage(signals)

        elif failure_mode == FAILURE_MODE_NOISE:
            evidence = self._detect_noise(signals)

        elif failure_mode == FAILURE_MODE_HALLUCINATION:
            evidence = self._detect_hallucination(signals)

        elif failure_mode == FAILURE_MODE_POOR_RETRIEVAL:
            evidence = self._detect_poor_retrieval(signals)

        elif failure_mode == FAILURE_MODE_TRAINING_INSTABILITY:
            evidence = self._detect_training_instability(signals)

        if not evidence:
            return None

        confidence = self._calculate_confidence(evidence)

        evidence_text = [
            item.explanation
            for item in evidence
        ]

        explanation = self._build_explanation(
            definition=definition,
            evidence=evidence,
            confidence=confidence,
        )

        return RootCause(
            name=failure_mode,
            confidence=confidence,
            evidence=evidence_text,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Failure-mode detectors
    # ------------------------------------------------------------------

    def _detect_overfitting(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect evidence consistent with overfitting.

        Strong evidence:
            - training loss decreases while validation loss increases
            - training metric improves while validation metric worsens
            - large train/validation metric gap
        """

        evidence: List[DiagnosticEvidence] = []

        train_loss_trend = self._get_numeric(
            signals,
            "train_loss_trend",
        )
        validation_loss_trend = self._get_numeric(
            signals,
            "validation_loss_trend",
        )

        if (
            train_loss_trend is not None
            and validation_loss_trend is not None
            and train_loss_trend < 0
            and validation_loss_trend > 0
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="train_validation_loss_divergence",
                    value={
                        "train_loss_trend": train_loss_trend,
                        "validation_loss_trend": validation_loss_trend,
                    },
                    strength=1.0,
                    explanation=(
                        "Training loss decreased while validation loss increased."
                    ),
                )
            )

        train_accuracy_trend = self._get_numeric(
            signals,
            "train_accuracy_trend",
        )
        validation_accuracy_trend = self._get_numeric(
            signals,
            "validation_accuracy_trend",
        )

        if (
            train_accuracy_trend is not None
            and validation_accuracy_trend is not None
            and train_accuracy_trend > 0
            and validation_accuracy_trend < train_accuracy_trend
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="train_validation_accuracy_gap",
                    value={
                        "train_accuracy_trend": train_accuracy_trend,
                        "validation_accuracy_trend": validation_accuracy_trend,
                    },
                    strength=0.85,
                    explanation=(
                        "Training accuracy improved substantially more than "
                        "validation accuracy."
                    ),
                )
            )

        loss_gap = self._get_nested_numeric(
            signals,
            "train_validation.loss.relative_gap",
        )

        if loss_gap is not None and loss_gap >= 0.50:
            evidence.append(
                DiagnosticEvidence(
                    signal="train_validation_loss_gap",
                    value=loss_gap,
                    strength=min(1.0, loss_gap / 2.0),
                    explanation=(
                        f"The final train/validation loss relative gap is "
                        f"{loss_gap:.3f}."
                    ),
                )
            )

        accuracy_gap = self._get_nested_numeric(
            signals,
            "train_validation.accuracy.relative_gap",
        )

        if accuracy_gap is not None and accuracy_gap <= -0.10:
            evidence.append(
                DiagnosticEvidence(
                    signal="train_validation_accuracy_gap",
                    value=accuracy_gap,
                    strength=min(1.0, abs(accuracy_gap) / 0.50),
                    explanation=(
                        f"The final validation accuracy is below training "
                        f"accuracy by a relative difference of "
                        f"{abs(accuracy_gap):.3f}."
                    ),
                )
            )

        return evidence

    def _detect_underfitting(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect evidence consistent with underfitting.
        """

        evidence: List[DiagnosticEvidence] = []

        train_loss_final = self._get_numeric(
            signals,
            "train_loss_final",
        )

        train_loss_trend = self._get_numeric(
            signals,
            "train_loss_trend",
        )

        if (
            train_loss_final is not None
            and train_loss_trend is not None
            and train_loss_final > 0
            and abs(train_loss_trend) < 0.10
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="training_loss_plateau",
                    value={
                        "final": train_loss_final,
                        "trend": train_loss_trend,
                    },
                    strength=0.65,
                    explanation=(
                        "Training loss remained relatively stable, "
                        "suggesting limited learning progress."
                    ),
                )
            )

        train_accuracy_final = self._get_numeric(
            signals,
            "train_accuracy_final",
        )

        if train_accuracy_final is not None and train_accuracy_final < 0.60:
            evidence.append(
                DiagnosticEvidence(
                    signal="poor_training_accuracy",
                    value=train_accuracy_final,
                    strength=0.75,
                    explanation=(
                        f"Final training accuracy is only "
                        f"{train_accuracy_final:.3f}."
                    ),
                )
            )

        validation_accuracy_final = self._get_numeric(
            signals,
            "validation_accuracy_final",
        )

        if (
            validation_accuracy_final is not None
            and validation_accuracy_final < 0.60
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="poor_validation_accuracy",
                    value=validation_accuracy_final,
                    strength=0.60,
                    explanation=(
                        f"Final validation accuracy is only "
                        f"{validation_accuracy_final:.3f}."
                    ),
                )
            )

        return evidence

    def _detect_data_leakage(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect explicit or strongly suspicious leakage indicators.
        """

        evidence: List[DiagnosticEvidence] = []

        boolean_signals = (
            "train_test_overlap",
            "duplicate_samples_across_splits",
            "target_information_in_features",
            "preprocessing_before_split",
        )

        for signal_name in boolean_signals:
            value = self._get_boolean(signals, signal_name)

            if value is True:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=1.0,
                        explanation=(
                            f"Leakage indicator '{signal_name}' was reported as true."
                        ),
                    )
                )

        suspicious_metric = self._get_numeric(
            signals,
            "unexpectedly_high_validation_metric",
        )

        if suspicious_metric is not None and suspicious_metric > 0:
            evidence.append(
                DiagnosticEvidence(
                    signal="unexpectedly_high_validation_metric",
                    value=suspicious_metric,
                    strength=0.50,
                    explanation=(
                        "The supplied signals report unexpectedly high "
                        "validation performance."
                    ),
                )
            )

        return evidence

    def _detect_class_imbalance(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect class imbalance using explicit ratio/percentage signals.
        """

        evidence: List[DiagnosticEvidence] = []

        majority_minority_ratio = self._get_numeric(
            signals,
            "majority_minority_ratio",
        )

        if (
            majority_minority_ratio is not None
            and majority_minority_ratio >= 5.0
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="majority_minority_ratio",
                    value=majority_minority_ratio,
                    strength=min(
                        1.0,
                        majority_minority_ratio / 10.0,
                    ),
                    explanation=(
                        f"The majority/minority class ratio is "
                        f"{majority_minority_ratio:.2f}:1."
                    ),
                )
            )

        minority_percentage = self._get_numeric(
            signals,
            "minority_class_percentage",
        )

        if (
            minority_percentage is not None
            and 0.0 <= minority_percentage <= 20.0
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="minority_class_percentage",
                    value=minority_percentage,
                    strength=min(
                        1.0,
                        (20.0 - minority_percentage) / 20.0,
                    ),
                    explanation=(
                        f"The minority class represents only "
                        f"{minority_percentage:.2f}% of the data."
                    ),
                )
            )

        return evidence

    def _detect_domain_shift(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect explicit distribution-shift indicators.
        """

        evidence: List[DiagnosticEvidence] = []

        shift_signals = (
            "feature_distribution_shift",
            "label_distribution_shift",
            "covariate_shift",
            "deployment_data_difference",
        )

        for signal_name in shift_signals:
            value = self._get_boolean_or_numeric(
                signals,
                signal_name,
            )

            if isinstance(value, bool) and value:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=1.0,
                        explanation=(
                            f"Distribution-shift signal '{signal_name}' "
                            "was reported as true."
                        ),
                    )
                )

            elif isinstance(value, (int, float)) and value > 0:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=min(1.0, float(value)),
                        explanation=(
                            f"Distribution-shift signal '{signal_name}' "
                            f"has value {value:.4f}."
                        ),
                    )
                )

        return evidence

    def _detect_temporal_leakage(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect explicit temporal leakage indicators.
        """

        evidence: List[DiagnosticEvidence] = []

        boolean_signals = (
            "future_information_in_features",
            "random_split_on_temporal_data",
            "timestamp_order_violation",
            "future_target_information",
        )

        for signal_name in boolean_signals:
            value = self._get_boolean(signals, signal_name)

            if value is True:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=1.0,
                        explanation=(
                            f"Temporal leakage indicator "
                            f"'{signal_name}' was reported as true."
                        ),
                    )
                )

        return evidence

    def _detect_noise(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect explicit dataset/training noise indicators.
        """

        evidence: List[DiagnosticEvidence] = []

        numeric_signals = (
            "missing_value_percentage",
            "outlier_percentage",
            "label_inconsistency_rate",
            "reward_variance",
        )

        for signal_name in numeric_signals:
            value = self._get_numeric(signals, signal_name)

            if value is None or value <= 0:
                continue

            evidence.append(
                DiagnosticEvidence(
                    signal=signal_name,
                    value=value,
                    strength=min(1.0, abs(value)),
                    explanation=(
                        f"Noise-related signal '{signal_name}' "
                        f"has value {value:.4f}."
                    ),
                )
            )

        boolean_signals = (
            "corrupted_inputs",
            "duplicate_samples",
        )

        for signal_name in boolean_signals:
            value = self._get_boolean(signals, signal_name)

            if value is True:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=0.70,
                        explanation=(
                            f"Noise-related indicator '{signal_name}' "
                            "was reported as true."
                        ),
                    )
                )

        return evidence

    def _detect_hallucination(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect Generative AI hallucination indicators.
        """

        evidence: List[DiagnosticEvidence] = []

        hallucination_rate = self._get_numeric(
            signals,
            "hallucination_rate",
        )

        if (
            hallucination_rate is not None
            and hallucination_rate > 0
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="hallucination_rate",
                    value=hallucination_rate,
                    strength=min(1.0, hallucination_rate),
                    explanation=(
                        f"Hallucination rate is "
                        f"{hallucination_rate:.3f}."
                    ),
                )
            )

        groundedness = self._get_numeric(
            signals,
            "groundedness",
        )

        if (
            groundedness is not None
            and 0.0 <= groundedness < 0.70
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="groundedness",
                    value=groundedness,
                    strength=min(
                        1.0,
                        (0.70 - groundedness) / 0.70,
                    ),
                    explanation=(
                        f"Groundedness is relatively low at "
                        f"{groundedness:.3f}."
                    ),
                )
            )

        faithfulness = self._get_numeric(
            signals,
            "faithfulness",
        )

        if (
            faithfulness is not None
            and 0.0 <= faithfulness < 0.70
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="faithfulness",
                    value=faithfulness,
                    strength=min(
                        1.0,
                        (0.70 - faithfulness) / 0.70,
                    ),
                    explanation=(
                        f"Faithfulness is relatively low at "
                        f"{faithfulness:.3f}."
                    ),
                )
            )

        return evidence

    def _detect_poor_retrieval(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect poor retrieval indicators for RAG systems.
        """

        evidence: List[DiagnosticEvidence] = []

        retrieval_metrics = (
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_relevance",
        )

        for signal_name in retrieval_metrics:
            value = self._get_numeric(signals, signal_name)

            if value is None:
                continue

            if 0.0 <= value < 0.70:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=min(
                            1.0,
                            (0.70 - value) / 0.70,
                        ),
                        explanation=(
                            f"{signal_name} is relatively low at "
                            f"{value:.3f}."
                        ),
                    )
                )

        retrieved_document_count = self._get_numeric(
            signals,
            "retrieved_document_count",
        )

        if (
            retrieved_document_count is not None
            and retrieved_document_count <= 0
        ):
            evidence.append(
                DiagnosticEvidence(
                    signal="retrieved_document_count",
                    value=retrieved_document_count,
                    strength=1.0,
                    explanation=(
                        "No retrieved documents were reported."
                    ),
                )
            )

        return evidence

    def _detect_training_instability(
        self,
        signals: Dict[str, Any],
    ) -> List[DiagnosticEvidence]:
        """
        Detect unstable optimization behavior.
        """

        evidence: List[DiagnosticEvidence] = []

        trend_signals = (
            "train_loss_trend",
            "validation_loss_trend",
            "policy_loss_trend",
            "value_loss_trend",
        )

        for signal_name in trend_signals:
            value = self._get_numeric(signals, signal_name)

            if value is None:
                continue

            if abs(value) >= 1.0:
                evidence.append(
                    DiagnosticEvidence(
                        signal=signal_name,
                        value=value,
                        strength=min(1.0, abs(value) / 2.0),
                        explanation=(
                            f"{signal_name} changed substantially "
                            f"with relative change {value:.3f}."
                        ),
                    )
                )

        approx_kl = self._get_numeric(
            signals,
            "approx_kl",
        )

        if approx_kl is not None and approx_kl > 0.05:
            evidence.append(
                DiagnosticEvidence(
                    signal="approx_kl",
                    value=approx_kl,
                    strength=min(1.0, approx_kl / 0.20),
                    explanation=(
                        f"Approximate KL divergence is "
                        f"{approx_kl:.4f}."
                    ),
                )
            )

        reward_variance = self._get_numeric(
            signals,
            "reward_variance",
        )

        if reward_variance is not None and reward_variance > 1.0:
            evidence.append(
                DiagnosticEvidence(
                    signal="reward_variance",
                    value=reward_variance,
                    strength=min(1.0, reward_variance / 10.0),
                    explanation=(
                        f"Reward variance is relatively high at "
                        f"{reward_variance:.4f}."
                    ),
                )
            )

        return evidence

    # ------------------------------------------------------------------
    # Confidence and explanation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        evidence: Sequence[DiagnosticEvidence],
    ) -> float:
        """
        Calculate confidence from multiple independent evidence signals.

        Confidence increases when:
            - evidence is individually strong
            - multiple signals support the same failure mode

        The result is always bounded to [0, 1].
        """

        if not evidence:
            return 0.0

        strengths = [
            max(0.0, min(1.0, item.strength))
            for item in evidence
        ]

        average_strength = sum(strengths) / len(strengths)

        # Multiple independent observations increase confidence without
        # allowing the score to exceed 1.0.
        coverage_bonus = min(
            0.25,
            0.10 * (len(strengths) - 1),
        )

        confidence = average_strength + coverage_bonus

        return round(
            min(1.0, confidence),
            4,
        )

    @staticmethod
    def _build_explanation(
        definition: Dict[str, Any],
        evidence: Sequence[DiagnosticEvidence],
        confidence: float,
    ) -> str:
        """
        Build a concise explanation from the failure definition and evidence.
        """

        evidence_summary = " ".join(
            item.explanation
            for item in evidence
        )

        return (
            f"{definition['description']} "
            f"Diagnostic confidence is {confidence:.2f}. "
            f"Observed evidence: {evidence_summary}"
        )

    # ------------------------------------------------------------------
    # Signal lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_numeric(
        signals: Dict[str, Any],
        name: str,
    ) -> Optional[float]:
        """
        Return a numeric signal using exact or suffix matching.
        """

        value = RootCauseEngine._find_signal(
            signals,
            name,
        )

        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if numeric_value != numeric_value:
            return None

        if numeric_value in (
            float("inf"),
            float("-inf"),
        ):
            return None

        return numeric_value

    @staticmethod
    def _get_boolean(
        signals: Dict[str, Any],
        name: str,
    ) -> Optional[bool]:
        """
        Return a boolean signal when available.
        """

        value = RootCauseEngine._find_signal(
            signals,
            name,
        )

        if isinstance(value, bool):
            return value

        return None

    @staticmethod
    def _get_boolean_or_numeric(
        signals: Dict[str, Any],
        name: str,
    ) -> Any:
        """
        Return a boolean or numeric signal.
        """

        value = RootCauseEngine._find_signal(
            signals,
            name,
        )

        if isinstance(value, bool):
            return value

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_nested_numeric(
        signals: Dict[str, Any],
        name: str,
    ) -> Optional[float]:
        """
        Look up a flattened nested numeric signal.
        """

        value = signals.get(name)

        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if numeric_value != numeric_value:
            return None

        return numeric_value

    @staticmethod
    def _find_signal(
        signals: Dict[str, Any],
        name: str,
    ) -> Any:
        """
        Find an exact signal or a flattened signal ending in '.<name>'.
        """

        if name in signals:
            return signals[name]

        suffix = f".{name}"

        for key, value in signals.items():
            if str(key).endswith(suffix):
                return value

        return None

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def describe_failure_mode(
        self,
        failure_mode: str,
    ) -> Dict[str, Any]:
        """
        Return diagnostic metadata for a failure mode.

        This is useful for the later LLM Research Advisor.
        """

        return {
            "definition": get_failure_mode_definition(failure_mode),
            "evidence_signals": get_evidence_signals(failure_mode),
            "affected_workloads": get_affected_workloads(failure_mode),
            "diagnostic_questions": get_diagnostic_questions(failure_mode),
        }


__all__ = [
    "DiagnosticEvidence",
    "RootCauseEngine",
]