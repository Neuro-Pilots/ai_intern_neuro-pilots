"""
LLM-based reasoning layer for NeuroPilots.

This module provides a provider-agnostic interface for sending structured
diagnostic context to an LLM and receiving research-oriented reasoning.

Supported providers:
    - Groq
    - Google Gemini
    - Hugging Face

The module does not make API calls during import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from groq import Groq
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient

from config.settings import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    HF_TOKEN,
)
from utils.models import Diagnosis, ExperimentContext


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "groq"

# This model is available to the configured Groq API key.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

SUPPORTED_PROVIDERS = (
    "groq",
    "gemini",
    "huggingface",
)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class LLMReasoningResult:
    """
    Result returned by the LLM reasoning layer.

    Attributes:
        provider: LLM provider used.
        model: Model used for the request.
        reasoning: Generated reasoning text.
        prompt: Prompt sent to the model.
        metadata: Additional provider/model information.
        success: Whether the request completed successfully.
        error: Error message when the request fails.
    """

    provider: str
    model: str
    reasoning: str = ""
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM Reasoner
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    Provider-agnostic LLM reasoning engine.

    The reasoner receives structured diagnostic information and asks the
    selected LLM to interpret the evidence and propose the highest-value
    next experiment.

    No API request is made during object construction.
    """

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        fallback_enabled: bool = False,
    ) -> None:
        """
        Initialize the LLM reasoner.

        Args:
            provider:
                LLM provider to use. Supported values:
                "groq", "gemini", "huggingface".

            model:
                Optional model override. If omitted, the provider default
                model is used.

            temperature:
                Sampling temperature.

            max_tokens:
                Maximum number of generated tokens.

            fallback_enabled:
                Whether provider fallback logic should be enabled.
                Fallback is intentionally disabled by default so that
                provider failures remain visible during development.
        """

        provider = provider.strip().lower()

        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. "
                f"Supported providers: {SUPPORTED_PROVIDERS}"
            )

        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        self.provider = provider
        self.model = model or self._default_model_for_provider(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_enabled = fallback_enabled

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reason(
        self,
        diagnosis: Diagnosis,
        context: Optional[ExperimentContext] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> LLMReasoningResult:
        """
        Generate LLM-based reasoning from a structured diagnosis.

        Args:
            diagnosis:
                Output from the deterministic RootCauseEngine.

            context:
                Optional experiment context.

            additional_context:
                Optional additional information such as deployment
                constraints, experiment history, business requirements,
                or user-provided instructions.

        Returns:
            LLMReasoningResult
        """

        prompt = self.build_prompt(
            diagnosis=diagnosis,
            context=context,
            additional_context=additional_context,
        )

        if not self.is_provider_configured(self.provider):
            return LLMReasoningResult(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                success=False,
                error=(
                    f"Provider '{self.provider}' is not configured. "
                    "Check the corresponding API key in the environment."
                ),
            )

        try:
            if self.provider == "groq":
                reasoning, metadata = self._reason_with_groq(prompt)

            elif self.provider == "gemini":
                reasoning, metadata = self._reason_with_gemini(prompt)

            elif self.provider == "huggingface":
                reasoning, metadata = self._reason_with_huggingface(prompt)

            else:
                raise ValueError(
                    f"Unsupported provider: {self.provider}"
                )

            return LLMReasoningResult(
                provider=self.provider,
                model=self.model,
                reasoning=reasoning,
                prompt=prompt,
                metadata=metadata,
                success=True,
            )

        except Exception as exc:
            return LLMReasoningResult(
                provider=self.provider,
                model=self.model,
                prompt=prompt,
                success=False,
                error=str(exc),
            )

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    def build_prompt(
        self,
        diagnosis: Diagnosis,
        context: Optional[ExperimentContext] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the structured research-advisor prompt.

        The prompt intentionally gives the LLM the evidence produced by the
        deterministic diagnostic layer instead of asking the LLM to invent
        diagnostics from scratch.
        """

        diagnosis_data = diagnosis.model_dump(mode="json")

        context_data: Dict[str, Any] = {}

        if context is not None:
            context_data = context.model_dump(
                mode="json",
                exclude_none=True,
            )

        additional_data = additional_context or {}

        return f"""
You are NeuroPilots, an AI Research Intern assisting an ML engineer.

Your job is to reason over an ML/DL experiment and recommend the
single highest-value next experiment.

IMPORTANT RULES:
1. Use the supplied evidence as the primary source of truth.
2. Do not invent experiment results.
3. Do not claim that an experiment was executed unless execution evidence
   is explicitly provided.
4. Clearly distinguish observed evidence from hypotheses.
5. If the evidence is insufficient, explicitly say so.
6. Prefer one concrete next experiment over a long list of generic ideas.
7. The recommendation should be measurable and experimentally testable.
8. Consider the workload and available experiment context.
9. Explain why the proposed experiment addresses the diagnosed issue.
10. Include the metric that should be used to evaluate the experiment.

CURRENT DIAGNOSIS:
{diagnosis_data}

EXPERIMENT CONTEXT:
{context_data}

ADDITIONAL CONTEXT:
{additional_data}

Provide your response using the following structure:

1. Interpretation
   - What appears to be happening?

2. Evidence
   - Which observations support that interpretation?

3. Uncertainty
   - What is not yet known or could have alternative explanations?

4. Next-Best Experiment
   - Give exactly one concrete experiment.
   - Specify what should be changed.

5. Why This Experiment
   - Explain why this is the highest-value next step.

6. Evaluation
   - State the primary metric to compare.
   - Explain what improvement would indicate success.

Keep the recommendation technically specific and actionable.
""".strip()

    # -----------------------------------------------------------------------
    # Provider helpers
    # -----------------------------------------------------------------------

    def _default_model_for_provider(self, provider: str) -> str:
        """Return the default model for a provider."""

        if provider == "groq":
            return DEFAULT_GROQ_MODEL

        if provider == "gemini":
            return DEFAULT_GEMINI_MODEL

        if provider == "huggingface":
            return DEFAULT_HF_MODEL

        raise ValueError(f"Unsupported provider: {provider}")

    def is_provider_configured(self, provider: Optional[str] = None) -> bool:
        """
        Check whether the required API key is configured.
        """

        provider = (provider or self.provider).strip().lower()

        if provider == "groq":
            return bool(GROQ_API_KEY)

        if provider == "gemini":
            return bool(GEMINI_API_KEY)

        if provider == "huggingface":
            return bool(HF_TOKEN)

        return False

    def get_provider_status(self) -> Dict[str, bool]:
        """
        Return configuration status for all supported providers.

        Only boolean availability is returned; secrets are never exposed.
        """

        return {
            provider: self.is_provider_configured(provider)
            for provider in SUPPORTED_PROVIDERS
        }

    # -----------------------------------------------------------------------
    # Groq
    # -----------------------------------------------------------------------

    def _reason_with_groq(
        self,
        prompt: str,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate reasoning using Groq.
        """

        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous ML research advisor. "
                        "Reason from evidence and do not invent results."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        reasoning = response.choices[0].message.content or ""

        metadata: Dict[str, Any] = {
            "provider": "groq",
            "model": self.model,
        }

        usage = getattr(response, "usage", None)

        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": getattr(
                    usage,
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    usage,
                    "completion_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    usage,
                    "total_tokens",
                    None,
                ),
            }

        return reasoning.strip(), metadata

    # -----------------------------------------------------------------------
    # Gemini
    # -----------------------------------------------------------------------

    def _reason_with_gemini(
        self,
        prompt: str,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate reasoning using Google Gemini.
        """

        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        reasoning = getattr(response, "text", "") or ""

        metadata: Dict[str, Any] = {
            "provider": "gemini",
            "model": self.model,
        }

        return reasoning.strip(), metadata

    # -----------------------------------------------------------------------
    # Hugging Face
    # -----------------------------------------------------------------------

    def _reason_with_huggingface(
        self,
        prompt: str,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate reasoning using Hugging Face Inference API.
        """

        client = InferenceClient(
            api_key=HF_TOKEN,
            provider="auto",
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous ML research advisor. "
                        "Reason from evidence and do not invent results."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        reasoning = response.choices[0].message.content or ""

        metadata: Dict[str, Any] = {
            "provider": "huggingface",
            "model": self.model,
        }

        return reasoning.strip(), metadata

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation useful during debugging."""

        return (
            f"LLMReasoner("
            f"provider='{self.provider}', "
            f"model='{self.model}', "
            f"fallback_enabled={self.fallback_enabled}"
            f")"
        )