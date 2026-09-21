"""
Recommendation structuring layer for NeuroPilots.

This module converts free-form reasoning produced by the LLM Research
Advisor into the structured ExperimentRecommendation contract used by
the rest of the NeuroPilots pipeline.

IMPORTANT DESIGN PRINCIPLE
--------------------------
This module does NOT select or hardcode experiments.

There is intentionally no logic such as:

    if root_cause == "overfitting":
        recommendation = "add dropout"

The LLM is responsible for deciding which experiment should be performed.

This module is responsible for:

    1. Parsing the LLM response.
    2. Extracting the LLM-generated experiment.
    3. Extracting the LLM-generated rationale.
    4. Extracting evaluation criteria when available.
    5. Preserving supporting evidence.
    6. Normalizing common LLM/Markdown formatting variations.
    7. Validating the final structure with Pydantic.

The recommendation remains dynamic and depends on the actual experiment
context and LLM output.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from utils.logger import get_logger
from utils.models import (
    Diagnosis,
    ExperimentContext,
    ExperimentRecommendation,
)


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structural section aliases
# ---------------------------------------------------------------------------

# These aliases describe where the LLM places information.
#
# They DO NOT define what the recommendation should be.

SECTION_ALIASES: Dict[str, tuple[str, ...]] = {
    "interpretation": (
        "interpretation",
        "interpretation of the diagnosis",
    ),
    "evidence": (
        "evidence",
        "supporting evidence",
    ),
    "uncertainty": (
        "uncertainty",
        "uncertainties",
        "limitations",
    ),
    "next_best_experiment": (
        "next-best experiment",
        "next best experiment",
        "next experiment",
        "proposed experiment",
        "recommended experiment",
    ),
    "why_experiment": (
        "why this experiment",
        "why this",
        "rationale",
        "reason",
        "reasoning",
    ),
    "evaluation": (
        "evaluation",
        "evaluation criteria",
        "success criteria",
        "metrics",
    ),
}


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    Structure and validate an LLM-generated recommendation.

    The class does not decide the recommendation.

    The LLM output remains the source of truth for:
        - experiment
        - change
        - rationale
        - expected improvement
        - cost
        - priority

    The deterministic diagnosis can contribute supporting evidence, but it
    never determines which experiment is selected.
    """

    def __init__(
        self,
        require_recommendation: bool = True,
    ) -> None:
        """
        Initialize the recommendation engine.

        Args:
            require_recommendation:
                If True, fail when a usable experiment cannot be extracted.
        """

        self.require_recommendation = require_recommendation

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(
        self,
        llm_reasoning: str,
        diagnosis: Optional[Diagnosis] = None,
        context: Optional[ExperimentContext] = None,
    ) -> ExperimentRecommendation:
        """
        Convert LLM-generated reasoning into ExperimentRecommendation.

        Args:
            llm_reasoning:
                Free-form response returned by the LLM.

            diagnosis:
                Optional deterministic diagnosis.

            context:
                Optional experiment context.

        Returns:
            ExperimentRecommendation

        Raises:
            ValueError:
                If the LLM response cannot be converted into a valid
                recommendation.
        """

        if not isinstance(llm_reasoning, str):
            raise ValueError(
                "llm_reasoning must be a string."
            )

        text = llm_reasoning.strip()

        if not text:
            raise ValueError(
                "LLM reasoning is empty."
            )

        LOGGER.info(
            "Structuring LLM recommendation from %d characters.",
            len(text),
        )

        # ---------------------------------------------------------------
        # First attempt: structured JSON.
        # ---------------------------------------------------------------

        recommendation_data = (
            self._extract_json_recommendation(text)
        )

        # ---------------------------------------------------------------
        # Second attempt: Markdown/text sections.
        # ---------------------------------------------------------------

        if recommendation_data is None:
            recommendation_data = (
                self._extract_from_sections(text)
            )

        # ---------------------------------------------------------------
        # Add deterministic evidence only when it already exists.
        # ---------------------------------------------------------------

        recommendation_data = (
            self._enrich_with_context(
                recommendation_data=recommendation_data,
                diagnosis=diagnosis,
                context=context,
            )
        )

        # ---------------------------------------------------------------
        # Validate against the shared data contract.
        # ---------------------------------------------------------------

        recommendation = (
            self._validate_recommendation(
                recommendation_data
            )
        )

        LOGGER.info(
            "Recommendation structured successfully: title=%s",
            recommendation.title,
        )

        return recommendation

    def parse(
        self,
        llm_reasoning: str,
        diagnosis: Optional[Diagnosis] = None,
        context: Optional[ExperimentContext] = None,
    ) -> ExperimentRecommendation:
        """
        Alias for generate().
        """

        return self.generate(
            llm_reasoning=llm_reasoning,
            diagnosis=diagnosis,
            context=context,
        )

    # -----------------------------------------------------------------------
    # JSON extraction
    # -----------------------------------------------------------------------

    def _extract_json_recommendation(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a recommendation from JSON returned by the LLM.

        Supports:
            - raw JSON
            - fenced JSON
            - JSON embedded in surrounding text
        """

        candidates: List[str] = []

        # Fenced JSON.
        fenced_matches = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        candidates.extend(fenced_matches)

        # Entire response is JSON.
        stripped = text.strip()

        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        # JSON embedded in text.
        embedded = (
            self._find_balanced_json_object(text)
        )

        if embedded:
            candidates.append(embedded)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)

            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            normalized = (
                self._normalize_json_fields(parsed)
            )

            if self._has_required_recommendation_fields(
                normalized
            ):
                LOGGER.debug(
                    "Structured JSON recommendation detected."
                )

                return normalized

        return None

    def _find_balanced_json_object(
        self,
        text: str,
    ) -> Optional[str]:
        """
        Find the first balanced JSON object in text.
        """

        start = text.find("{")

        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if escaped:
                escaped = False
                continue

            if char == "\\" and in_string:
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return text[start:index + 1]

        return None

    def _normalize_json_fields(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize equivalent LLM JSON field names.

        This is structural normalization only.
        """

        aliases = {
            "name": "title",
            "experiment": "title",
            "experiment_title": "title",
            "recommendation": "title",
            "recommended_experiment": "title",
            "description": "change",
            "modification": "change",
            "change_description": "change",
            "rationale": "reason",
            "why": "reason",
            "expected_gain": "expected_improvement",
            "effort": "cost",
            "importance": "priority",
            "supporting_evidence": "evidence",
        }

        normalized: Dict[str, Any] = {}

        for key, value in data.items():
            normalized_key = aliases.get(
                str(key).strip().lower(),
                key,
            )

            normalized[normalized_key] = value

        return normalized

    # -----------------------------------------------------------------------
    # Markdown/text section parsing
    # -----------------------------------------------------------------------

    def _extract_from_sections(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract recommendation information from LLM Markdown/text.

        The parser is deliberately tolerant of:

            **4. Next-Best Experiment**
            ### Next-Best Experiment
            4. Next-Best Experiment
            Next-Best Experiment
            Next Best Experiment

        It also supports tables inside the experiment section.
        """

        sections = self._split_sections(text)

        experiment_text = sections.get(
            "next_best_experiment",
            "",
        ).strip()

        why_text = sections.get(
            "why_experiment",
            "",
        ).strip()

        evaluation_text = sections.get(
            "evaluation",
            "",
        ).strip()

        evidence_text = sections.get(
            "evidence",
            "",
        ).strip()

        interpretation_text = sections.get(
            "interpretation",
            "",
        ).strip()

        uncertainty_text = sections.get(
            "uncertainty",
            "",
        ).strip()

        if not experiment_text:
            raise ValueError(
                "Unable to extract a Next-Best Experiment "
                "from the LLM response."
            )

        title, change = (
            self._extract_experiment_title_and_change(
                experiment_text
            )
        )

        # Prefer explicit "Why This Experiment" content.
        reason = self._clean_markdown_block(
            why_text
        )

        # If the LLM omitted that section, use its interpretation.
        if not reason:
            reason = self._clean_markdown_block(
                interpretation_text
            )

        if not reason:
            raise ValueError(
                "The LLM response contains an experiment but "
                "does not contain sufficient rationale."
            )

        evidence = self._extract_evidence(
            evidence_text=evidence_text,
            interpretation_text=interpretation_text,
            uncertainty_text=uncertainty_text,
        )

        expected_improvement = (
            self._extract_expected_improvement(
                evaluation_text
            )
        )

        cost = self._extract_cost(
            experiment_text
        )

        priority = self._extract_priority(
            experiment_text=experiment_text,
            evaluation_text=evaluation_text,
        )

        result: Dict[str, Any] = {
            "title": title,
            "change": change,
            "reason": reason,
            "evidence": evidence,
        }

        if expected_improvement is not None:
            result["expected_improvement"] = (
                expected_improvement
            )

        if cost is not None:
            result["cost"] = cost

        if priority is not None:
            result["priority"] = priority

        return result

    def _split_sections(
        self,
        text: str,
    ) -> Dict[str, str]:
        """
        Split an LLM response into semantic sections.
        """

        sections: Dict[str, str] = {}

        current_section: Optional[str] = None
        current_lines: List[str] = []

        for line in text.splitlines():
            detected_section = (
                self._detect_section_heading(line)
            )

            if detected_section is not None:
                if current_section is not None:
                    sections[current_section] = (
                        "\n".join(
                            current_lines
                        ).strip()
                    )

                current_section = detected_section
                current_lines = []
                continue

            if current_section is not None:
                current_lines.append(line)

        if current_section is not None:
            sections[current_section] = (
                "\n".join(
                    current_lines
                ).strip()
            )

        return sections

    def _detect_section_heading(
        self,
        line: str,
    ) -> Optional[str]:
        """
        Detect a semantic section heading.
        """

        normalized = self._normalize_heading(
            line
        )

        if not normalized:
            return None

        return self._match_section(
            normalized
        )

    def _normalize_heading(
        self,
        heading: str,
    ) -> str:
        """
        Normalize an LLM-generated heading.

        Handles:
            - Unicode normalization
            - Unicode dashes
            - Markdown emphasis
            - Markdown heading markers
            - section numbers
            - capitalization
            - repeated whitespace
        """

        normalized = unicodedata.normalize(
            "NFKC",
            heading,
        )

        # Normalize Unicode dash variants.
        unicode_dashes = (
            "\u2010",
            "\u2011",
            "\u2012",
            "\u2013",
            "\u2014",
            "\u2015",
            "\u2212",
            "\uFE58",
            "\uFE63",
            "\uFF0D",
        )

        for dash in unicode_dashes:
            normalized = normalized.replace(
                dash,
                "-",
            )

        # Remove Markdown emphasis/backticks.
        normalized = re.sub(
            r"[*_`~]",
            "",
            normalized,
        )

        # Remove Markdown heading markers.
        normalized = re.sub(
            r"^\s*#+\s*",
            "",
            normalized,
        )

        # Remove numbered section prefix.
        normalized = re.sub(
            r"^\s*\d+\s*[\.\)]\s*",
            "",
            normalized,
        )

        # Normalize whitespace.
        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        normalized = normalized.strip(
            " \t:;-–—"
        )

        return normalized.lower()

    def _match_section(
        self,
        heading: str,
    ) -> Optional[str]:
        """
        Match a normalized heading against structural aliases.
        """

        for section_name, aliases in (
            SECTION_ALIASES.items()
        ):
            for alias in aliases:
                normalized_alias = (
                    self._normalize_heading(
                        alias
                    )
                )

                if heading == normalized_alias:
                    return section_name

                if heading.startswith(
                    normalized_alias + ":"
                ):
                    return section_name

                if heading.startswith(
                    normalized_alias + " "
                ):
                    return section_name

        return None

    # -----------------------------------------------------------------------
    # Experiment extraction
    # -----------------------------------------------------------------------

    def _extract_experiment_title_and_change(
        self,
        experiment_text: str,
    ) -> tuple[str, str]:
        """
        Extract the actual experiment from the LLM response.

        Supports both:

            plain text

        and:

            Markdown tables

        without hardcoding any experiment type.
        """

        cleaned = experiment_text.strip()

        if not cleaned:
            raise ValueError(
                "The Next-Best Experiment section is empty."
            )

        # ---------------------------------------------------------------
        # Detect Markdown table.
        # ---------------------------------------------------------------

        table_rows = self._parse_markdown_table(
            cleaned
        )

        if table_rows:
            title, change = (
                self._extract_from_experiment_table(
                    table_rows
                )
            )

            if title and change:
                return title, change

        # ---------------------------------------------------------------
        # Plain text fallback.
        # ---------------------------------------------------------------

        lines = [
            line.strip()
            for line in cleaned.splitlines()
            if line.strip()
        ]

        title = ""

        # Explicit title labels.
        for line in lines:
            plain = self._clean_markdown(
                line
            )

            match = re.match(
                r"^(?:experiment|title|"
                r"recommended experiment)"
                r"\s*:\s*(.+)$",
                plain,
                flags=re.IGNORECASE,
            )

            if match:
                title = match.group(1).strip()
                break

        # Otherwise extract the first meaningful statement.
        if not title:
            title = (
                self._derive_title_from_llm_text(
                    lines
                )
            )

        if not title:
            raise ValueError(
                "Unable to extract a usable experiment "
                "title from the LLM response."
            )

        return title, cleaned

    def _parse_markdown_table(
        self,
        text: str,
    ) -> List[List[str]]:
        """
        Parse a simple Markdown table.

        Returns:
            List of rows, where each row is a list of cell values.

        Returns an empty list when the text does not contain a valid
        Markdown table.
        """

        rows: List[List[str]] = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if "|" not in stripped:
                continue

            # Remove outer pipes.
            stripped = stripped.strip("|")

            cells = [
                self._clean_markdown(
                    cell.strip()
                )
                for cell in stripped.split("|")
            ]

            if len(cells) < 2:
                continue

            rows.append(cells)

        if len(rows) < 2:
            return []

        # A Markdown table should have a separator row.
        has_separator = any(
            self._is_markdown_separator_row(row)
            for row in rows
        )

        if not has_separator:
            return []

        return rows

    def _is_markdown_separator_row(
        self,
        row: List[str],
    ) -> bool:
        """
        Identify Markdown table separator rows.

        Examples:
            |--------|-------------|
            | :----- | :---------- |
        """

        if not row:
            return False

        for cell in row:
            normalized = cell.strip()

            if not normalized:
                return False

            if not re.fullmatch(
                r":?-{2,}:?",
                normalized,
            ):
                return False

        return True

    def _extract_from_experiment_table(
        self,
        rows: List[List[str]],
    ) -> tuple[str, str]:
        """
        Extract the actual experiment from a Markdown table.

        The table header itself is never used as the recommendation.

        Example:

            | Change | Description |
            |--------|-------------|
            | Add L2 weight-decay | Set weight_decay=1e-4 |

        becomes:

            title  = "Add L2 weight-decay"
            change = full LLM-generated experiment description
        """

        header_index: Optional[int] = None
        separator_index: Optional[int] = None

        for index, row in enumerate(rows):
            if self._is_markdown_separator_row(
                row
            ):
                separator_index = index

                if index > 0:
                    header_index = index - 1

                break

        if separator_index is None:
            return "", ""

        data_rows = rows[
            separator_index + 1:
        ]

        if not data_rows:
            return "", ""

        # ---------------------------------------------------------------
        # Identify semantic columns from the header.
        # ---------------------------------------------------------------

        header = (
            rows[header_index]
            if header_index is not None
            else []
        )

        change_column = (
            self._find_column(
                header,
                (
                    "change",
                    "experiment",
                    "modification",
                    "action",
                ),
            )
        )

        description_column = (
            self._find_column(
                header,
                (
                    "description",
                    "details",
                    "implementation",
                    "configuration",
                ),
            )
        )

        first_data_row = data_rows[0]

        # ---------------------------------------------------------------
        # If the LLM gave a standard Change/Description table, extract
        # those columns.
        # ---------------------------------------------------------------

        if (
            change_column is not None
            and change_column < len(first_data_row)
        ):
            title = self._clean_markdown(
                first_data_row[
                    change_column
                ]
            )

            description = ""

            if (
                description_column is not None
                and description_column
                < len(first_data_row)
            ):
                description = self._clean_markdown(
                    first_data_row[
                        description_column
                    ]
                )

            # Preserve all table data rows in the change description.
            full_change = (
                self._format_table_rows(
                    rows
                )
            )

            if not full_change:
                full_change = (
                    f"{title}. {description}".strip()
                    if description
                    else title
                )

            if title and full_change:
                return title, full_change

        # ---------------------------------------------------------------
        # Generic fallback for an unknown table schema.
        #
        # We use the first non-empty data cell as the title because it is
        # content generated by the LLM. No experiment is selected here.
        # ---------------------------------------------------------------

        for row in data_rows:
            meaningful_cells = [
                self._clean_markdown(cell)
                for cell in row
                if self._clean_markdown(cell)
            ]

            if not meaningful_cells:
                continue

            title = meaningful_cells[0]

            full_change = (
                self._format_table_rows(
                    rows
                )
            )

            if title and full_change:
                return title, full_change

        return "", ""

    def _find_column(
        self,
        header: List[str],
        names: tuple[str, ...],
    ) -> Optional[int]:
        """
        Find a semantic column in a Markdown table header.
        """

        normalized_names = {
            self._normalize_heading(name)
            for name in names
        }

        for index, cell in enumerate(header):
            normalized_cell = (
                self._normalize_heading(cell)
            )

            if normalized_cell in normalized_names:
                return index

        return None

    def _format_table_rows(
        self,
        rows: List[List[str]],
    ) -> str:
        """
        Preserve meaningful LLM-generated table content while removing
        Markdown separator rows.

        This prevents table headers and separator syntax from becoming
        recommendation content.
        """

        meaningful_rows: List[str] = []

        for row in rows:
            if self._is_markdown_separator_row(
                row
            ):
                continue

            cleaned_cells = [
                self._clean_markdown(cell)
                for cell in row
            ]

            if not any(cleaned_cells):
                continue

            meaningful_rows.append(
                " | ".join(
                    cell
                    for cell in cleaned_cells
                    if cell
                )
            )

        return "\n".join(
            meaningful_rows
        ).strip()

    def _derive_title_from_llm_text(
        self,
        lines: List[str],
    ) -> str:
        """
        Derive a concise title from LLM-generated text.

        This only shortens text that the LLM already generated.
        """

        for line in lines:
            candidate = self._clean_markdown(
                line
            )

            candidate = re.sub(
                r"^[\-\*\+\d\.\)\s]+",
                "",
                candidate,
            ).strip()

            if not candidate:
                continue

            # Ignore Markdown table separators.
            if re.fullmatch(
                r"[\s\-:|]+",
                candidate,
            ):
                continue

            # Ignore table header names.
            if candidate.lower() in {
                "change",
                "description",
                "experiment",
                "details",
            }:
                continue

            if len(candidate) <= 120:
                return candidate

            sentence = re.split(
                r"(?<=[.!?])\s+",
                candidate,
                maxsplit=1,
            )[0]

            return sentence[:120].strip()

        return ""

    # -----------------------------------------------------------------------
    # Evidence extraction
    # -----------------------------------------------------------------------

    def _extract_evidence(
        self,
        evidence_text: str,
        interpretation_text: str,
        uncertainty_text: str,
    ) -> List[str]:
        """
        Extract evidence generated by the LLM.

        Markdown tables are handled separately so that headers and separator
        rows are not treated as evidence.
        """

        evidence: List[str] = []

        if evidence_text:
            # First check whether the evidence itself is a table.
            table_rows = (
                self._parse_markdown_table(
                    evidence_text
                )
            )

            if table_rows:
                evidence.extend(
                    self._extract_evidence_from_table(
                        table_rows
                    )
                )
            else:
                evidence.extend(
                    self._extract_list_items(
                        evidence_text
                    )
                )

                if not evidence:
                    cleaned = (
                        self._clean_markdown_block(
                            evidence_text
                        )
                    )

                    if cleaned:
                        evidence.append(
                            cleaned
                        )

        # If no explicit evidence section was supplied, preserve the
        # interpretation as supporting evidence.
        if (
            not evidence
            and interpretation_text
        ):
            cleaned = (
                self._clean_markdown_block(
                    interpretation_text
                )
            )

            if cleaned:
                evidence.append(
                    cleaned
                )

        # Uncertainty is deliberately not evidence.
        _ = uncertainty_text

        return self._unique_non_empty(
            evidence
        )

    def _extract_evidence_from_table(
        self,
        rows: List[List[str]],
    ) -> List[str]:
        """
        Extract meaningful rows from an evidence Markdown table.

        Header and separator rows are excluded.
        """

        evidence: List[str] = []

        separator_index: Optional[int] = None

        for index, row in enumerate(rows):
            if self._is_markdown_separator_row(
                row
            ):
                separator_index = index
                break

        if separator_index is None:
            return evidence

        data_rows = rows[
            separator_index + 1:
        ]

        for row in data_rows:
            cells = [
                self._clean_markdown(cell)
                for cell in row
                if self._clean_markdown(cell)
            ]

            if not cells:
                continue

            evidence.append(
                " | ".join(cells)
            )

        return evidence

    def _extract_list_items(
        self,
        text: str,
    ) -> List[str]:
        """
        Extract Markdown bullet/numbered list items.

        Table syntax is deliberately ignored here.
        """

        items: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            # Do not treat table rows as bullet evidence.
            if stripped.startswith("|"):
                continue

            cleaned = re.sub(
                r"^\s*(?:[-*+]|\d+[\.\)])\s*",
                "",
                stripped,
            )

            cleaned = self._clean_markdown(
                cleaned
            )

            if cleaned:
                items.append(cleaned)

        return items

    # -----------------------------------------------------------------------
    # Optional fields
    # -----------------------------------------------------------------------

    def _extract_expected_improvement(
        self,
        evaluation_text: str,
    ) -> Optional[str]:
        """
        Extract explicitly stated expected improvement/success criteria.

        Returns None when the LLM does not provide one.
        """

        if not evaluation_text:
            return None

        cleaned = self._clean_markdown_block(
            evaluation_text
        )

        patterns = (
            r"(?:expected improvement|expected gain|"
            r"success criterion|success criteria)"
            r"\s*:\s*(.+)",

            r"(?:improvement|gain)"
            r"\s*(?:of|:)\s*(.+)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                cleaned,
                flags=re.IGNORECASE,
            )

            if match:
                value = self._clean_markdown(
                    match.group(1)
                )

                if value:
                    return value

        # Also preserve an explicitly stated evaluation target if the
        # LLM did not use one of the exact labels above.
        for line in cleaned.splitlines():
            normalized = line.strip()

            if re.search(
                r"\b(?:increase|decrease|improve|"
                r"reduce|target|achieve)\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                value = self._clean_markdown(
                    normalized
                )

                if value:
                    return value

        return None

    def _extract_cost(
        self,
        experiment_text: str,
    ) -> Optional[str]:
        """
        Extract explicitly stated cost/effort.

        No value is invented.
        """

        patterns = (
            r"(?:cost|effort|complexity)"
            r"\s*:\s*(.+)",

            r"\b(?:low|medium|high)"
            r"\s+(?:cost|effort)\b",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                experiment_text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            if match.lastindex:
                value = self._clean_markdown(
                    match.group(1)
                )
            else:
                value = self._clean_markdown(
                    match.group(0)
                )

            if value:
                return value

        return None

    def _extract_priority(
        self,
        experiment_text: str,
        evaluation_text: str,
    ) -> Optional[str]:
        """
        Extract explicitly stated priority.

        No priority is invented.
        """

        combined = (
            f"{experiment_text}\n"
            f"{evaluation_text}"
        )

        match = re.search(
            r"(?:priority)"
            r"\s*:\s*(high|medium|low)",
            combined,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).lower()

        return None

    # -----------------------------------------------------------------------
    # Context enrichment
    # -----------------------------------------------------------------------

    def _enrich_with_context(
        self,
        recommendation_data: Dict[str, Any],
        diagnosis: Optional[Diagnosis],
        context: Optional[ExperimentContext],
    ) -> Dict[str, Any]:
        """
        Add already-known deterministic evidence.

        The diagnosis is never converted into a recommendation.
        """

        result = dict(
            recommendation_data
        )

        existing_evidence = result.get(
            "evidence",
            [],
        )

        if not isinstance(
            existing_evidence,
            list,
        ):
            existing_evidence = [
                str(existing_evidence)
            ]

        deterministic_evidence: List[str] = []

        if diagnosis is not None:

            if (
                diagnosis.primary_root_cause
                is not None
            ):
                root_cause = (
                    diagnosis.primary_root_cause
                )

                deterministic_evidence.append(
                    (
                        "Deterministic diagnosis: "
                        f"{root_cause.name} "
                        f"(confidence="
                        f"{root_cause.confidence:.3f})"
                    )
                )

                deterministic_evidence.extend(
                    root_cause.evidence
                )

            for root_cause in (
                diagnosis.additional_root_causes
            ):
                deterministic_evidence.append(
                    (
                        "Additional deterministic diagnosis: "
                        f"{root_cause.name} "
                        f"(confidence="
                        f"{root_cause.confidence:.3f})"
                    )
                )

        result["evidence"] = (
            self._unique_non_empty(
                [
                    *existing_evidence,
                    *deterministic_evidence,
                ]
            )
        )

        # Context is retained only as internal metadata.
        # It never influences recommendation selection.
        if context is not None:
            result["_context"] = {
                "domain": context.domain,
                "workload": context.workload,
            }

        return result

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_recommendation(
        self,
        data: Dict[str, Any],
    ) -> ExperimentRecommendation:
        """
        Validate the structured recommendation with Pydantic.
        """

        allowed_fields = {
            "title",
            "change",
            "reason",
            "expected_improvement",
            "cost",
            "priority",
            "evidence",
        }

        model_data = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }

        required_fields = (
            "title",
            "change",
            "reason",
        )

        missing = [
            field
            for field in required_fields
            if not self._has_value(
                model_data.get(field)
            )
        ]

        if missing:
            raise ValueError(
                "LLM recommendation is missing required fields: "
                + ", ".join(missing)
            )

        try:
            return ExperimentRecommendation(
                **model_data
            )

        except ValidationError as exc:
            LOGGER.error(
                "LLM recommendation failed validation: %s",
                exc,
            )

            raise ValueError(
                "The LLM-generated recommendation failed validation: "
                f"{exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _has_required_recommendation_fields(
        data: Dict[str, Any],
    ) -> bool:
        """
        Check for the core recommendation fields.
        """

        return all(
            RecommendationEngine._has_value(
                data.get(field)
            )
            for field in (
                "title",
                "change",
                "reason",
            )
        )

    @staticmethod
    def _has_value(
        value: Any,
    ) -> bool:
        """
        Determine whether a value is usable.
        """

        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, list):
            return bool(value)

        if isinstance(value, dict):
            return bool(value)

        return True

    @staticmethod
    def _clean_markdown(
        text: str,
    ) -> str:
        """
        Remove presentation-only Markdown from a single text value.
        """

        cleaned = unicodedata.normalize(
            "NFKC",
            str(text),
        )

        cleaned = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"__(.*?)__",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"`([^`]*)`",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"^\s*#+\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        return cleaned.strip()

    def _clean_markdown_block(
        self,
        text: str,
    ) -> str:
        """
        Clean a multi-line Markdown block while preserving its substance.
        """

        cleaned_lines: List[str] = []

        for line in text.splitlines():
            cleaned = self._clean_markdown(
                line
            )

            if cleaned:
                cleaned_lines.append(
                    cleaned
                )

        return "\n".join(
            cleaned_lines
        ).strip()

    @staticmethod
    def _unique_non_empty(
        values: List[Any],
    ) -> List[str]:
        """
        Preserve order while removing duplicates and empty values.
        """

        result: List[str] = []
        seen: set[str] = set()

        for value in values:
            if value is None:
                continue

            text = str(value).strip()

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)
            result.append(text)

        return result

    def __repr__(self) -> str:
        """
        Return a concise debugging representation.
        """

        return (
            "RecommendationEngine("
            f"require_recommendation="
            f"{self.require_recommendation}"
            ")"
        )