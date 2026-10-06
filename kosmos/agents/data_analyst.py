"""
Data Analyst Agent.

Claude-powered agent for interpreting experiment results, detecting patterns,
identifying anomalies, and generating scientific insights.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np

from kosmos.agents.base import BaseAgent, AgentStatus
from kosmos.core.llm import get_client
from kosmos.models.result import ExperimentResult
from kosmos.models.hypothesis import Hypothesis

logger = logging.getLogger(__name__)


class ResultInterpretation:
    """Structured interpretation of experiment results."""

    def __init__(
        self,
        experiment_id: str,
        hypothesis_supported: Optional[bool],
        confidence: float,
        summary: str,
        key_findings: List[str],
        significance_interpretation: str,
        biological_significance: Optional[str],
        comparison_to_prior_work: Optional[str],
        potential_confounds: List[str],
        follow_up_experiments: List[str],
        anomalies_detected: List[str],
        patterns_detected: List[str],
        overall_assessment: str,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize result interpretation.

        Args:
            experiment_id: ID of experiment being interpreted
            hypothesis_supported: Whether hypothesis is supported (None if unclear)
            confidence: Confidence in interpretation (0.0-1.0)
            summary: High-level summary of results
            key_findings: List of 3-5 key findings
            significance_interpretation: Interpretation of statistical significance
            biological_significance: Scientific/practical meaning (if applicable)
            comparison_to_prior_work: How results compare to literature
            potential_confounds: List of potential confounding factors
            follow_up_experiments: Suggested follow-up experiments
            anomalies_detected: Any anomalies found in results
            patterns_detected: Patterns identified across results
            overall_assessment: Overall assessment of experiment quality
            created_at: Timestamp of interpretation creation
        """
        self.experiment_id = experiment_id
        self.hypothesis_supported = hypothesis_supported
        self.confidence = confidence
        self.summary = summary
        self.key_findings = key_findings
        self.significance_interpretation = significance_interpretation
        self.biological_significance = biological_significance
        self.comparison_to_prior_work = comparison_to_prior_work
        self.potential_confounds = potential_confounds
        self.follow_up_experiments = follow_up_experiments
        self.anomalies_detected = anomalies_detected
        self.patterns_detected = patterns_detected
        self.overall_assessment = overall_assessment
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "hypothesis_supported": self.hypothesis_supported,
            "confidence": self.confidence,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "significance_interpretation": self.significance_interpretation,
            "biological_significance": self.biological_significance,
            "comparison_to_prior_work": self.comparison_to_prior_work,
            "potential_confounds": self.potential_confounds,
            "follow_up_experiments": self.follow_up_experiments,
            "anomalies_detected": self.anomalies_detected,
            "patterns_detected": self.patterns_detected,
            "overall_assessment": self.overall_assessment,
            "created_at": self.created_at.isoformat()
        }


class DataAnalystAgent(BaseAgent):
    """
    Agent for analyzing and interpreting experiment results using Claude.

    Capabilities:
    - Interpret statistical results in scientific context
    - Detect patterns across multiple experiments
    - Identify anomalies in results
    - Provide significance interpretation beyond p-values
    - Generate actionable insights for next steps

    Example:
        ```python
        agent = DataAnalystAgent(config={
            "use_literature_context": True,
            "detailed_interpretation": True
        })
        agent.start()

        # Interpret results
        interpretation = agent.interpret_results(
            result=experiment_result,
            hypothesis=original_hypothesis,
            literature_context="Recent papers found..."
        )

        print(f"Hypothesis supported: {interpretation.hypothesis_supported}")
        print(f"Key findings: {interpretation.key_findings}")
        ```
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Data Analyst Agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type name
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type or "DataAnalystAgent", config)

        # Configuration
        self.use_literature_context = self.config.get("use_literature_context", True)
        self.detailed_interpretation = self.config.get("detailed_interpretation", True)
        self.anomaly_detection_enabled = self.config.get("anomaly_detection_enabled", True)
        self.pattern_detection_enabled = self.config.get("pattern_detection_enabled", True)
        self.significance_threshold_strict = self.config.get("significance_threshold_strict", 0.01)
        self.significance_threshold_relaxed = self.config.get("significance_threshold_relaxed", 0.05)
        self.effect_size_threshold = self.config.get("effect_size_threshold", 0.3)

        # Components
        self.llm_client = get_client()

        # State: Store interpretations for pattern detection
        self.interpretation_history: List[ResultInterpretation] = []

        logger.info(f"Initialized DataAnalystAgent {self.agent_id}")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent task.

        Args:
            task: Task specification with:
                - action: "interpret_results", "detect_patterns", "detect_anomalies"
                - result: ExperimentResult object
                - hypothesis: Optional Hypothesis object
                - literature_context: Optional literature context string

        Returns:
            dict: Task result with interpretation
        """
        self.status = AgentStatus.WORKING
        action = task.get("action", "interpret_results")

        try:
            if action == "interpret_results":
                result = task["result"]
                hypothesis = task.get("hypothesis")
                literature_context = task.get("literature_context")

                interpretation = self.interpret_results(result, hypothesis, literature_context)

                return {
                    "success": True,
                    "interpretation": interpretation.to_dict()
                }

            elif action == "detect_patterns":
                results = task["results"]
                patterns = self.detect_patterns_across_results(results)

                return {
                    "success": True,
                    "patterns": patterns
                }

            elif action == "detect_anomalies":
                result = task["result"]
                anomalies = self.detect_anomalies(result)

                return {
                    "success": True,
                    "anomalies": anomalies
                }

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error executing task in DataAnalystAgent: {e}")
            self.errors_encountered += 1
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.status = AgentStatus.IDLE
            self.tasks_completed += 1

    # ========================================================================
    # RESULT INTERPRETATION
    # ========================================================================

    def interpret_results(
        self,
        result: ExperimentResult,
        hypothesis: Optional[Hypothesis] = None,
        literature_context: Optional[str] = None
    ) -> ResultInterpretation:
        """
        Interpret experiment results using Claude.

        Args:
            result: ExperimentResult object to interpret
            hypothesis: Optional original hypothesis
            literature_context: Optional context from literature

        Returns:
            ResultInterpretation: Structured interpretation
        """
        logger.info(f"Interpreting results for experiment {result.experiment_id}")

        # Extract key information from result
        result_summary = self._extract_result_summary(result)

        # Build interpretation prompt
        prompt = self._build_interpretation_prompt(
            result_summary, hypothesis, literature_context
        )

        system = ("You are an expert scientific data analyst. Provide nuanced, "
                  "evidence-based interpretations of experimental results. Focus on "
                  "scientific meaning, not just statistical significance.")
        # The interpretation is structured output, so take the HARDENED path:
        # generate_structured uses JSON mode plus the provider's empty /
        # truncated / unparseable retries. The old `generate()` + find('{') +
        # json.loads path had none of those, so a single flaky draw (an empty
        # finish_reason=length reply, or prose without an object) dropped the
        # whole verdict to the automated fallback -- which writes NO
        # supports/rejects decision, leaving the hypothesis stuck at 'generated'
        # and never adjudicated. That was the real cause of "no supported
        # results". 4096 carries the free-text lists (findings, confounds,
        # follow-ups); the provider bumps it if a draw truncates.
        schema = {
            "hypothesis_supported": "boolean (true if the result supports the hypothesis, false if it refutes it) or null if untestable",
            "confidence": "float 0.0-1.0",
            "summary": "string",
            "key_findings": ["string"],
            "significance_interpretation": "string",
            "biological_significance": "string",
            "comparison_to_prior_work": "string",
            "potential_confounds": ["string"],
            "follow_up_experiments": ["string"],
            "overall_assessment": "string",
        }
        try:
            if hasattr(self.llm_client, "generate_structured"):
                data = self.llm_client.generate_structured(
                    prompt=prompt, schema=schema, system=system,
                    max_tokens=4096, temperature=0.3,
                )
                interpretation = self._interpretation_from_data(
                    data, result.experiment_id, result
                )
            else:
                # A client without structured output (e.g. a bare ClaudeClient):
                # keep the original free-text parse.
                response = self.llm_client.generate(
                    prompt=prompt, system=system, max_tokens=4096, temperature=0.3,
                )
                response_text = response.content if hasattr(response, 'content') else str(response)
                interpretation = self._parse_interpretation_response(
                    response_text, result.experiment_id, result
                )

            self.interpretation_history.append(interpretation)
            logger.info(f"Completed interpretation for {result.experiment_id}")
            return interpretation

        except Exception as e:
            logger.error(f"Error getting interpretation: {e}")
            # Return fallback interpretation (now a genuine last resort, not the
            # routine outcome of one empty draw).
            return self._create_fallback_interpretation(result)

    def _extract_result_summary(self, result: ExperimentResult) -> Dict[str, Any]:
        """Extract key information from result for prompt."""
        summary = {
            "experiment_id": result.experiment_id,
            "status": result.status.value,
            "primary_test": result.primary_test,
            "primary_p_value": result.primary_p_value,
            "primary_effect_size": result.primary_effect_size,
            "supports_hypothesis": result.supports_hypothesis,
            "statistical_tests": []
        }

        # Add statistical test details
        for test in result.statistical_tests:
            summary["statistical_tests"].append({
                "test_name": test.test_name,
                "statistic": test.statistic,
                "p_value": test.p_value,
                "effect_size": test.effect_size,
                "effect_size_type": test.effect_size_type,
                "significance_label": test.significance_label,
                "sample_size": test.sample_size
            })

        # Add variable summaries
        if result.variable_results:
            summary["variables"] = []
            for var in result.variable_results[:5]:  # Top 5 variables
                summary["variables"].append({
                    "name": var.variable_name,
                    "mean": var.mean,
                    "median": var.median,
                    "std": var.std,
                    "min": var.min,
                    "max": var.max,
                    "n_samples": var.n_samples
                })

        return summary

    def _build_interpretation_prompt(
        self,
        result_summary: Dict[str, Any],
        hypothesis: Optional[Hypothesis],
        literature_context: Optional[str]
    ) -> str:
        """Build prompt for Claude interpretation."""
        prompt_parts = []

        # Hypothesis context
        if hypothesis:
            prompt_parts.append(f"""
HYPOTHESIS:
{hypothesis.statement}

Domain: {hypothesis.domain}
Expected Outcome: {getattr(hypothesis, 'expected_outcome', 'Not specified')}
""")

        # Result summary
        prompt_parts.append(f"""
EXPERIMENTAL RESULTS:
Status: {result_summary['status']}
Primary Test: {result_summary['primary_test']}
Primary P-value: {result_summary['primary_p_value']}
Primary Effect Size: {result_summary['primary_effect_size']}
Hypothesis Supported: {result_summary['supports_hypothesis']}

Statistical Tests:
""")

        for i, test in enumerate(result_summary['statistical_tests'][:3], 1):
            # `statistic` is a REQUIRED model field, so a test reported as an
            # effect + p-value (e.g. a correlation's r, or an MR beta) that has no
            # separate test statistic gets a placeholder 0.0. Rendering that as
            # "0.0000" reads as a computed zero and gets flagged as a reporting
            # error -- so show it as "not reported" when it is a 0.0 placeholder
            # alongside a real effect size, pointing to the effect instead.
            stat = test['statistic']
            eff = test['effect_size']
            if isinstance(stat, (int, float)) and stat == 0.0 and eff not in (None, ""):
                stat_str = "not reported (see effect size)"
            elif isinstance(stat, (int, float)):
                stat_str = f"{stat:.4f}"
            else:
                stat_str = str(stat)
            prompt_parts.append(f"""
Test {i}: {test['test_name']}
  - Statistic: {stat_str}
  - P-value: {test['p_value']:.6f}
  - Effect Size: {test['effect_size']} ({test['effect_size_type']})
  - Significance: {test['significance_label']}
  - Sample Size: {test['sample_size']}
""")

        # Literature context
        if literature_context and self.use_literature_context:
            prompt_parts.append(f"""
LITERATURE CONTEXT:
{literature_context[:1000]}  # Limit to 1000 chars
""")

        # Instructions
        prompt_parts.append("""
Please provide a comprehensive scientific interpretation of these results:

1. HYPOTHESIS SUPPORT: Does the data support or reject the hypothesis? (Be nuanced - consider strength of evidence)

2. KEY FINDINGS: What are the 3-5 most important findings from these results?

3. STATISTICAL SIGNIFICANCE: Interpret the statistical significance beyond just "p < 0.05". What does it mean scientifically?

4. EFFECT SIZE: Is the effect size scientifically/practically meaningful, even if statistically significant?

5. BIOLOGICAL/PHYSICAL SIGNIFICANCE: What is the real-world meaning of these results? (If applicable)

6. COMPARISON TO PRIOR WORK: How do these results compare to existing literature? (If context provided)

7. POTENTIAL CONFOUNDS: What are 2-3 potential confounding factors or limitations?

8. FOLLOW-UP EXPERIMENTS: What 3-5 follow-up experiments would you recommend based on these results?

9. OVERALL ASSESSMENT: Rate the quality and reliability of this experiment (Low/Medium/High) and explain why.

Format your response as JSON with the following structure:
{
    "hypothesis_supported": true/false/null,
    "confidence": 0.0-1.0,
    "summary": "2-3 sentence overview",
    "key_findings": ["finding 1", "finding 2", ...],
    "significance_interpretation": "detailed statistical interpretation",
    "biological_significance": "real-world meaning or null",
    "comparison_to_prior_work": "comparison or null",
    "potential_confounds": ["confound 1", "confound 2", ...],
    "follow_up_experiments": ["experiment 1", "experiment 2", ...],
    "overall_assessment": "Quality: X/5. Explanation..."
}
""")

        return "\n".join(prompt_parts)

    def _parse_interpretation_response(
        self,
        response: str,
        experiment_id: str,
        result: ExperimentResult
    ) -> ResultInterpretation:
        """Parse Claude's JSON response into ResultInterpretation."""
        try:
            # Extract JSON from response (Claude sometimes adds text before/after)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                # No object at all: an empty reply, or a reasoning trace handed
                # back as text. Say which, rather than "Expecting value at char 0".
                logger.error(
                    "Interpretation response holds no JSON object (%d chars%s); "
                    "using fallback interpretation",
                    len(response), "" if response.strip() else ", empty",
                )
                return self._create_fallback_interpretation(result)
            json_str = response[json_start:json_end]

            data = json.loads(json_str)
            return self._interpretation_from_data(data, experiment_id, result)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Claude response: {e}")
            logger.debug(f"Response was: {response}")
            return self._create_fallback_interpretation(result)

    # Substrings that, as (part of) a result key with a numeric value, mark a
    # primary statistical quantity. GENERAL -- no experiment-specific names.
    _STAT_KEY_TOKENS = (
        "p_value", "pvalue", "pval", "q_value", "qvalue", "fdr",
        "effect", "beta", "coef", "estimate", "odds_ratio", "oddsratio",
        "hazard_ratio", "hazardratio", "correlation", "corr", "rho", "spearman",
        "pearson", "r_squared", "rsquared", "statistic", "t_stat", "z_score",
        "zscore", "ci_lower", "ci_upper", "conf_int", "confidence_interval",
        "slope", "log10p", "chisq", "chi2", "f_stat", "wald",
    )

    @classmethod
    def _dict_has_statistic(cls, obj: Any, _depth: int = 0) -> bool:
        """Does a captured payload contain a stat-named key with a numeric value?

        Bounded recursive scan of the raw/processed result so a genuine result
        whose statistics were emitted in a custom dict (rather than mapped onto the
        typed fields) is NOT mistaken for a test-less run.
        """
        if _depth > 5 or obj is None:
            return False
        if isinstance(obj, dict):
            for k, v in obj.items():
                norm = str(k).lower().replace(" ", "").replace("-", "_")
                if any(tok in norm for tok in cls._STAT_KEY_TOKENS):
                    if isinstance(v, bool):
                        continue  # a boolean flag named e.g. 'significant' is not a value
                    if isinstance(v, (int, float)):
                        return True
                    if isinstance(v, str):
                        try:
                            float(v)
                            return True
                        except ValueError:
                            pass
                if cls._dict_has_statistic(v, _depth + 1):
                    return True
            return False
        if isinstance(obj, (list, tuple)):
            return any(cls._dict_has_statistic(v, _depth + 1) for v in obj)
        return False

    @classmethod
    def _result_has_statistical_evidence(cls, result: ExperimentResult) -> bool:
        """Whether the experiment actually produced a primary statistical quantity.

        A verdict (supported/refuted) is only meaningful when SOME statistic was
        computed -- a p-value, an effect size / coefficient / estimate, a CI, a
        correlation, or a named test. An experiment that merely completed a
        data-matching pipeline and emitted descriptive metadata carries none of
        these, and a true/false verdict over it is unfounded. General: it tests for
        the PRESENCE of any statistical quantity, never for a specific test.
        """
        if getattr(result, "statistical_tests", None):
            return True
        for attr in ("primary_p_value", "primary_effect_size",
                     "primary_ci_lower", "primary_ci_upper"):
            if getattr(result, attr, None) is not None:
                return True
        if getattr(result, "primary_test", None):
            return True
        for vr in (getattr(result, "variable_results", None) or []):
            if getattr(vr, "p_value", None) is not None or \
               getattr(vr, "effect_size", None) is not None:
                return True
        # Stats emitted in a custom shape land in raw_data/processed_data.
        return (cls._dict_has_statistic(getattr(result, "raw_data", None))
                or cls._dict_has_statistic(getattr(result, "processed_data", None)))

    def _interpretation_from_data(
        self,
        data: Dict[str, Any],
        experiment_id: str,
        result: ExperimentResult,
    ) -> ResultInterpretation:
        """Build a ResultInterpretation from an already-parsed interpretation dict.

        Shared by the structured-output path and the legacy free-text parser so
        the field mapping lives in one place.
        """
        anomalies = self.detect_anomalies(result) if self.anomaly_detection_enabled else []
        # Gate an unfounded verdict: if the model returned supported/refuted but the
        # experiment executed NO primary statistical test, discard the verdict so
        # the hypothesis lands at INCONCLUSIVE rather than being adjudicated on a
        # result that never tested it. This is the single chokepoint both the
        # structured path and the legacy parser build through. We do NOT regenerate
        # -- the deficiency is in the DATA (no test ran), not in the output's form,
        # so a re-prompt would only reproduce the same unfounded verdict.
        hypothesis_supported = data.get("hypothesis_supported")
        if hypothesis_supported is not None and \
                not self._result_has_statistical_evidence(result):
            logger.warning(
                "Verdict discarded for %s: result carries no primary statistical "
                "quantity (no p-value/effect/statistic/CI/test); marking "
                "inconclusive instead of %s.",
                experiment_id, hypothesis_supported,
            )
            hypothesis_supported = None
        return ResultInterpretation(
            experiment_id=experiment_id,
            hypothesis_supported=hypothesis_supported,
            confidence=data.get("confidence", 0.5),
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            significance_interpretation=data.get("significance_interpretation", ""),
            biological_significance=data.get("biological_significance"),
            comparison_to_prior_work=data.get("comparison_to_prior_work"),
            potential_confounds=data.get("potential_confounds", []),
            follow_up_experiments=data.get("follow_up_experiments", []),
            anomalies_detected=anomalies,
            patterns_detected=[],
            overall_assessment=data.get("overall_assessment", "")
        )

    def _create_fallback_interpretation(self, result: ExperimentResult) -> ResultInterpretation:
        """Create fallback interpretation if Claude fails."""
        return ResultInterpretation(
            experiment_id=result.experiment_id,
            hypothesis_supported=result.supports_hypothesis,
            confidence=0.5,
            summary=f"Experiment {result.status.value} with p-value {result.primary_p_value}",
            key_findings=[
                f"Primary test: {result.primary_test}",
                f"P-value: {result.primary_p_value}",
                f"Effect size: {result.primary_effect_size}"
            ],
            significance_interpretation=(
                f"P-value of {result.primary_p_value} indicates "
                f"{'significant' if result.primary_p_value < 0.05 else 'non-significant'} results"
                if result.primary_p_value is not None
                else "No p-value available (experiment produced no valid statistical result)"
            ),
            biological_significance=None,
            comparison_to_prior_work=None,
            potential_confounds=["Automated analysis - manual review recommended"],
            follow_up_experiments=["Manual review needed for recommendations"],
            anomalies_detected=[],
            patterns_detected=[],
            overall_assessment="Automated fallback interpretation - LLM interpretation unavailable (the analyst call failed or the result carried nothing to interpret; see log)"
        )

    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================

    def detect_anomalies(self, result: ExperimentResult) -> List[str]:
        """
        Detect anomalies in experimental results.

        Checks for:
        - Unusual p-value distributions
        - Effect size/significance mismatches
        - Outliers in statistical tests
        - Data quality issues

        Args:
            result: ExperimentResult to analyze

        Returns:
            list: List of anomaly descriptions
        """
        anomalies = []

        # Check for significant p-value with tiny effect size
        if result.primary_p_value is not None and result.primary_effect_size is not None:
            if result.primary_p_value < self.significance_threshold_strict:
                if abs(result.primary_effect_size) < self.effect_size_threshold:
                    anomalies.append(
                        f"ANOMALY: Statistically significant (p={result.primary_p_value:.4f}) "
                        f"but tiny effect size ({result.primary_effect_size:.4f}). "
                        f"May indicate large sample size masking practical insignificance."
                    )

        # Check for large effect size with non-significant p-value
        if result.primary_p_value is not None and result.primary_effect_size is not None:
            if result.primary_p_value > self.significance_threshold_relaxed:
                if abs(result.primary_effect_size) > 0.5:  # Cohen's d > 0.5 is medium/large
                    anomalies.append(
                        f"ANOMALY: Large effect size ({result.primary_effect_size:.4f}) "
                        f"but non-significant p-value (p={result.primary_p_value:.4f}). "
                        f"May indicate insufficient sample size or high variance."
                    )

        # Check for p-value exactly 0 or 1 (usually indicates error)
        if result.primary_p_value is not None:
            if result.primary_p_value == 0.0:
                anomalies.append(
                    "ANOMALY: P-value is exactly 0.0. This is unusual and may indicate "
                    "a computational error or extremely strong effect."
                )
            elif result.primary_p_value == 1.0:
                anomalies.append(
                    "ANOMALY: P-value is exactly 1.0. This may indicate a computational error."
                )

        # Check for inconsistent statistical tests
        if len(result.statistical_tests) >= 2:
            p_values = [t.p_value for t in result.statistical_tests if t.p_value is not None]
            if len(p_values) >= 2:
                # Check if some tests are significant and others not (may be expected, but worth noting)
                significant = [p < self.significance_threshold_relaxed for p in p_values]
                if any(significant) and not all(significant):
                    sig_count = sum(significant)
                    anomalies.append(
                        f"NOTE: {sig_count}/{len(significant)} statistical tests are significant. "
                        f"Mixed results may indicate variability in experimental conditions."
                    )

        # Check variable results for outliers
        if result.variable_results:
            for var in result.variable_results:
                if var.mean is not None and var.std is not None and var.std > 0:
                    cv = var.std / var.mean  # Coefficient of variation
                    if cv > 1.0:  # Very high variability
                        anomalies.append(
                            f"ANOMALY: Variable '{var.variable_name}' has very high variability "
                            f"(CV={cv:.2f}). This may affect result reliability."
                        )

        logger.debug(f"Detected {len(anomalies)} anomalies in {result.experiment_id}")
        return anomalies

    # ========================================================================
    # PATTERN DETECTION
    # ========================================================================

    def detect_patterns_across_results(
        self,
        results: List[ExperimentResult]
    ) -> List[str]:
        """
        Detect patterns across multiple experiment results.

        Looks for:
        - Consistent trends (e.g., always positive/negative effects)
        - Non-linear relationships
        - Unexpected similarities/differences

        Args:
            results: List of ExperimentResult objects

        Returns:
            list: List of pattern descriptions
        """
        patterns = []

        if len(results) < 2:
            return patterns

        # Extract p-values and effect sizes
        p_values = [r.primary_p_value for r in results if r.primary_p_value is not None]
        effect_sizes = [r.primary_effect_size for r in results if r.primary_effect_size is not None]

        # Pattern: Consistent effect direction
        if len(effect_sizes) >= 3:
            positive = [e > 0 for e in effect_sizes]
            if all(positive):
                patterns.append(
                    f"PATTERN: All {len(effect_sizes)} experiments show positive effects "
                    f"(mean effect size: {np.mean(effect_sizes):.3f}). "
                    f"This suggests a consistent underlying phenomenon."
                )
            elif not any(positive):
                patterns.append(
                    f"PATTERN: All {len(effect_sizes)} experiments show negative effects "
                    f"(mean effect size: {np.mean(effect_sizes):.3f}). "
                    f"This suggests a consistent inverse relationship."
                )

        # Pattern: Increasing/decreasing trend in effect sizes
        if len(effect_sizes) >= 4:
            # Simple monotonicity check
            increasing = all(effect_sizes[i] <= effect_sizes[i+1] for i in range(len(effect_sizes)-1))
            decreasing = all(effect_sizes[i] >= effect_sizes[i+1] for i in range(len(effect_sizes)-1))

            if increasing:
                patterns.append(
                    f"PATTERN: Effect sizes show increasing trend across {len(effect_sizes)} experiments "
                    f"({effect_sizes[0]:.3f} → {effect_sizes[-1]:.3f}). "
                    f"This may indicate a dose-response or temporal relationship."
                )
            elif decreasing:
                patterns.append(
                    f"PATTERN: Effect sizes show decreasing trend across {len(effect_sizes)} experiments "
                    f"({effect_sizes[0]:.3f} → {effect_sizes[-1]:.3f}). "
                    f"This may indicate diminishing returns or saturation effects."
                )

        # Pattern: Bimodal p-value distribution (either very significant or not)
        if len(p_values) >= 5:
            very_sig = sum(p < 0.01 for p in p_values)
            very_nonsig = sum(p > 0.1 for p in p_values)
            middle = len(p_values) - very_sig - very_nonsig

            if middle == 0 and very_sig > 0 and very_nonsig > 0:
                patterns.append(
                    f"PATTERN: Bimodal p-value distribution detected ({very_sig} highly significant, "
                    f"{very_nonsig} clearly non-significant, 0 borderline). "
                    f"This suggests different experimental conditions or subgroups."
                )

        logger.debug(f"Detected {len(patterns)} patterns across {len(results)} results")
        return patterns

    # ========================================================================
    # SIGNIFICANCE INTERPRETATION
    # ========================================================================
