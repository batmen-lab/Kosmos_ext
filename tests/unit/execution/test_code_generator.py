"""
Tests for code generation system.

Tests template matching, code generation, LLM fallback, and validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from kosmos.execution.code_generator import (
    ExperimentCodeGenerator,
    TTestComparisonCodeTemplate,
    CorrelationAnalysisCodeTemplate,
    LogLogScalingCodeTemplate,
    MLExperimentCodeTemplate,
    CodeTemplate
)
from kosmos.models.experiment import (
    ExperimentProtocol,
    ExperimentType,
    Variable,
    VariableType,
    StatisticalTestSpec,
    StatisticalTest,
    ProtocolStep,
    ResourceRequirements,
)


# Fixtures

@pytest.fixture
def ttest_protocol():
    """Create T-test experiment protocol."""
    return ExperimentProtocol(
        id="test-001",
        name="T-Test Experiment Protocol",
        hypothesis_id="hyp-001",
        domain="statistics",
        description="T-test comparison experiment for statistical analysis of treatment vs control groups",
        objective="Compare means between two groups using T-test",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.T_TEST,
                description="Two-sample T-test for group comparison",
                null_hypothesis="No difference between group means",
                variables=["group", "measurement"],
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Execute T-test",
                description="Load data and run T-test analysis",
                action="run_ttest",
                expected_duration_minutes=5
            )
        ],
        variables={
            "group": Variable(name="group", type=VariableType.INDEPENDENT, description="Group variable"),
            "measurement": Variable(name="measurement", type=VariableType.DEPENDENT, description="Measurement")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["group", "measurement"]},
        expected_duration_minutes=10
    )


@pytest.fixture
def correlation_protocol():
    """Create correlation analysis protocol."""
    return ExperimentProtocol(
        id="test-002",
        name="Correlation Analysis Protocol",
        hypothesis_id="hyp-002",
        domain="statistics",
        description="Correlation analysis experiment to test linear relationships between variables",
        objective="Calculate Pearson correlation between X and Y variables",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.CORRELATION,
                description="Pearson correlation analysis",
                null_hypothesis="No correlation between variables",
                variables=["x", "y"],
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Run Correlation",
                description="Calculate correlation coefficient",
                action="run_correlation",
                expected_duration_minutes=5
            )
        ],
        variables={
            "x": Variable(name="x", type=VariableType.INDEPENDENT, description="X variable"),
            "y": Variable(name="y", type=VariableType.DEPENDENT, description="Y variable")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["x", "y"]},
        expected_duration_minutes=10
    )


@pytest.fixture
def loglog_protocol():
    """Create log-log scaling protocol."""
    return ExperimentProtocol(
        id="test-003",
        name="Log-Log Scaling Analysis Protocol",
        hypothesis_id="hyp-003",
        domain="statistics",
        description="Power law and log-log scaling analysis to identify scale-free relationships",
        objective="Detect power law scaling relationships in data",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[],  # No statistical tests - matching is done via name/description
        steps=[
            ProtocolStep(
                step_number=1,
                title="Log-Log Analysis",
                description="Perform log-log scaling analysis",
                action="run_loglog",
                expected_duration_minutes=5
            )
        ],
        variables={
            "x": Variable(name="x", type=VariableType.INDEPENDENT, description="X variable input"),
            "y": Variable(name="y", type=VariableType.DEPENDENT, description="Y variable output")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["x", "y"]},
        expected_duration_minutes=10
    )


@pytest.fixture
def ml_protocol():
    """Create ML experiment protocol (using COMPUTATIONAL type for ML)."""
    return ExperimentProtocol(
        id="test-004",
        name="Machine Learning Classification Protocol",
        hypothesis_id="hyp-004",
        domain="machine_learning",
        description="Machine learning classification experiment to train and evaluate predictive models",
        objective="Train and evaluate ML model for classification task",
        experiment_type=ExperimentType.COMPUTATIONAL,  # ML uses COMPUTATIONAL type
        statistical_tests=[],  # ML doesn't use traditional statistical tests
        steps=[
            ProtocolStep(
                step_number=1,
                title="Train ML Model",
                description="Train and evaluate classification model",
                action="train_model",
                expected_duration_minutes=15
            )
        ],
        variables={
            "features": Variable(name="features", type=VariableType.INDEPENDENT, description="Input features for ML model training"),
            "target": Variable(name="target", type=VariableType.DEPENDENT, description="Target variable for prediction")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=1800,
            cpu_cores=2,
            memory_gb=4,
            storage_gb=1
        ),
        data_requirements={"format": "csv"},
        expected_duration_minutes=30
    )


def make_valid_protocol(
    id: str = "test-001",
    hypothesis_id: str = "hyp-001",
    name: str = "Test Experiment Protocol",
    domain: str = "statistics",
    description: str = "Test experiment for validating code generation functionality",
    objective: str = "Validate code generation",
    experiment_type: ExperimentType = ExperimentType.DATA_ANALYSIS,
    statistical_tests: list = None,
    variables: dict = None,
    steps: list = None,
    data_requirements: dict = None,
    expected_duration_minutes: int = 5,
) -> ExperimentProtocol:
    """Helper to create valid ExperimentProtocol with all required fields."""
    if statistical_tests is None:
        statistical_tests = []
    if variables is None:
        variables = {}
    if steps is None:
        steps = [
            ProtocolStep(
                step_number=1,
                title="Execute Analysis",
                description="Run the experiment analysis",
                action="run_analysis",
                expected_duration_minutes=5
            )
        ]
    if data_requirements is None:
        data_requirements = {}

    return ExperimentProtocol(
        id=id,
        name=name,
        hypothesis_id=hypothesis_id,
        domain=domain,
        description=description,
        objective=objective,
        experiment_type=experiment_type,
        statistical_tests=statistical_tests,
        steps=steps,
        variables=variables,
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements=data_requirements,
        expected_duration_minutes=expected_duration_minutes,
    )


@pytest.fixture
def code_generator():
    """Create code generator without LLM."""
    return ExperimentCodeGenerator(use_templates=True, use_llm=False)


@pytest.fixture
def code_generator_with_llm():
    """Create code generator with LLM."""
    mock_llm = Mock()
    mock_llm.generate.return_value = "import numpy as np\nresults = {'value': 42}"
    return ExperimentCodeGenerator(use_templates=True, use_llm=True, llm_client=mock_llm)


# Template Matching Tests

class TestTemplateMatching:
    """Tests for template matching logic."""

    def test_ttest_template_matches_ttest_protocol(self, ttest_protocol):
        """Test T-test template matches T-test protocol."""
        template = TTestComparisonCodeTemplate()
        assert template.matches(ttest_protocol)

    def test_correlation_template_matches_correlation_protocol(self, correlation_protocol):
        """Test correlation template matches correlation protocol."""
        template = CorrelationAnalysisCodeTemplate()
        assert template.matches(correlation_protocol)

    def test_loglog_template_matches_scaling_protocol(self, loglog_protocol):
        """Test log-log template matches scaling protocol."""
        template = LogLogScalingCodeTemplate()
        assert template.matches(loglog_protocol)

    def test_ml_template_matches_ml_protocol(self, ml_protocol):
        """Test ML template matches ML protocol."""
        template = MLExperimentCodeTemplate()
        assert template.matches(ml_protocol)

    def test_ttest_template_does_not_match_correlation(self, correlation_protocol):
        """Test T-test template doesn't match correlation protocol."""
        template = TTestComparisonCodeTemplate()
        assert not template.matches(correlation_protocol)

    def test_generator_selects_correct_template_for_ttest(self, code_generator, ttest_protocol):
        """Test generator selects T-test template."""
        template = code_generator._match_template(ttest_protocol)
        assert isinstance(template, TTestComparisonCodeTemplate)

    def test_generator_selects_correct_template_for_correlation(self, code_generator, correlation_protocol):
        """Test generator selects correlation template."""
        template = code_generator._match_template(correlation_protocol)
        assert isinstance(template, CorrelationAnalysisCodeTemplate)


# Code Generation Tests

class TestCodeGeneration:
    """Tests for code generation from templates."""

    def test_ttest_code_generation(self, code_generator, ttest_protocol):
        """Test T-test code generation."""
        code = code_generator.generate(ttest_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "DataAnalyzer" in code
        assert "ttest_comparison" in code
        assert "results" in code

    def test_correlation_code_generation(self, code_generator, correlation_protocol):
        """Test correlation code generation."""
        code = code_generator.generate(correlation_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "DataAnalyzer" in code
        assert "correlation_analysis" in code

    def test_loglog_code_generation(self, code_generator, loglog_protocol):
        """Test log-log scaling code generation."""
        code = code_generator.generate(loglog_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "DataAnalyzer" in code
        assert "log_log_scaling_analysis" in code

    def test_ml_code_generation(self, code_generator, ml_protocol):
        """Test ML code generation."""
        code = code_generator.generate(ml_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "MLAnalyzer" in code
        assert "run_experiment" in code or "cross_validate" in code

    def test_generated_code_is_valid_python(self, code_generator, ttest_protocol):
        """Test generated code is valid Python syntax."""
        import ast

        code = code_generator.generate(ttest_protocol)

        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        assert syntax_valid, f"Generated code has syntax errors:\n{code}"

    def test_generated_code_contains_result_variable(self, code_generator, ttest_protocol):
        """Test generated code assigns to results variable."""
        code = code_generator.generate(ttest_protocol)
        assert "results" in code or "result" in code


# LLM Fallback Tests

class TestLLMFallback:
    """Tests for LLM-based code generation fallback."""

    def test_llm_used_when_no_template_matches(self, code_generator_with_llm):
        """Test LLM used when no template matches."""
        # Create custom protocol that doesn't match any template
        # Use LITERATURE_SYNTHESIS which has no template
        custom_protocol = make_valid_protocol(
            id="custom-001",
            name="Custom Experiment Protocol",
            description="Novel experiment type that doesn't match any standard template",
            experiment_type=ExperimentType.LITERATURE_SYNTHESIS,
        )

        code = code_generator_with_llm.generate(custom_protocol)

        # Should have called LLM
        assert code_generator_with_llm.llm_client.generate.called
        assert code is not None

    def test_template_preferred_over_llm_when_available(self, code_generator_with_llm, ttest_protocol):
        """Test template used instead of LLM when available."""
        code = code_generator_with_llm.generate(ttest_protocol)

        # Should use template, not LLM
        assert "ttest_comparison" in code
        # LLM might still be called if enhance mode is on, but template should be primary

    def test_llm_can_enhance_template_code(self):
        """Test LLM enhancement of template code."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "# Enhanced\nimport pandas as pd\nresults = {}"

        generator = ExperimentCodeGenerator(
            use_templates=True,
            use_llm=True,
            llm_enhance_templates=True,
            llm_client=mock_llm
        )

        protocol = make_valid_protocol(
            description="Test experiment for LLM enhancement validation",
            statistical_tests=[
                StatisticalTestSpec(
                    test_type=StatisticalTest.T_TEST,
                    description="T-test for enhancement test",
                    null_hypothesis="No difference",
                    variables=["x"],
                )
            ],
        )

        code = generator.generate(protocol)

        # LLM should have been called for enhancement
        assert mock_llm.generate.called


# Validation Tests

class TestCodeValidation:
    """Tests for code validation and syntax checking."""

    def test_validate_syntax_valid_code(self, code_generator):
        """Test validation accepts valid code."""
        valid_code = "import numpy as np\nx = np.array([1, 2, 3])\nresults = {'mean': np.mean(x)}"

        try:
            code_generator._validate_syntax(valid_code)
            is_valid = True
        except Exception:
            is_valid = False

        assert is_valid

    def test_validate_syntax_invalid_code(self, code_generator):
        """Test validation rejects invalid code."""
        invalid_code = "import numpy as np\nx = [1, 2, 3\nresults = {'mean': x}"

        # _validate_syntax raises ValueError (which wraps SyntaxError message)
        with pytest.raises((SyntaxError, ValueError)):
            code_generator._validate_syntax(invalid_code)

    def test_generated_code_passes_validation(self, code_generator, ttest_protocol):
        """Test all generated code passes validation."""
        code = code_generator.generate(ttest_protocol)

        try:
            code_generator._validate_syntax(code)
            is_valid = True
        except Exception:
            is_valid = False

        assert is_valid


# Variable Extraction Tests

class TestVariableExtraction:
    """Tests for extracting variables from protocols."""

    def test_extract_dependent_variable(self, ttest_protocol):
        """Test extraction of dependent variable."""
        template = TTestComparisonCodeTemplate()

        dependent_vars = [
            var for var in ttest_protocol.variables.values()
            if var.type == VariableType.DEPENDENT
        ]

        assert len(dependent_vars) > 0
        assert dependent_vars[0].name == "measurement"

    def test_extract_independent_variable(self, ttest_protocol):
        """Test extraction of independent variable."""
        template = TTestComparisonCodeTemplate()

        independent_vars = [
            var for var in ttest_protocol.variables.values()
            if var.type == VariableType.INDEPENDENT
        ]

        assert len(independent_vars) > 0
        assert independent_vars[0].name == "group"


# Integration Tests

class TestCodeGeneratorIntegration:
    """Integration tests for code generator."""

    def test_end_to_end_ttest_generation(self, code_generator, ttest_protocol):
        """Test complete T-test code generation pipeline."""
        code = code_generator.generate(ttest_protocol)

        # Verify code structure
        assert "import" in code
        assert "DataAnalyzer" in code
        assert "ttest_comparison" in code
        assert "results" in code

        # Verify valid syntax
        import ast
        ast.parse(code)

    def test_end_to_end_ml_generation(self, code_generator, ml_protocol):
        """Test complete ML code generation pipeline."""
        code = code_generator.generate(ml_protocol)

        assert "import" in code
        assert "MLAnalyzer" in code
        assert "results" in code

        # Verify valid syntax
        import ast
        ast.parse(code)

    def test_generator_handles_minimal_protocol(self, code_generator):
        """Test generator handles minimal protocol gracefully."""
        minimal_protocol = make_valid_protocol(
            id="minimal-001",
            name="Minimal Protocol Test",
            description="Minimal protocol for testing graceful handling",
        )

        code = code_generator.generate(minimal_protocol)

        # Should generate fallback code
        assert code is not None
        assert len(code) > 0


# Edge Cases and Error Handling

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_generator_with_no_templates_and_no_llm(self):
        """Test generator behavior when both templates and LLM disabled."""
        generator = ExperimentCodeGenerator(use_templates=False, use_llm=False)

        protocol = make_valid_protocol(
            description="Test protocol for no-template no-LLM scenario",
        )

        code = generator.generate(protocol)

        # Should generate basic fallback
        assert code is not None
        assert "import" in code

    def test_generator_handles_empty_variables(self, code_generator):
        """Test generator handles protocol with no variables."""
        protocol = make_valid_protocol(
            description="Test protocol with no variables defined",
            experiment_type=ExperimentType.LITERATURE_SYNTHESIS,  # Use valid enum value
            variables={},  # Empty
        )

        code = code_generator.generate(protocol)
        assert code is not None

    def test_generator_handles_missing_data_requirements(self, code_generator):
        """Test generator handles missing data requirements."""
        protocol = make_valid_protocol(
            description="Test protocol with missing data requirements scenario",
            statistical_tests=[
                StatisticalTestSpec(
                    test_type=StatisticalTest.T_TEST,
                    description="T-test for missing data requirements test",
                    null_hypothesis="No difference",
                    variables=["x"],
                )
            ],
            data_requirements={},  # Empty
        )

        code = code_generator.generate(protocol)
        assert code is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Substance Validation Tests

class TestSubstanceValidation:
    """The gate that `ast.parse` alone could not be.

    The fragment in `test_the_observed_fragment_is_rejected` is the real one
    from the myocardial-fibrosis run: valid Python, executed, raised NameError,
    and the run still reported COMPLETED with empty results.
    """

    def test_the_observed_fragment_is_rejected(self):
        fragment = "analyzer = DataAnalyzer()\nanalysis_results = analyzer.run()"

        ExperimentCodeGenerator._validate_syntax(fragment)  # syntax is fine
        with pytest.raises(ValueError, match="never opens a dataset"):
            ExperimentCodeGenerator._validate_substance(fragment)

    def test_code_that_loads_data_but_returns_nothing_is_rejected(self):
        code = "import pandas as pd\ndf = pd.read_csv(data_path)\nprint(df.head())"

        with pytest.raises(ValueError, match="at module level"):
            ExperimentCodeGenerator._validate_substance(code)

    def test_code_that_returns_results_without_reading_data_is_rejected(self):
        """The shape a hallucinated answer takes: conclusions, no data."""
        code = "results = {'p_value': 0.01, 'effect': -0.3}"

        with pytest.raises(ValueError, match="never opens a dataset"):
            ExperimentCodeGenerator._validate_substance(code)

    def test_a_real_single_dataset_analysis_passes(self):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "results = {'n': len(df)}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_multi_dataset_analysis_passes(self):
        code = (
            "import pandas as pd\n"
            "pqtl = pd.read_csv(datasets['cis_pqtl'])\n"
            "t1 = pd.read_csv(datasets['t1_gwas'])\n"
            "merged = pqtl.merge(t1, on='variant')\n"
            "results = {'n_shared': len(merged)}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_results_built_only_inside_a_function_is_rejected(self):
        """A local `results` is gone before the sandbox looks for it.

        The capture reads `globals()` after the script finishes, so this script
        runs cleanly for eight seconds and hands back nothing. Observed exactly
        that: an experiment that raised no exception, produced no payload, and
        was reported as having produced no output.
        """
        code = (
            "import pandas as pd\n"
            "def main():\n"
            "    df = pd.read_csv(data_path)\n"
            "    results = {'n': len(df)}\n"
            "    return results\n"
            "out = main()\n"
        )

        with pytest.raises(ValueError, match="at module level"):
            ExperimentCodeGenerator._validate_substance(code)

    def test_assigning_the_call_at_module_level_passes(self):
        """The same script, one line different, is capturable."""
        code = (
            "import pandas as pd\n"
            "def main():\n"
            "    df = pd.read_csv(data_path)\n"
            "    return {'n': len(df)}\n"
            "results = main()\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_top_level_conditional_still_counts_as_module_scope(self):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "if len(df):\n"
            "    results = {'n': len(df)}\n"
            "else:\n"
            "    results = {'n': 0}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_the_gate_uses_the_executors_own_capture_names(self):
        """Drift here would gate on a contract the sandbox does not honour."""
        from kosmos.execution.executor import (
            RESULT_CAPTURE_NAMES,
            _SANDBOX_RESULT_CAPTURE,
        )

        for name in RESULT_CAPTURE_NAMES:
            assert f'"{name}"' in _SANDBOX_RESULT_CAPTURE, (
                f"{name} is gated on but the sandbox never looks for it"
            )
            code = (
                "import pandas as pd\n"
                "df = pd.read_csv(data_path)\n"
                f"{name} = {{'n': len(df)}}\n"
            )
            ExperimentCodeGenerator._validate_substance(code)

    def test_results_populated_by_key_passes(self):
        code = (
            "import duckdb\n"
            "con = duckdb.connect(data_path)\n"
            "results = {}\n"
            "results['rows'] = con.execute('select count(*) from t').fetchone()[0]\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_fragment_falls_back_to_the_template_instead_of_executing(self):
        """The behaviour that matters: reject, then still produce an experiment."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "analyzer = DataAnalyzer()"

        generator = ExperimentCodeGenerator(
            use_templates=True, use_llm=True, llm_client=mock_llm
        )
        protocol = make_valid_protocol(
            name="Custom Experiment Protocol",
            description="Novel experiment type that doesn't match any template",
            experiment_type=ExperimentType.LITERATURE_SYNTHESIS,
        )

        code = generator.generate(protocol)

        assert mock_llm.generate.called
        assert "DataAnalyzer" not in code, "the fragment reached the sandbox"
        ExperimentCodeGenerator._validate_substance(code)

    def test_every_shipped_template_passes_the_gate(self, ttest_protocol):
        """The fallback path must never be the thing the gate rejects."""
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)

        ExperimentCodeGenerator._validate_substance(generator.generate(ttest_protocol))


# Truncation Retry Tests

class TestTruncationRetry:
    """A script cut off mid-statement costs the whole multi-dataset analysis.

    The observed report line was: "fell back to the single-table template
    'generic_computational' ... Fallback reason: ValueError: Invalid Python
    syntax in generated code: unterminated string literal (detected at line
    54)". The model had not failed at the task; it ran out of room.
    """

    def _gen(self, llm, datasets):
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.use_templates = True
        gen.use_llm = True
        gen.llm_enhance_templates = False
        gen.llm_client = llm
        gen.datasets = datasets
        gen.dataset_context = None
        gen.templates = []

        # A template that MATCHES, mirroring production: the shipped
        # `generic_computational` template matches any DATA_ANALYSIS protocol,
        # so the fallback path is always a template and Step 2 never re-asks
        # the LLM. Stubbing this to None instead made every fallback take an
        # extra LLM call that no real run makes, so the call counts below
        # measured the test's own setup rather than the retry.
        class _Template:
            name = "generic_computational"

            def generate(self, protocol):
                return "import pandas as pd\nd = pd.read_csv(data_path)\nresults = {}\n"

        gen._match_template = lambda protocol: _Template()
        return gen

    def _datasets(self):
        return {"x": "/x.csv", "y": "/y.csv"}

    def _protocol(self):
        from kosmos.models.experiment import (
            ExperimentProtocol, ExperimentType, ProtocolStep, ResourceRequirements,
        )
        return ExperimentProtocol(
            name="cross-dataset comparison",
            hypothesis_id="hyp_1",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="biology",
            description="Join two datasets and correlate their effect sizes.",
            objective="Test whether effects agree across datasets.",
            steps=[ProtocolStep(
                step_number=1, title="Join the datasets",
                description="Join both datasets on the shared feature column.",
                action="Join both datasets on the shared feature column and correlate.",
            )],
            variables={},
            resource_requirements=ResourceRequirements(),
            statistical_tests=[],
        )

    def test_a_truncated_script_is_retried_and_the_analysis_survives(self):
        good = ("import pandas as pd\n"
                "a = pd.read_csv(datasets['x'])\n"
                "results = {'n': len(a)}\n")

        class _LLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, **kw):
                self.calls.append(prompt)
                if len(self.calls) == 1:
                    # Cut off mid-string, exactly as the provider returned it.
                    return "```python\nimport pandas as pd\nx = 'unterminated\n```"
                return f"```python\n{good}```"

        llm = _LLM()
        gen = self._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        code = ExperimentCodeGenerator.generate(gen, self._protocol())

        assert len(llm.calls) == 2, "the truncated response was not retried"
        assert "LENGTH LIMIT" in llm.calls[1], "the retry did not ask for a shorter script"
        assert gen.last_generation["path"] == "multi_dataset_llm"
        assert gen.generation_note() is None, "a recovered run must carry no caveat"

    def test_a_plain_syntax_error_is_retried_too(self):
        """A sampler does not reliably repeat an unbalanced paren.

        This used to assert the opposite, on the reasoning that a "real" syntax
        error reproduces. It cost a whole multi-dataset MR: the run fell back to
        a single-table template over `unmatched ')' at line 40` and reported a
        correlation between chromosome number and genomic position instead. One
        call is cheaper than the analysis the run exists to perform.
        """
        good = ("import pandas as pd\n"
                "a = pd.read_csv(datasets['x'])\n"
                "results = {'n': len(a)}\n")

        class _LLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, **kw):
                self.calls.append(prompt)
                if len(self.calls) == 1:
                    return "```python\nx = (1, 2))\n```"
                return f"```python\n{good}```"

        llm = _LLM()
        gen = self._gen(llm, self._datasets())
        ExperimentCodeGenerator.generate(gen, self._protocol())

        assert len(llm.calls) == 2, "a syntax error was not retried"
        assert "CORRECTION" in llm.calls[1]
        assert "unmatched" in llm.calls[1], "the retry did not quote the error"
        assert gen.last_generation["path"] == "multi_dataset_llm"
        assert gen.generation_note() is None

    def test_the_syntax_error_names_the_offending_line(self):
        """The model cannot see the file; the line text is what it can act on."""
        code = "import pandas as pd\ndf = pd.read_csv(data_path\nresults = {}\n"

        with pytest.raises(ValueError, match="the line reads"):
            ExperimentCodeGenerator._validate_syntax(code)

    def test_a_second_failure_still_falls_back(self):
        """One retry, not a loop."""

        class _LLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kw):
                self.calls += 1
                return "```python\nx = (1, 2))\n```"

        llm = _LLM()
        gen = self._gen(llm, self._datasets())
        ExperimentCodeGenerator.generate(gen, self._protocol())

        assert llm.calls == 2, "expected exactly one retry"
        assert "NOT the designed analysis" in gen.generation_note()

    def test_a_second_truncation_falls_back_and_says_so(self):
        class _LLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kw):
                self.calls += 1
                return "```python\nx = 'unterminated\n```"

        llm = _LLM()
        gen = self._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, self._protocol())

        assert llm.calls == 2, "expected exactly one retry"
        assert "unterminated" in gen.last_generation["fallback_reason"]
        assert "NOT the designed analysis" in gen.generation_note()

    def test_code_generation_asks_for_its_own_token_budget(self):
        seen = {}

        class _LLM:
            def generate(self, prompt, **kw):
                seen.update(kw)
                return "```python\nimport pandas as pd\nd = pd.read_csv(data_path)\nresults = {}\n```"

        gen = self._gen(_LLM(), {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, self._protocol())

        assert seen.get("max_tokens", 0) >= 16384

    def test_a_client_without_max_tokens_still_works(self):
        """Older clients take the prompt only; asking must not break them."""

        class _LLM:
            def generate(self, prompt):
                return "```python\nimport pandas as pd\nd = pd.read_csv(data_path)\nresults = {}\n```"

        gen = self._gen(_LLM(), {"x": "/x.csv", "y": "/y.csv"})
        code = ExperimentCodeGenerator.generate(gen, self._protocol())

        assert "results" in code


class TestUndefinedNameGate:
    """`NameError` in the sandbox is a crash the generator could have caught.

    Observed: the model defined `_validate_columns` and called `_validate_cols`.
    The container raised `NameError: name '_validate_cols' is not defined. Did
    you mean: '_validate_columns'?`, and the executor retried the SAME code
    three times -- re-execution cannot fix a misspelled call.
    """

    def test_the_observed_misspelling_is_caught(self):
        code = (
            "import pandas as pd\n"
            "def _validate_columns(df, cols):\n"
            "    return True\n"
            "cis = pd.read_csv(datasets['cis_pqtl'])\n"
            "_validate_cols(cis, ['ID'], 'cis_pqtl')\n"
            "results = {'n': len(cis)}\n"
        )

        with pytest.raises(ValueError, match="calls undefined name"):
            ExperimentCodeGenerator._validate_substance(code)

    def test_executor_injected_names_are_defined(self):
        """`data_path` and `datasets` exist at run time, not in the script."""
        code = (
            "import pandas as pd\n"
            "a = pd.read_csv(datasets['x'])\n"
            "b = pd.read_csv(data_path)\n"
            "results = {'n': len(a) + len(b)}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_builtins_comprehensions_and_handlers_are_not_flagged(self):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "squares = [x * x for x in range(10)]\n"
            "with open('/tmp/f') as fh:\n"
            "    body = fh.read()\n"
            "try:\n"
            "    z = int('1')\n"
            "except ValueError as err:\n"
            "    z = str(err)\n"
            "results = {'n': len(squares), 'z': z, 'b': len(body)}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_name_defined_in_another_function_is_not_flagged(self):
        """Whole-module, not per-scope: under-report rather than false-positive."""
        code = (
            "import pandas as pd\n"
            "def build():\n"
            "    frame = pd.read_csv(data_path)\n"
            "    return frame\n"
            "def use():\n"
            "    return len(frame)\n"
            "results = {'n': build() is not None}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_star_import_disables_the_check(self):
        """A star import defines an unknowable set; guessing would be worse."""
        code = (
            "from numpy import *\n"
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "results = {'m': mean(df.values)}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_an_undefined_name_is_retried_with_a_correction(self):
        bad = (
            "import pandas as pd\n"
            "def _validate_columns(df):\n"
            "    return True\n"
            "cis = pd.read_csv(datasets['x'])\n"
            "_validate_cols(cis)\n"
            "results = {'n': len(cis)}\n"
        )
        good = (
            "import pandas as pd\n"
            "cis = pd.read_csv(datasets['x'])\n"
            "results = {'n': len(cis)}\n"
        )

        class _LLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, **kw):
                self.calls.append(prompt)
                return f"```python\n{bad if len(self.calls) == 1 else good}```"

        llm = _LLM()
        gen = TestTruncationRetry()._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert len(llm.calls) == 2, "the undefined name was not retried"
        assert "CORRECTION" in llm.calls[1]
        assert "_validate_cols" in llm.calls[1], "the retry did not name the offender"
        assert "LENGTH LIMIT" not in llm.calls[1], "wrong retry instruction"
        assert gen.last_generation["path"] == "multi_dataset_llm"
        assert gen.generation_note() is None


class TestInventedAnalyzerMethods:
    """A class named in the prompt without its API gets methods invented on it.

    Observed: `analyzer.chi_square(table)` raised `AttributeError: 'DataAnalyzer'
    object has no attribute 'chi_square'` in the sandbox, and the executor
    retried the identical code three times. The prompt said only "Use
    DataAnalyzer for statistical tests".
    """

    def test_the_prompt_lists_the_real_methods(self):
        from kosmos.execution.code_generator import _analyzer_api_text, _analyzer_methods

        text = _analyzer_api_text()

        assert "ttest_comparison" in text
        assert "chi_square" not in text
        for method in _analyzer_methods():
            assert method in text, f"{method} missing from the prompt"

    def test_the_prompt_says_where_to_go_for_everything_else(self):
        from kosmos.execution.code_generator import _analyzer_api_text

        text = _analyzer_api_text()
        assert "scipy.stats" in text
        assert "chi-square" in text and "Mendelian randomization" in text

    def test_the_observed_invented_method_is_caught(self):
        code = (
            "import pandas as pd\n"
            "from kosmos.execution.data_analysis import DataAnalyzer\n"
            "analyzer = DataAnalyzer()\n"
            "df = pd.read_csv(data_path)\n"
            "chi2_p = analyzer.chi_square(df)\n"
            "results = {'p': chi2_p}\n"
        )

        with pytest.raises(ValueError, match="chi_square"):
            ExperimentCodeGenerator._validate_substance(code)

    def test_a_real_method_passes(self):
        code = (
            "import pandas as pd\n"
            "from kosmos.execution.data_analysis import DataAnalyzer\n"
            "analyzer = DataAnalyzer()\n"
            "df = pd.read_csv(data_path)\n"
            "results = analyzer.correlation_analysis(df, 'a', 'b')\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_attributes_on_other_objects_are_not_judged(self):
        """The check follows DataAnalyzer instances only."""
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "results = {'n': df.chi_square_lookalike_column.sum()}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_an_invented_method_is_retried_with_the_real_list(self):
        bad = (
            "import pandas as pd\n"
            "from kosmos.execution.data_analysis import DataAnalyzer\n"
            "analyzer = DataAnalyzer()\n"
            "d = pd.read_csv(datasets['x'])\n"
            "results = {'p': analyzer.chi_square(d)}\n"
        )
        good = (
            "import pandas as pd\n"
            "from scipy import stats\n"
            "d = pd.read_csv(datasets['x'])\n"
            "results = {'n': len(d)}\n"
        )

        class _LLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, **kw):
                self.calls.append(prompt)
                return f"```python\n{bad if len(self.calls) == 1 else good}```"

        llm = _LLM()
        gen = TestTruncationRetry()._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert len(llm.calls) == 2, "the invented method was not retried"
        assert "CORRECTION" in llm.calls[1]
        assert "chi_square" in llm.calls[1]
        assert gen.generation_note() is None


class TestUndefinedNameFalsePositives:
    """Every binding form Python has, or the gate rejects working code.

    A false positive here is expensive in a way a miss is not: it discards a
    correct multi-dataset analysis and falls back to a single-table template.
    Observed exactly that -- two consecutive generations rejected for undefined
    names `x` and `r`, both of them lambda parameters.
    """

    def test_a_lambda_parameter_is_bound(self):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "df['z'] = df.apply(lambda r: r['a'] + 1, axis=1)\n"
            "results = {'n': len(df)}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_sort_key_lambda_is_bound(self):
        code = (
            "import pandas as pd\n"
            "rows = pd.read_csv(data_path).to_dict('records')\n"
            "rows = sorted(rows, key=lambda x: x['p'])\n"
            "results = {'top': rows[0]}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_lambda_with_defaults_and_varargs_is_bound(self):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "f = lambda a, *rest, k=1, **kw: (a, rest, k, kw)\n"
            "results = {'v': f(len(df))}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_match_capture_is_bound(self):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "match len(df):\n"
            "    case 0:\n"
            "        n = 0\n"
            "    case other:\n"
            "        n = other\n"
            "results = {'n': n}\n"
        )

        ExperimentCodeGenerator._validate_substance(code)

    def test_a_genuinely_undefined_name_is_still_caught(self):
        """The fix must not blunt the check it is protecting."""
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv(data_path)\n"
            "df['z'] = df.apply(lambda r: r['a'] + missing_helper(r), axis=1)\n"
            "results = {'n': len(df)}\n"
        )

        with pytest.raises(ValueError, match="missing_helper"):
            ExperimentCodeGenerator._validate_substance(code)


class TestRuntimeErrorFeedback:
    """A traceback the model never sees is a lesson it cannot learn.

    The executor's own `retry_on_error` re-runs the SAME source, so three
    attempts at `KeyError: 'beta_t1'` produced three identical KeyErrors. Every
    runtime failure this project has hit -- NameError on a misspelled helper,
    AttributeError on an invented method, KeyError on a column the merge never
    created -- is correctable once the model is told what happened.
    """

    def _gen(self, llm, datasets):
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.use_templates = True
        gen.use_llm = True
        gen.llm_enhance_templates = False
        gen.llm_client = llm
        gen.datasets = datasets
        gen.dataset_context = None
        gen.templates = []

        class _T:
            name = "generic_computational"

            def generate(self, protocol):
                return "import pandas as pd\nd = pd.read_csv(data_path)\nresults = {}\n"

        gen._match_template = lambda protocol: _T()
        return gen

    def test_the_runtime_error_reaches_the_prompt(self):
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(datasets['x'])\nresults = {'n': len(d)}\n```")

        llm = _LLM()
        gen = self._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(
            gen, TestTruncationRetry()._protocol(),
            runtime_error="KeyError: 'beta_t1'",
        )

        assert "CORRECTION" in llm.prompts[0]
        assert "beta_t1" in llm.prompts[0]

    def test_the_correction_names_the_merge_suffix_trap(self):
        """The actual cause of `beta_t1`: no collision, so no suffix."""
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(datasets['x'])\nresults = {'n': len(d)}\n```")

        llm = _LLM()
        gen = self._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(
            gen, TestTruncationRetry()._protocol(), runtime_error="KeyError: 'beta_t1'",
        )

        prompt = llm.prompts[0]
        assert "merge suffixes ONLY to columns whose names collide" in prompt
        assert "assert the columns you need" in prompt

    def test_generation_without_an_error_is_unchanged(self):
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(datasets['x'])\nresults = {'n': len(d)}\n```")

        llm = _LLM()
        gen = self._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert "CORRECTION" not in llm.prompts[0]


class TestSandboxEnvironmentIsDeclared:
    """The model was guessing at the sandbox's package set, and guessed wrong.

    `from scipy.stats import multipletests` killed a run on line 61 before any
    analysis ran. `multipletests` is real -- it lives in statsmodels, which the
    sandbox HAS. Nothing told the model that.
    """

    def test_the_package_list_comes_from_the_image_requirements(self):
        import re
        from pathlib import Path

        from kosmos.execution.code_generator import SANDBOX_PACKAGES

        req = Path(__file__).resolve().parents[3] / "docker" / "sandbox" / "requirements.txt"
        declared = set()
        for line in req.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                declared.add(re.split(r"[=><!]", line)[0].strip())

        listed = {p.strip() for p in SANDBOX_PACKAGES.split(",")}
        assert listed == declared, "the prompt's package list has drifted from the image"

    def test_the_prompt_corrects_the_observed_import(self):
        from kosmos.execution.code_generator import _environment_text

        text = _environment_text()
        assert "statsmodels.stats.multitest" in text
        assert "NOT in scipy.stats" in text

    def test_the_prompt_states_there_is_no_network(self):
        """A model that thinks it can pip install writes code that cannot run."""
        from kosmos.execution.code_generator import _environment_text

        assert "NO network access" in _environment_text()

    def test_the_environment_section_reaches_the_generated_prompt(self):
        from kosmos.execution.code_generator import ExperimentCodeGenerator

        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(datasets['x'])\nresults = {'n': len(d)}\n```")

        llm = _LLM()
        gen = TestTruncationRetry()._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert "AVAILABLE PACKAGES" in llm.prompts[0]
        assert "statsmodels" in llm.prompts[0]


class TestProviderFailureIsRetryable:
    """The one failure a second draw fixes was the only one with no retry.

    A provider timeout made `_generate_with_llm` return None; `ast.parse(None)`
    raises TypeError, which the ValueError-keyed retry never sees. The run fell
    to the single-table template and the report blamed
    "compile() arg 1 must be a string" for what was an API timeout.
    """

    def test_a_provider_failure_is_retried_like_any_other_rejection(self):
        good = ("import pandas as pd\n"
                "a = pd.read_csv(datasets['x'])\n"
                "results = {'n': len(a)}\n")

        class _LLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("read timeout contacting the provider")
                return f"```python\n{good}```"

        llm = _LLM()
        gen = TestTruncationRetry()._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert llm.calls == 2, "the provider failure was not retried"
        assert gen.last_generation["path"] == "multi_dataset_llm"
        assert gen.generation_note() is None

    def test_the_recorded_reason_names_the_provider_not_the_parser(self):
        class _LLM:
            def generate(self, prompt, **kw):
                raise RuntimeError("read timeout contacting the provider")

        gen = TestTruncationRetry()._gen(_LLM(), {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        reason = gen.last_generation["fallback_reason"]
        assert "read timeout" in reason
        assert "compile()" not in reason, "the report would blame the parser"

    def test_the_prompt_states_the_filesystem_is_read_only(self):
        from kosmos.execution.code_generator import _environment_text

        text = _environment_text()
        assert "READ-ONLY except /tmp" in text
        assert "Errno 30" in text


class TestSymbolTableSyntaxErrors:
    """`ast.parse` is only the first of two passes the compiler makes.

    Observed: `SyntaxError: name 'p12' is assigned to before global
    declaration` killed the container at line 250 after 117ms, having passed a
    gate whose whole job is to catch syntax errors. `ast.parse` accepts it;
    the error is raised when the symbol table is built.
    """

    def test_the_observed_global_declaration_error_is_caught(self):
        code = (
            "import pandas as pd\n"
            "def f():\n"
            "    p12 = 1\n"
            "    global p12\n"
            "    return p12\n"
            "df = pd.read_csv(data_path)\n"
            "results = {'n': len(df)}\n"
        )

        import ast as _ast
        _ast.parse(code)  # the old gate accepted this

        with pytest.raises(ValueError, match="global declaration"):
            ExperimentCodeGenerator._validate_syntax(code)

    @pytest.mark.parametrize("snippet,fragment", [
        ("def f(a, a):\n    return a\n", "argument"),
        ("return 1\n", "outside function"),
        ("x = await foo()\n", "await"),
    ])
    def test_other_symbol_table_errors_are_caught(self, snippet, fragment):
        with pytest.raises(ValueError):
            ExperimentCodeGenerator._validate_syntax(snippet)

    def test_valid_code_still_passes_both_passes(self):
        code = (
            "import pandas as pd\n"
            "TOTAL = 0\n"
            "def add(df):\n"
            "    global TOTAL\n"
            "    TOTAL += len(df)\n"
            "    return TOTAL\n"
            "df = pd.read_csv(data_path)\n"
            "results = {'n': add(df)}\n"
        )

        ExperimentCodeGenerator._validate_syntax(code)
        ExperimentCodeGenerator._validate_substance(code)

    def test_a_symbol_table_error_is_retried_like_any_rejection(self):
        good = ("import pandas as pd\n"
                "a = pd.read_csv(datasets['x'])\n"
                "results = {'n': len(a)}\n")

        class _LLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, **kw):
                self.calls += 1
                if self.calls == 1:
                    return "```python\ndef f():\n    p12 = 1\n    global p12\n```"
                return f"```python\n{good}```"

        llm = _LLM()
        gen = TestTruncationRetry()._gen(llm, {"x": "/x.csv", "y": "/y.csv"})
        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert llm.calls == 2
        assert gen.generation_note() is None


class TestSingleDatasetRegeneration:
    """A template cannot be the remedy for its own runtime failure.

    Observed on a single-dataset run: the ML template fed the string label
    column to StandardScaler (`could not convert string to float: 'B'`), the
    director logged "regenerating once", and nothing was generated -- Step 1
    matched the same template, which ignores the error entirely and re-emitted
    byte-identical code that the caller then skipped as unchanged.
    """

    def _gen(self, llm, datasets):
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.use_templates = True
        gen.use_llm = True
        gen.llm_enhance_templates = False
        gen.llm_client = llm
        gen.datasets = datasets
        gen.dataset_context = None
        gen.templates = []
        self.template_used = []

        class _T:
            name = "ml_experiment"

            def generate(_self, protocol):
                self.template_used.append(True)
                return "import pandas as pd\nd = pd.read_csv(data_path)\nresults = {'n': len(d)}\n"

        gen._match_template = lambda protocol: _T()
        return gen

    def _llm(self):
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(data_path)\n"
                        "num = d.select_dtypes('number')\n"
                        "results = {'n': len(num)}\n```")

        return _LLM()

    def test_a_runtime_error_reaches_the_llm_on_a_single_dataset(self):
        llm = self._llm()
        gen = self._gen(llm, {"only": "/only.csv"})

        code = ExperimentCodeGenerator.generate(
            gen, TestTruncationRetry()._protocol(),
            runtime_error="ValueError: could not convert string to float: 'B'",
        )

        assert llm.prompts, "the retry never reached the model"
        assert "could not convert string to float" in llm.prompts[0]
        assert not self.template_used, "the failing template was re-emitted"
        assert "select_dtypes" in code, "the corrected draft was not used"

    def test_the_single_dataset_retry_keeps_the_single_table_prompt(self):
        """One dataset must not be handed the multi-dataset instructions."""
        llm = self._llm()
        gen = self._gen(llm, {"only": "/only.csv"})

        ExperimentCodeGenerator.generate(
            gen, TestTruncationRetry()._protocol(), runtime_error="KeyError: 'x'",
        )

        assert "ONLY the real dataset at" in llm.prompts[0]
        assert "DATASETS" not in llm.prompts[0]

    def test_a_normal_single_dataset_run_still_uses_the_template(self):
        """No runtime error means nothing changes: templates come first."""
        llm = self._llm()
        gen = self._gen(llm, {"only": "/only.csv"})

        ExperimentCodeGenerator.generate(gen, TestTruncationRetry()._protocol())

        assert self.template_used, "the template path was skipped"
        assert not llm.prompts, "the LLM was called when a template matched"


class TestTemplatesEmitWorkingFStrings:
    """Four of six templates printed literal placeholders instead of numbers.

    The lines were written as plain string literals carrying `{{...}}` -- the
    escaping you need inside an f-string, and exactly wrong outside one. The
    doubled braces survived verbatim into the generated code, so
    `print(f"Test Accuracy: {{results['...']['accuracy']:.4f}}")` printed the
    text `{results['...']['accuracy']:.4f}` rather than the accuracy. It never
    crashed, which is why it lasted: only the run's stdout was wrong.
    """

    def _protocol(self, name):
        from kosmos.models.experiment import (
            ExperimentProtocol, ExperimentType, ProtocolStep, ResourceRequirements,
        )

        return ExperimentProtocol(
            name=name, hypothesis_id="h",
            experiment_type=ExperimentType.DATA_ANALYSIS, domain="biology",
            description="A description long enough to satisfy validation.",
            objective="An objective long enough to satisfy validation.",
            steps=[ProtocolStep(step_number=1, title="Run",
                                description="Run the analysis fully.", action="Run it.")],
            variables={}, resource_requirements=ResourceRequirements(),
            statistical_tests=[],
        )

    def test_no_template_emits_doubled_braces(self):
        gen = ExperimentCodeGenerator(use_templates=True, use_llm=False)

        offenders = {}
        for template in gen.templates:
            code = template.generate(self._protocol(f"{template.name} experiment"))
            bad = [l.strip() for l in code.splitlines() if "{{" in l or "}}" in l]
            if bad:
                offenders[template.name] = bad[:2]

        assert not offenders, (
            f"templates emitting literal braces into generated code: {offenders}"
        )

    def test_every_template_still_compiles(self):
        """Unescaping must not have broken a dict or set literal."""
        gen = ExperimentCodeGenerator(use_templates=True, use_llm=False)

        for template in gen.templates:
            code = template.generate(self._protocol(f"{template.name} experiment"))
            compile(code, f"<{template.name}>", "exec")

    def test_a_generated_print_interpolates(self):
        """The observed line, executed, must print a number not a placeholder."""
        gen = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        ml = next(t for t in gen.templates if t.name == "ml_experiment")

        code = ml.generate(self._protocol("ml_experiment experiment"))
        line = next(l for l in code.splitlines() if "Test Accuracy" in l)

        results = {"train_test_results": {"accuracy": 0.9123}}
        printed = []
        exec(compile(line.strip(), "<line>", "exec"),
             {"results": results, "print": printed.append})

        assert printed == ["Test Accuracy: 0.9123"], printed


def test_both_prompt_branches_build():
    """A prompt is an f-string; a brace edit inside it fails only at call time.

    Unescaping `{{len(a)}}` to `{len(a)}` in the multi-dataset instructions
    turned a literal example into an interpolation, and
    `_data_access_instructions` raised `NameError: name 'a' is not defined` --
    caught upstream as a failed generation, so the run fell back to a
    single-table template and reported a generation failure that was really a
    typo in its own prompt.
    """
    gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
    gen.dataset_context = None

    single = gen._data_access_instructions({"only": "/a.csv"})
    multi = gen._data_access_instructions({"a": "/a.csv", "b": "/b.csv"})

    assert "ONLY the real dataset at" in single
    assert "no rows after merging X and Y: {len(a)} x {len(b)}" in multi, (
        "the example must reach the model as literal text, not be interpolated"
    )


class TestSingleDatasetTemplateIsDisclosed:
    """The protocol text and the template's output can describe different studies.

    Observed: a design promising a nested logistic regression, a
    likelihood-ratio test and an odds ratio for `number_inpatient`, printed
    above findings that were the ml_experiment template's accuracy, ROC-AUC and
    5-fold CV over all 150 columns. The run was reported as successful and
    nothing said the two were different analyses -- the caveat only fired when
    more than one dataset was mounted.
    """

    def _gen(self, datasets, last):
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.datasets = datasets
        gen.last_generation = last
        return gen

    def test_a_single_dataset_template_run_is_disclosed(self):
        gen = self._gen(
            {"only": "/a.csv"},
            {"path": "template", "template": "ml_experiment", "fallback_reason": None},
        )

        note = gen.generation_note()

        assert note is not None
        assert "ml_experiment" in note
        assert "not by code written for this protocol" in note
        # It is not a fallback, so it must not be described as one.
        assert "NOT the designed analysis" not in note

    def test_the_multi_dataset_wording_is_unchanged(self):
        gen = self._gen(
            {"a": "/a.csv", "b": "/b.csv"},
            {"path": "template", "template": "generic_computational",
             "fallback_reason": "ValueError: boom"},
        )

        note = gen.generation_note()

        assert "NOT the designed analysis" in note
        assert "2 datasets were mounted" in note

    def test_generated_code_still_carries_no_note(self):
        for datasets in ({"only": "/a.csv"}, {"a": "/a.csv", "b": "/b.csv"}):
            gen = self._gen(datasets, {"path": "multi_dataset_llm", "template": None,
                                       "fallback_reason": None})
            assert gen.generation_note() is None

    def test_the_basic_fallback_is_disclosed_despite_having_no_name(self):
        """This asserted `is None` and that silence was the bug: the least
        informative path of all reported nothing, so a 170 MB data dump was
        presented as a successful experiment with no provenance line."""
        gen = self._gen({"only": "/a.csv"},
                        {"path": "basic_template", "template": None, "fallback_reason": None})

        assert "BASIC FALLBACK" in gen.generation_note()


class TestMatcherRoutesToCodegen:
    """A template must not claim a protocol it cannot carry out.

    Observed: "Prior Inpatient Utilization as Predictor of Hospital Readmission
    Beyond LOS and Medication Count" -- eight steps, a baseline model, a full
    model, and a `custom` likelihood-ratio test between them -- was claimed by
    `ml_experiment` because its description contained "logistic regression".
    The template fit one model over all 150 columns and reported accuracy and
    AUC. The run succeeded and answered a different question.
    """

    def _protocol(self, name, tests=(), etype=None, description=None):
        from kosmos.models.experiment import (
            ExperimentProtocol, ExperimentType, ProtocolStep, ResourceRequirements,
            StatisticalTestSpec, StatisticalTest,
        )

        specs = [
            StatisticalTestSpec(
                test_type=StatisticalTest(t),
                description=f"A {t} test, described at sufficient length.",
                null_hypothesis="H0: no effect",
                variables=["x"],
            )
            for t in tests
        ]
        return ExperimentProtocol(
            name=name, hypothesis_id="h",
            experiment_type=etype or ExperimentType.DATA_ANALYSIS, domain="biology",
            description=description or "Fit a nested logistic regression and compare the models.",
            objective="Test the incremental value of one predictor.",
            steps=[ProtocolStep(step_number=1, title="Run",
                                description="Run the analysis fully.", action="Run it.")],
            variables={}, resource_requirements=ResourceRequirements(),
            statistical_tests=specs,
        )

    def _gen(self, use_llm=True):
        return ExperimentCodeGenerator(
            use_templates=True, use_llm=use_llm,
            llm_client=object() if use_llm else None,
        )

    def test_a_custom_test_is_not_claimed_by_any_template(self):
        """`custom` means bespoke: no canned template implements it."""
        protocol = self._protocol("Nested logistic regression comparison", tests=["custom"])

        assert self._gen()._match_template(protocol) is None

    def test_a_declared_t_test_still_reaches_its_template(self):
        protocol = self._protocol("Group comparison", tests=["t_test"])

        matched = self._gen()._match_template(protocol)

        assert matched is not None and matched.name == "ttest_comparison"

    def test_a_correlation_protocol_still_reaches_its_template(self):
        protocol = self._protocol("Correlation of effect sizes", tests=["correlation"])

        matched = self._gen()._match_template(protocol)

        assert matched is not None and matched.name == "correlation_analysis"

    def test_a_test_no_template_handles_routes_to_codegen(self):
        """chi-square is declared by no template, so nothing may claim it."""
        protocol = self._protocol("Enrichment of variants", tests=["chi_square"])

        assert self._gen()._match_template(protocol) is None

    def test_the_catch_all_defers_to_the_llm_when_one_exists(self):
        """Reaching the catch-all means no template recognised the experiment."""
        protocol = self._protocol(
            "Some novel analysis",
            description="Summarise the released variants and describe their spread.",
        )

        assert self._gen(use_llm=True)._match_template(protocol) is None

    def test_the_catch_all_is_still_used_without_an_llm(self):
        protocol = self._protocol(
            "Some novel analysis",
            description="Summarise the released variants and describe their spread.",
        )

        matched = self._gen(use_llm=False)._match_template(protocol)

        assert matched is not None and matched.name == "generic_computational"

    def test_a_model_fitting_protocol_with_no_declared_tests_keeps_its_template(self):
        """`can_satisfy` cannot discriminate when nothing is declared, and a
        protocol that declares no hypothesis test genuinely is a model-fitting
        exercise -- which is what ml_experiment does."""
        protocol = self._protocol("Predict readmission with logistic regression")

        matched = self._gen()._match_template(protocol)

        assert matched is not None and matched.name == "ml_experiment"


class TestEmptyFrameCorrection:
    """sklearn reports an empty frame as a split problem, so the model fixes the split.

    Observed three times on one dataset: the model wrote
    `df['readmitted'].map({'<30': 1, 'NO': 0, '>30': 0})` -- the ORIGINAL UCI
    encoding -- against a column this copy stores as integers 0/1. Every row
    became NaN and sklearn raised `With n_samples=0 ... the resulting train set
    will be empty`. That message names the split, not the mapping, so the next
    draft repeated the mapping.
    """

    def test_the_observed_error_is_recognised(self):
        from kosmos.execution.code_generator import _looks_like_an_empty_frame

        assert _looks_like_an_empty_frame(
            "ValueError: With n_samples=0, test_size=0.3 and train_size=None, "
            "the resulting train set will be empty."
        )

    def test_unrelated_errors_are_not_given_the_advice(self):
        from kosmos.execution.code_generator import _looks_like_an_empty_frame

        for other in ("KeyError: 'age'",
                      "ValueError: could not convert string to float: 'B'",
                      "NameError: name '_validate_cols' is not defined"):
            assert not _looks_like_an_empty_frame(other), other

    def test_the_advice_reaches_the_correction_prompt(self):
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(data_path)\nresults = {'n': len(d)}\n```")

        llm = _LLM()
        gen = TestSingleDatasetRegeneration()._gen(self, llm) if False else None

        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.use_templates = True
        gen.use_llm = True
        gen.llm_enhance_templates = False
        gen.llm_client = llm
        gen.datasets = {"only": "/only.csv"}
        gen.dataset_context = None
        gen.templates = []
        gen._match_template = lambda protocol: None

        ExperimentCodeGenerator.generate(
            gen, TestTruncationRetry()._protocol(),
            runtime_error="ValueError: With n_samples=0, the resulting train set will be empty.",
        )

        prompt = llm.prompts[0]
        assert "a VALUE you compared against does not occur in this data" in prompt
        assert "not that the split or the sample size is wrong" in prompt
        assert "integers 0/1 where the original release used strings like '<30'" in prompt

    def test_an_ordinary_correction_is_unchanged(self):
        class _LLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kw):
                self.prompts.append(prompt)
                return ("```python\nimport pandas as pd\n"
                        "d = pd.read_csv(data_path)\nresults = {'n': len(d)}\n```")

        llm = _LLM()
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.use_templates = True
        gen.use_llm = True
        gen.llm_enhance_templates = False
        gen.llm_client = llm
        gen.datasets = {"only": "/only.csv"}
        gen.dataset_context = None
        gen.templates = []
        gen._match_template = lambda protocol: None

        ExperimentCodeGenerator.generate(
            gen, TestTruncationRetry()._protocol(), runtime_error="KeyError: 'age'",
        )

        assert "CORRECTION" in llm.prompts[0]
        assert "does not occur in this data" not in llm.prompts[0]


class TestBasicFallbackSummarises:
    """`results = {'data': df.to_dict(), 'shape': df.shape}` wrote 170 MB.

    On an 81,410 x 151 table the basic fallback serialised 12 million cells
    into the results row. The report rendered it as `data: 151 entries`, the
    run was recorded as a successful experiment, and nothing said the analysis
    had not been performed -- the payload passed every emptiness check by being
    enormous.
    """

    def _protocol(self):
        from kosmos.models.experiment import (
            ExperimentProtocol, ExperimentType, ProtocolStep, ResourceRequirements,
        )

        return ExperimentProtocol(
            name="basic fallback", hypothesis_id="h",
            experiment_type=ExperimentType.DATA_ANALYSIS, domain="biology",
            description="A description long enough to satisfy validation.",
            objective="An objective long enough to satisfy validation.",
            steps=[ProtocolStep(step_number=1, title="Run",
                                description="Run the analysis fully.", action="Run it.")],
            variables={}, resource_requirements=ResourceRequirements(),
        )

    def _run_it(self, df):
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        code = ExperimentCodeGenerator._generate_basic_template(gen, self._protocol())
        body = "\n".join(
            l for l in code.splitlines()
            if not l.startswith(("import ", "df = pd.read_csv"))
        )
        ns = {"df": df, "print": lambda *a, **k: None}
        exec(compile(body, "<basic>", "exec"), ns)
        return ns["results"]

    def test_the_frame_is_not_dumped(self):
        import json

        import numpy as np
        import pandas as pd

        df = pd.DataFrame({f"c{i}": np.arange(500, dtype=float) for i in range(60)})
        results = self._run_it(df)

        assert "data" not in results, "the frame itself is back in the payload"
        payload = len(json.dumps(results, default=str))
        assert payload < 100_000, f"{payload:,} bytes for a 500x60 frame"

    def test_it_reports_shape_missingness_and_statistics(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0, None], "b": [3.0, 4.0, 5.0]})
        results = self._run_it(df)

        # The template drops incomplete rows, so `n_rows` is the CLEANED count
        # and missingness is counted before that -- counting after always
        # reports zero and hides what is worth reporting.
        assert results["rows_before_cleaning"] == 3
        assert results["n_rows"] == 2 and results["n_columns"] == 2
        assert results["missing_by_column"] == {"a": 1}
        assert "mean" in results["numeric_summary"]["b"]

    def test_it_says_it_did_not_test_the_hypothesis(self):
        import pandas as pd

        results = self._run_it(pd.DataFrame({"a": [1.0, 2.0]}))

        assert "basic fallback template" in results["note"]
        assert "not a test of the hypothesis" in results["note"]

    def test_the_basic_path_is_disclosed_in_the_report(self):
        gen = ExperimentCodeGenerator.__new__(ExperimentCodeGenerator)
        gen.datasets = {"only": "/a.csv"}
        gen.last_generation = {"path": "basic_template", "template": None,
                               "fallback_reason": None}

        note = gen.generation_note()

        assert note is not None, "the least informative path disclosed nothing"
        assert "BASIC FALLBACK" in note
        assert "does not test the hypothesis" in note
