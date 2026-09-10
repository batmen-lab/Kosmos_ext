"""
Code generation for experiment execution.

Generates executable Python code from experiment protocols using:
1. Template-based generation for common patterns (from kosmos-figures)
2. LLM-based generation for novel experiments
3. Hybrid approach combining both

Based on patterns from docs/integration-plan.md.
"""

import ast
import builtins
import os
from typing import Dict, List, Optional, Any, Callable
import logging
from pathlib import Path

from kosmos.models.experiment import ExperimentProtocol, ProtocolStep, ExperimentType
from kosmos.models.hypothesis import Hypothesis
from kosmos.core.llm import ClaudeClient
from kosmos.core.prompts import EXPERIMENT_DESIGNER

logger = logging.getLogger(__name__)


class CodeTemplate:
    """Base class for code generation templates."""

    def __init__(self, name: str, experiment_type: ExperimentType):
        """
        Initialize code template.

        Args:
            name: Template name
            experiment_type: Type of experiment this template handles
        """
        self.name = name
        self.experiment_type = experiment_type

    # The declared statistical tests this template actually emits code for.
    # Empty means it emits no hypothesis test at all -- `ml_experiment` reports
    # accuracy and AUC, `generic_computational` reports descriptives.
    handled_tests: frozenset = frozenset()

    def can_satisfy(self, protocol: ExperimentProtocol) -> bool:
        """Can this template perform the tests the protocol DECLARES?

        Matching on experiment type and keywords let a template claim a
        protocol it could not carry out. Observed: a protocol named "Prior
        Inpatient Utilization as Predictor of Hospital Readmission Beyond LOS
        and Medication Count", declaring a `custom` likelihood-ratio test
        between a baseline and a full model across eight steps, was claimed by
        `ml_experiment` because its description contained the words "logistic
        regression". The template fit ONE model over all 150 columns and
        reported accuracy and AUC. The run succeeded, and answered a different
        question than the one designed.

        `custom` is decisive: a protocol that declares a bespoke test is by
        definition asking for something no canned template implements, so a
        template that claims it will always substitute its own analysis.
        """
        for test in protocol.statistical_tests or []:
            declared = getattr(test.test_type, "value", None) or str(test.test_type)
            if declared not in self.handled_tests:
                return False
        return True

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if this template matches the protocol."""
        return protocol.experiment_type == self.experiment_type

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate code from protocol."""
        raise NotImplementedError

    @staticmethod
    def clean_data_lines() -> list[str]:
        """Emit the row-cleaning step, without letting one empty column win.

        `df.dropna()` drops a row if ANY column is null. Harmonised omics tables
        routinely carry a column that is empty for every row -- scPerturb leaves
        `disease` blank across all 247,914 cells of the RPE1 screen, because the
        line is healthy -- and a single such column makes the plain dropna()
        delete the ENTIRE dataset. Downstream that surfaced as
        `ValueError: x and y must have length at least 2` from pearsonr, which
        says nothing about the real cause.

        So: drop all-empty COLUMNS first (they carry no information and cannot
        be an analysis variable), then drop rows. If cleaning still empties the
        frame, say which columns did it rather than letting an empty frame flow
        into scipy.
        """
        return [
            "_na_counts = df.isna().sum()",
            "_all_empty = [c for c in df.columns if _na_counts[c] == len(df)]",
            "if _all_empty:",
            "    print(f'Dropping {len(_all_empty)} all-empty column(s): {_all_empty}')",
            "    df = df.drop(columns=_all_empty)",
            "_rows_before = len(df)",
            "df = df.dropna()",
            "if len(df) == 0 and _rows_before > 0:",
            "    _worst = _na_counts[_na_counts > 0].sort_values(ascending=False).head(5)",
            "    raise RuntimeError(",
            "        f'Cleaning removed all {_rows_before} rows: every row has a null in '",
            "        f'at least one column. Nulls per column (top 5): '",
            "        f'{ {c: int(n) for c, n in _worst.items()} }. '",
            "        'Restrict the analysis to the columns it needs rather than '",
            "        'dropping rows on every column.')",
        ]

    @staticmethod
    def column_binding_lines(wanted: list[str]) -> list[str]:
        """Emit code binding protocol variable names to the dataset's columns.

        A protocol's variable names are written by a model reasoning about the
        science: it asks for `working_hours` while the column is
        `hours_per_week`, or `marital_status` where the file says
        `marital.status`. Nothing downstream reconciled the two, so every
        template indexed `df['<protocol name>']` and died on KeyError against
        any dataset whose columns did not happen to match.

        Rather than patch each indexing site, this renames the *frame* once, so
        every later `df['<protocol name>']` reference resolves unchanged.

        Matching is exact first, then case- and separator-insensitive
        (`working_hours` ~ `Working Hours` ~ `working.hours`). It deliberately
        does NOT guess beyond that: binding `working_hours` to `hours_per_week`
        is a judgement about meaning, not spelling, and a wrong guess would
        silently analyse the wrong column. Unresolved names raise, naming what
        was available, so the failure says which column was missing instead of
        surfacing as a bare KeyError.
        """
        return [
            "",
            "# Bind protocol variable names to real columns (see column_binding_lines)",
            "def _canon(s):",
            "    return ''.join(ch for ch in str(s).lower() if ch.isalnum())",
            "_by_canon = {}",
            "for _c in df.columns:",
            "    _by_canon.setdefault(_canon(_c), _c)",
            f"for _want in {wanted!r}:",
            "    if _want in df.columns:",
            "        continue",
            "    _hit = _by_canon.get(_canon(_want))",
            "    if _hit is None:",
            "        raise ValueError(",
            "            f\"protocol names variable '{_want}', which is not a column \"",
            "            f'in this dataset. Available columns: {list(df.columns)}'",
            "        )",
            "    df = df.rename(columns={_hit: _want})",
            "",
        ]


class TTestComparisonCodeTemplate(CodeTemplate):
    """
    Template for t-test comparison experiments.

    Pattern from: kosmos-figures Figure_2_hypothermia_nucleotide_salvage
    """

    def __init__(self):
        super().__init__("ttest_comparison", ExperimentType.DATA_ANALYSIS)
        self.handled_tests = frozenset({"t_test"})

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol needs t-test comparison."""
        if protocol.experiment_type != ExperimentType.DATA_ANALYSIS:
            return False

        # Check for t-test in statistical tests
        for test in protocol.statistical_tests:
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 't_test' in test_type_str.lower() or 't-test' in test_type_str.lower():
                return True

        return False

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate t-test comparison code."""
        # Extract variable information
        indep_vars = [v for v in protocol.variables.values() if v.type.value == 'independent']
        dep_vars = [v for v in protocol.variables.values() if v.type.value == 'dependent']

        group_var = indep_vars[0].name if indep_vars else 'group'
        measure_var = dep_vars[0].name if dep_vars else 'measurement'

        # Preferred group labels from the protocol. These are LABELS, not data
        # values: a protocol names its arms 'experimental' / 'other_occupations'
        # while the column actually holds 'Exec-managerial', 'Craft-repair'.
        # Emitting them as literals produced a t-test over two EMPTY frames on
        # every real dataset -- it only ever worked against the synthetic example
        # in DataAnalyzer.ttest_comparison's own docstring. They are now a
        # preference that the generated code checks against the data, falling
        # back to the levels actually present.
        groups = []
        if protocol.control_groups:
            groups.append(protocol.control_groups[0].name)
        else:
            groups.append('control')
        groups.append('experimental')

        # Get random seed from protocol or use default
        seed = getattr(protocol, 'random_seed', 42) or 42
        n_samples = 100  # Default sample size

        # Read effect size from protocol if available; default to 0.0 (null hypothesis)
        effect_size = 0.0
        if protocol.statistical_tests:
            es = getattr(protocol.statistical_tests[0], 'expected_effect_size', None)
            if es is not None:
                effect_size = es

        code_lines = [
            "# T-Test Comparison Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from pathlib import Path",
            "from kosmos.execution.data_analysis import DataAnalyzer",
            "",
            "# Data loading (real dataset only - no synthetic fallback)",
            f"# Expected format: CSV with columns '{group_var}' and '{measure_var}'",
            "df = None",
            "if 'data_path' in dir() and data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        raise RuntimeError(f'Failed to load required dataset from {data_path}: {e}. Real data only; no synthetic fallback.')",
            "if df is None:",
            "    raise RuntimeError('No dataset available at data_path; refusing to fabricate synthetic data (real data only).')",
            "",
            "# Clean data",
            *self.clean_data_lines(),
            *self.column_binding_lines([group_var, measure_var]),
            "",
            "# A t-test needs a NUMERIC dependent variable. Without this check a",
            "# categorical measure reaches scipy and surfaces as an opaque",
            "# 'could not convert string to float' from inside shapiro().",
            f"if not pd.api.types.is_numeric_dtype(df['{measure_var}']):",
            f"    raise ValueError(f\"measure column '{measure_var}' has dtype \"",
            f"                     f\"{{df['{measure_var}'].dtype}}, which is not numeric; \"",
            "                     f'a t-test needs a numeric dependent variable')",
            "",
            "# Resolve the comparison arms against the data. The protocol's group",
            "# names are labels; the column holds whatever it holds.",
            f"df['{group_var}'] = df['{group_var}'].astype(str)",
            f"_preferred = [{groups[1]!r}, {groups[0]!r}]",
            f"_levels = df['{group_var}'].value_counts()",
            "_available = [str(x) for x in _levels.index]",
            "_chosen = [g for g in _preferred if g in _available]",
            "if len(_chosen) < 2:",
            "    if len(_available) < 2:",
            f"        raise ValueError(f\"column '{group_var}' has \"",
            "                         f'{len(_available)} distinct value(s); '",
            "                         f'a two-group comparison needs at least 2')",
            "    _chosen = _available[:2]",
            f"    print(f\"Protocol arms {{_preferred}} are not values of '{group_var}'; \"",
            "          f'comparing the two most common levels instead: {_chosen}')",
            "_g_a, _g_b = _chosen[0], _chosen[1]",
            "",
            "# Check statistical assumptions before t-test",
            f"_group_data = {{g: df[df['{group_var}']==g]['{measure_var}'].values for g in (_g_a, _g_b)}}",
            "for _gname, _gvals in _group_data.items():",
            "    _shap_stat, _shap_p = stats.shapiro(_gvals[:5000]) if len(_gvals) >= 8 else (1.0, 1.0)",
            "    if _shap_p < 0.05:",
            "        print(f'WARNING: Normality assumption violated for group {_gname} (Shapiro p={_shap_p:.4f})')",
            "_groups_list = list(_group_data.values())",
            "if len(_groups_list) == 2:",
            "    _lev_stat, _lev_p = stats.levene(*_groups_list)",
            "    if _lev_p < 0.05:",
            "        print(f'WARNING: Equal variance assumption violated (Levene p={_lev_p:.4f})')",
            "",
            "# Perform t-test comparison",
            "analyzer = DataAnalyzer()",
            f"result = analyzer.ttest_comparison(",
            f"    df, '{group_var}', '{measure_var}',",
            "    groups=(_g_a, _g_b),",
            f"    log_transform={'True' if any('log' in str(s.action).lower() for s in protocol.steps) else 'False'}",
            ")",
            "",
            "# Print results",
            "print(f\"T-statistic: {result['t_statistic']:.4f}\")",
            "print(f\"P-value: {result['p_value']:.6f}\")",
            "print(f\"Significance: {result['significance_label']}\")",
            "print(f\"Mean difference: {result['mean_difference']:.4f}\")",
            "",
            "# Generate publication-quality figure (Issue #60)",
            "from kosmos.analysis.visualization import PublicationVisualizer",
            "viz = PublicationVisualizer()",
            "",
            f"# Create data dictionary for box plot",
            f"groups_unique = df['{group_var}'].unique()",
            f"plot_data = {{str(g): df[df['{group_var}']==g]['{measure_var}'].values for g in groups_unique}}",
            "",
            "# Generate box plot if figure_path is provided",
            "if 'figure_path' in dir() and figure_path:",
            "    viz.box_plot_with_points(",
            "        data=plot_data,",
            f"        title={protocol.name!r},",
            f"        y_label='{measure_var}',",
            "        output_path=str(figure_path)",
            "    )",
            "    result['figure_path'] = str(figure_path)",
            "",
            "# Propagate data source and assumption checks into results",
            "if '_data_source' in dir():",
            "    result['data_source'] = _data_source",
            "result['assumption_checks'] = {",
            "    'normality_tested': True,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Return results for collection",
            "results = result"
        ]

        return "\n".join(code_lines)


class CorrelationAnalysisCodeTemplate(CodeTemplate):
    """
    Template for correlation analysis experiments.

    Pattern from: kosmos-figures Figure_3_perovskite_solar_cell
    """

    def __init__(self):
        super().__init__("correlation_analysis", ExperimentType.DATA_ANALYSIS)
        self.handled_tests = frozenset({"correlation", "regression"})

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol needs correlation analysis."""
        if protocol.experiment_type != ExperimentType.DATA_ANALYSIS:
            return False

        # Check for correlation in statistical tests or protocol name
        for test in protocol.statistical_tests:
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 'correlation' in test_type_str.lower() or 'regression' in test_type_str.lower():
                return True

        return 'correlation' in protocol.name.lower()

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate correlation analysis code."""
        # Get variables
        vars_list = list(protocol.variables.keys())
        x_var = vars_list[0] if len(vars_list) > 0 else 'x'
        y_var = vars_list[1] if len(vars_list) > 1 else 'y'

        # Determine correlation method
        method = 'pearson'
        for test in protocol.statistical_tests:
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 'spearman' in test_type_str.lower():
                method = 'spearman'
                break

        seed = getattr(protocol, 'random_seed', 42) or 42

        code_lines = [
            "# Correlation Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from pathlib import Path",
            "from kosmos.execution.data_analysis import DataAnalyzer",
            "",
            "# Data loading (real dataset only - no synthetic fallback)",
            f"# Expected format: CSV with columns '{x_var}' and '{y_var}'",
            "df = None",
            "if 'data_path' in dir() and data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        raise RuntimeError(f'Failed to load required dataset from {data_path}: {e}. Real data only; no synthetic fallback.')",
            "if df is None:",
            "    raise RuntimeError('No dataset available at data_path; refusing to fabricate synthetic data (real data only).')",
            "",
            "# Clean data",
            *self.clean_data_lines(),
            *self.column_binding_lines([x_var, y_var]),
            "",
            "# Check statistical assumptions before correlation",
            f"for _col in ['{x_var}', '{y_var}']:",
            "    _vals = df[_col].values",
            "    if len(_vals) >= 8:",
            "        _shap_stat, _shap_p = stats.shapiro(_vals[:5000])",
            "        if _shap_p < 0.05:",
            f"            print(f'WARNING: Normality assumption violated for {{_col}} (Shapiro p={{_shap_p:.4f}})')",
            "",
            "# Perform correlation analysis (Pearson + Spearman)",
            "analyzer = DataAnalyzer()",
            f"result = analyzer.correlation_analysis(",
            f"    df, '{x_var}', '{y_var}',",
            f"    method='{method}'",
            ")",
            "",
            "# Also compute Spearman rank correlation for nonlinear relationships",
            "from scipy.stats import spearmanr, pearsonr",
            f"_x_vals = df['{x_var}'].values",
            f"_y_vals = df['{y_var}'].values",
            "_pearson_r, _pearson_p = pearsonr(_x_vals, _y_vals)",
            "_spearman_r, _spearman_p = spearmanr(_x_vals, _y_vals)",
            "result['pearson_r'] = float(_pearson_r)",
            "result['pearson_p'] = float(_pearson_p)",
            "result['spearman_r'] = float(_spearman_r)",
            "result['spearman_p'] = float(_spearman_p)",
            "",
            "# Use the more significant result for hypothesis support",
            "if _spearman_p < _pearson_p:",
            "    result['best_method'] = 'spearman'",
            "    result['best_correlation'] = float(_spearman_r)",
            "    result['best_p_value'] = float(_spearman_p)",
            "else:",
            "    result['best_method'] = 'pearson'",
            "    result['best_correlation'] = float(_pearson_r)",
            "    result['best_p_value'] = float(_pearson_p)",
            "result['supports_hypothesis'] = result['best_p_value'] < 0.05",
            "",
            "# Print results",
            "print(f\"Pearson r: {_pearson_r:.4f}, p={_pearson_p:.6f}\")",
            "print(f\"Spearman rho: {_spearman_r:.4f}, p={_spearman_p:.6f}\")",
            "print(f\"Best method: {result['best_method']} (p={result['best_p_value']:.6f})\")",
            # f-string HERE so `method` is resolved at generation time. As a
            # plain string it passed `{method}` through into the emitted code,
            # where nothing of that name exists -- every correlation experiment
            # died on NameError before printing a single result.
            f"print(f\"Correlation ({method}): {{result['correlation']:.4f}}\")",
            "print(f\"P-value: {result['p_value']:.6f}\")",
            "print(f\"R-squared: {result['r_squared']:.4f}\")",
            "print(f\"Significance: {result['significance']}\")",
            "print(f\"Regression equation: {result['equation']}\")",
            "",
            "# Generate publication-quality figure (Issue #60)",
            "from kosmos.analysis.visualization import PublicationVisualizer",
            "viz = PublicationVisualizer()",
            "",
            "# Generate scatter plot with regression if figure_path is provided",
            "if 'figure_path' in dir() and figure_path:",
            "    viz.scatter_with_regression(",
            f"        x=df['{x_var}'].values,",
            f"        y=df['{y_var}'].values,",
            f"        x_label='{x_var}',",
            f"        y_label='{y_var}',",
            f"        title={protocol.name!r},",
            "        output_path=str(figure_path)",
            "    )",
            "    result['figure_path'] = str(figure_path)",
            "",
            "# Propagate data source and assumption checks into results",
            "if '_data_source' in dir():",
            "    result['data_source'] = _data_source",
            "result['assumption_checks'] = {",
            "    'normality_tested': True,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Return results",
            "results = result"
        ]

        return "\n".join(code_lines)


class LogLogScalingCodeTemplate(CodeTemplate):
    """
    Template for log-log scaling analysis.

    Pattern from: kosmos-figures Figure_4_neural_network
    """

    def __init__(self):
        super().__init__("log_log_scaling", ExperimentType.DATA_ANALYSIS)
        self.handled_tests = frozenset({"correlation", "regression"})

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol needs log-log scaling analysis."""
        # Check for keywords in name or description
        keywords = ['scaling', 'power law', 'log-log', 'power-law']

        text = f"{protocol.name} {protocol.description}".lower()

        return any(keyword in text for keyword in keywords)

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate log-log scaling analysis code."""
        vars_list = list(protocol.variables.keys())
        x_var = vars_list[0] if len(vars_list) > 0 else 'x'
        y_var = vars_list[1] if len(vars_list) > 1 else 'y'

        seed = getattr(protocol, 'random_seed', 42) or 42

        code_lines = [
            "# Log-Log Scaling Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from pathlib import Path",
            "from kosmos.execution.data_analysis import DataAnalyzer, DataCleaner",
            "",
            "# Data loading (real dataset only - no synthetic fallback)",
            f"# Expected format: CSV with columns '{x_var}' and '{y_var}'",
            "df = None",
            "if 'data_path' in dir() and data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        raise RuntimeError(f'Failed to load required dataset from {data_path}: {e}. Real data only; no synthetic fallback.')",
            "if df is None:",
            "    raise RuntimeError('No dataset available at data_path; refusing to fabricate synthetic data (real data only).')",
            "",
            "# Clean data - remove NaN and non-positive values (required for log-log)",
            "df = DataCleaner.filter_positive(df, ['" + x_var + "', '" + y_var + "'])",
            "",
            "# Perform log-log scaling analysis",
            "analyzer = DataAnalyzer()",
            f"result = analyzer.log_log_scaling_analysis(df, '{x_var}', '{y_var}')",
            "",
            "# Print results",
            "print(f\"Spearman correlation: {result['spearman_rho']:.4f}\")",
            "print(f\"P-value: {result['p_value']:.6f}\")",
            "print(f\"Power law equation: {result['equation']}\")",
            "print(f\"Exponent: {result['power_law_exponent']:.4f}\")",
            "print(f\"R-squared: {result['r_squared']:.4f}\")",
            "",
            "# Generate publication-quality figure (Issue #60)",
            "from kosmos.analysis.visualization import PublicationVisualizer",
            "viz = PublicationVisualizer()",
            "",
            "# Generate log-log plot if figure_path is provided (600 DPI for panels)",
            "if 'figure_path' in dir() and figure_path:",
            "    viz.log_log_plot(",
            f"        x=df['{x_var}'].values,",
            f"        y=df['{y_var}'].values,",
            f"        x_label='{x_var}',",
            f"        y_label='{y_var}',",
            f"        title={protocol.name!r},",
            "        output_path=str(figure_path)",
            "    )",
            "    result['figure_path'] = str(figure_path)",
            "",
            "# Propagate data source and assumption checks into results",
            "if '_data_source' in dir():",
            "    result['data_source'] = _data_source",
            "result['assumption_checks'] = {",
            "    'normality_tested': False,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Return results",
            "results = result"
        ]

        return "\n".join(code_lines)


class MLExperimentCodeTemplate(CodeTemplate):
    """Template for machine learning experiments."""

    def __init__(self):
        super().__init__("ml_experiment", ExperimentType.COMPUTATIONAL)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol is ML experiment."""
        keywords = ['machine learning', 'classification', 'cross-validation',
                     'random forest', 'neural network', 'logistic regression',
                     'decision tree', 'svm', 'support vector']

        text = f"{protocol.name} {protocol.description}".lower()

        return any(keyword in text for keyword in keywords)

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate ML experiment code."""
        code_lines = [
            "# Machine Learning Experiment",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.datasets import make_classification",
            "from pathlib import Path",
            "from kosmos.execution.ml_experiments import MLAnalyzer",
            "",
            "# Data loading (real dataset only - no synthetic fallback)",
            "df = None",
            "if 'data_path' in dir() and data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        raise RuntimeError(f'Failed to load required dataset from {data_path}: {e}. Real data only; no synthetic fallback.')",
            "if df is None:",
            "    raise RuntimeError('No dataset available at data_path; refusing to fabricate synthetic data (real data only).')",
            "",
            "# Prepare features and target",
            "# Assuming last column is target",
            "X = df.iloc[:, :-1]",
            "y = df.iloc[:, -1]",
            "",
            "# Initialize ML analyzer",
            "analyzer = MLAnalyzer(random_state=42)",
            "",
            "# Run complete experiment with cross-validation",
            "model = LogisticRegression(max_iter=1000)",
            "results = analyzer.run_experiment(",
            "    model, X, y,",
            "    test_size=0.2,",
            "    cv=5,",
            "    task_type='classification',",
            "    scale_features=True",
            ")",
            "",
            "# Print results",
            "print(f\"Test Accuracy: {results['train_test_results']['accuracy']:.4f}\")",
            "print(f\"CV Mean Score: {results['cv_results']['mean_score']:.4f}\")",
            "print(f\"F1 Score: {results['train_test_results']['f1_score']:.4f}\")",
            "",
            "# Generate publication-quality figure (Issue #60)",
            "from kosmos.analysis.visualization import PublicationVisualizer",
            "viz = PublicationVisualizer()",
            "",
            "# Generate predicted vs actual scatter plot if figure_path is provided",
            "if 'figure_path' in dir() and figure_path and 'y_test' in results.get('train_test_results', {}):",
            "    y_test = results['train_test_results'].get('y_test', [])",
            "    y_pred = results['train_test_results'].get('y_pred', [])",
            "    if len(y_test) > 0 and len(y_pred) > 0:",
            "        viz.scatter_with_regression(",
            "            x=np.array(y_test),",
            "            y=np.array(y_pred),",
            "            x_label='Actual',",
            "            y_label='Predicted',",
            f"            title={protocol.name!r},",
            "            output_path=str(figure_path)",
            "        )",
            "        results['figure_path'] = str(figure_path)",
            "",
            "# Propagate data source and assumption checks into results",
            "if '_data_source' in dir():",
            "    results['data_source'] = _data_source",
            "results['assumption_checks'] = {",
            "    'normality_tested': False,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Return results",
            "results = results"
        ]

        return "\n".join(code_lines)


class MendelianRandomizationCodeTemplate(CodeTemplate):
    """
    General template for inverse-variance-weighted (IVW) Mendelian randomization.

    Dataset-agnostic: it matches any protocol whose text signals an MR / IVW /
    instrumental-variable analysis, and the generated code auto-detects the four
    required per-instrument columns (exposure & outcome effect sizes and their
    standard errors) by flexible name matching. It is NOT tied to any specific
    dataset, protein, or study. Input is expected to be MR-ready (one row per
    genetic instrument, exposure & outcome betas harmonized to the same effect
    allele) as produced by standard MR harmonization.
    """

    def __init__(self):
        super().__init__("mendelian_randomization_ivw", ExperimentType.COMPUTATIONAL)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Trigger on MR signals anywhere in the protocol text (any experiment type)."""
        parts = [
            protocol.name or "",
            getattr(protocol, "description", "") or "",
            getattr(protocol, "objective", "") or "",
        ]
        for step in getattr(protocol, "steps", []) or []:
            parts += [getattr(step, "title", "") or "",
                      getattr(step, "action", "") or "",
                      getattr(step, "description", "") or ""]
        for test in getattr(protocol, "statistical_tests", []) or []:
            tt = test.test_type.value if hasattr(test.test_type, "value") else str(test.test_type)
            parts.append(tt or "")
        hay = " ".join(parts).lower()
        signals = ("mendelian", "ivw", "inverse-variance", "inverse variance",
                   "instrumental variable")
        return any(s in hay for s in signals)

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate general IVW-MR code (auto-detects columns; no synthetic fallback)."""
        # NOTE: a normal (non-f) triple-quoted string so the sandbox f-string
        # braces below are preserved literally.
        return '''# Mendelian Randomization (IVW) Analysis
# General template: works for any MR-ready dataset with per-instrument exposure
# and outcome effect sizes. The four required columns are auto-detected.
import pandas as pd
import numpy as np
from scipy import stats

# --- Data loading (real dataset only - no synthetic fallback) ---
df = None
if 'data_path' in dir() and data_path:
    try:
        df = pd.read_csv(data_path)
        _data_source = 'file'
    except Exception as e:
        raise RuntimeError(f'Failed to load required dataset from {data_path}: {e}. Real data only; no synthetic fallback.')
if df is None:
    raise RuntimeError('No dataset available at data_path; refusing to fabricate synthetic data (real data only).')

# --- Auto-detect the four MR columns (exposure/outcome effect sizes + SEs) ---
def _find_col(cols, aliases, tok_groups):
    low = {c.lower().strip(): c for c in cols}
    for a in aliases:
        if a in low:
            return low[a]
    for c in cols:
        cl = c.lower()
        if all(any(t in cl for t in grp) for grp in tok_groups):
            return c
    return None

_cols = list(df.columns)
_bx = _find_col(_cols, ['beta_exposure','beta.exposure','beta_exp','effect_exposure','bx','beta_x'],
                [['beta','effect'], ['exposure','exp','_x']])
_sx = _find_col(_cols, ['se_exposure','se.exposure','se_exp','sebeta_exposure','stderr_exposure','se_x'],
                [['se','std','stderr','standard'], ['exposure','exp','_x']])
_by = _find_col(_cols, ['beta_outcome','beta.outcome','beta_out','effect_outcome','by','beta_y'],
                [['beta','effect'], ['outcome','out','_y']])
_sy = _find_col(_cols, ['se_outcome','se.outcome','se_out','sebeta_outcome','stderr_outcome','se_y'],
                [['se','std','stderr','standard'], ['outcome','out','_y']])

_missing = [n for n, v in [('beta_exposure', _bx), ('se_exposure', _sx),
                           ('beta_outcome', _by), ('se_outcome', _sy)] if v is None]
if _missing:
    raise RuntimeError('Mendelian randomization requires per-instrument exposure/outcome '
                       'effect sizes. Missing column(s) for: ' + ', '.join(_missing) +
                       '. Columns present: ' + ', '.join(_cols))

# --- Assemble and clean instruments ---
bx = pd.to_numeric(df[_bx], errors='coerce').values.astype(float)
sx = pd.to_numeric(df[_sx], errors='coerce').values.astype(float)
by = pd.to_numeric(df[_by], errors='coerce').values.astype(float)
sy = pd.to_numeric(df[_sy], errors='coerce').values.astype(float)
_ok = np.isfinite(bx) & np.isfinite(sx) & np.isfinite(by) & np.isfinite(sy) & (sy > 0) & (bx != 0)
bx, sx, by, sy = bx[_ok], sx[_ok], by[_ok], sy[_ok]
n = int(len(bx))
if n < 1:
    raise RuntimeError('No valid MR instruments after cleaning (need finite bx/sx/by/sy, sy>0, bx!=0).')

# --- Inverse-variance-weighted (IVW) MR, fixed effect ---
w = 1.0 / (sy ** 2)
_denom = float(np.sum(w * bx ** 2))
ivw_beta = float(np.sum(w * bx * by) / _denom)
ivw_se = float(np.sqrt(1.0 / _denom))
ivw_z = ivw_beta / ivw_se
ivw_p = float(2 * stats.norm.sf(abs(ivw_z)))

# --- Per-instrument Wald ratios ---
wald = by / bx
wald_ratios = [float(x) for x in wald]

results = {
    'method': 'IVW Mendelian randomization (fixed-effect)',
    'n_instruments': n,
    'ivw_beta': ivw_beta,
    'ivw_se': ivw_se,
    'ivw_p_value': ivw_p,
    'effect_size': ivw_beta,
    'p_value': ivw_p,
    'supports_hypothesis': bool(ivw_p < 0.05),
    'direction': ('inverse (higher exposure -> lower outcome)' if ivw_beta < 0
                  else 'positive (higher exposure -> higher outcome)'),
    'columns_used': {'beta_exposure': _bx, 'se_exposure': _sx,
                     'beta_outcome': _by, 'se_outcome': _sy},
    'wald_ratios': wald_ratios,
    'data_source': _data_source,
}

# --- Cochran's Q heterogeneity (>1 instrument) ---
if n > 1:
    _wq = bx ** 2 / (sy ** 2)
    _Q = float(np.sum(_wq * (wald - ivw_beta) ** 2))
    results['heterogeneity_Q'] = _Q
    results['heterogeneity_p'] = float(stats.chi2.sf(_Q, n - 1))

# --- MR-Egger (>=3 instruments): intercept tests directional pleiotropy ---
if n >= 3:
    try:
        Sw = float(np.sum(w)); Swx = float(np.sum(w * bx)); Swy = float(np.sum(w * by))
        Swxx = float(np.sum(w * bx * bx)); Swxy = float(np.sum(w * bx * by))
        _d = Sw * Swxx - Swx * Swx
        if _d != 0:
            results['egger_slope'] = float((Sw * Swxy - Swx * Swy) / _d)
            results['egger_intercept'] = float((Swxx * Swy - Swx * Swxy) / _d)
    except Exception:
        pass

# --- Report ---
print(f"MR (IVW): {n} instrument(s); columns exposure=({_bx},{_sx}) outcome=({_by},{_sy})")
print(f"IVW beta = {ivw_beta:.4f}  SE = {ivw_se:.4f}  p = {ivw_p:.3e}")
print(f"Direction: {results['direction']}")
if n == 1:
    print("Note: single instrument -> IVW equals the Wald ratio.")
'''


class GenericComputationalCodeTemplate(CodeTemplate):
    """
    Generic template for computational experiments (biology, chemistry, etc.).

    Acts as a catch-all when no other template matches. Generates scipy-based
    analysis code with data loading, statistical tests, curve fitting, and
    visualization.
    """

    def __init__(self):
        super().__init__("generic_computational", ExperimentType.COMPUTATIONAL)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Match COMPUTATIONAL or DATA_ANALYSIS experiments (catch-all fallback)."""
        return protocol.experiment_type in (
            ExperimentType.COMPUTATIONAL,
            ExperimentType.DATA_ANALYSIS,
        )

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate generic computational analysis code."""
        vars_list = list(protocol.variables.keys())
        x_var = vars_list[0] if len(vars_list) > 0 else 'x'
        y_var = vars_list[1] if len(vars_list) > 1 else 'y'

        seed = getattr(protocol, 'random_seed', 42) or 42

        # Determine statistical tests from protocol
        stat_tests = []
        for test in protocol.statistical_tests:
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            stat_tests.append(test_type_str.lower())

        code_lines = [
            "# Computational Experiment Analysis",
            f"# Protocol: {protocol.name}",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from scipy.optimize import curve_fit",
            "from pathlib import Path",
            "",
            "# Data loading (real dataset only - no synthetic fallback)",
            f"# Expected format: CSV with relevant columns",
            "df = None",
            "if 'data_path' in dir() and data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        raise RuntimeError(f'Failed to load required dataset from {data_path}: {e}. Real data only; no synthetic fallback.')",
            "if df is None:",
            "    raise RuntimeError('No dataset available at data_path; refusing to fabricate synthetic data (real data only).')",
            "",
            "# Clean data",
            *self.clean_data_lines(),
            f"print(f'Loaded {{len(df)}} samples (source: {{_data_source}})')",
            "",
            "# Statistical analysis",
            "results = {}",
            "results['n_samples'] = len(df)",
            "results['data_source'] = _data_source",
            "",
            "# Descriptive statistics",
            "results['descriptive'] = {}",
            "for col in df.select_dtypes(include=[np.number]).columns:",
            "    results['descriptive'][col] = {",
            "        'mean': float(df[col].mean()),",
            "        'std': float(df[col].std()),",
            "        'median': float(df[col].median()),",
            "        'min': float(df[col].min()),",
            "        'max': float(df[col].max()),",
            "    }",
            "",
            "# Primary statistical test",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()",
            "# Correlation needs at least 3 points to mean anything, and scipy",
            "# raises an opaque ValueError below 2. Fail loudly here instead.",
            "if len(numeric_cols) >= 2 and len(df) < 3:",
            "    raise RuntimeError(",
            "        f'Only {len(df)} row(s) survive cleaning; correlation needs >= 3.')",
            "if len(numeric_cols) >= 2 and len(df) >= 3:",
            "    col_x = numeric_cols[0]",
            "    col_y = numeric_cols[1]",
            "    x_vals = df[col_x].values",
            "    y_vals = df[col_y].values",
            "",
            "    # Normality check",
            "    for _col, _vals in [(col_x, x_vals), (col_y, y_vals)]:",
            "        if len(_vals) >= 8:",
            "            _shap_stat, _shap_p = stats.shapiro(_vals[:5000])",
            "            if _shap_p < 0.05:",
            "                print(f'WARNING: Normality assumption violated for {_col} (Shapiro p={_shap_p:.4f})')",
            "",
            "    # Correlation analysis",
            "    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)",
            "    spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)",
            "    results['correlation'] = {",
            "        'pearson_r': float(pearson_r),",
            "        'pearson_p': float(pearson_p),",
            "        'spearman_r': float(spearman_r),",
            "        'spearman_p': float(spearman_p),",
            "    }",
            "    results['p_value'] = float(pearson_p)",
            "    results['effect_size'] = float(pearson_r)",
            "",
            "    # Nonlinear curve fitting (exponential decay model)",
            "    try:",
            "        def exp_model(x, a, b, c):",
            "            return a * np.exp(b * x) + c",
            "        popt, pcov = curve_fit(exp_model, x_vals, y_vals, p0=[1, -0.1, 0], maxfev=5000)",
            "        y_fit = exp_model(x_vals, *popt)",
            "        ss_res = np.sum((y_vals - y_fit) ** 2)",
            "        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)",
            "        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0",
            "        results['curve_fit'] = {",
            "            'model': 'exponential',",
            "            'params': {'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2])},",
            "            'r_squared': float(r_squared),",
            "        }",
            "    except Exception as e:",
            "        print(f'Curve fitting failed: {e}')",
            "        results['curve_fit'] = {'model': 'exponential', 'error': str(e)}",
            "",
            "    # Linear regression as baseline",
            "    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)",
            "    results['linear_regression'] = {",
            "        'slope': float(slope),",
            "        'intercept': float(intercept),",
            "        'r_squared': float(r_value ** 2),",
            "        'p_value': float(p_value),",
            "        'std_err': float(std_err),",
            "    }",
            "",
            "elif len(numeric_cols) == 1:",
            "    # Single variable: one-sample t-test against zero",
            "    col = numeric_cols[0]",
            "    vals = df[col].values",
            "    t_stat, p_value = stats.ttest_1samp(vals, 0)",
            "    results['one_sample_ttest'] = {",
            "        't_statistic': float(t_stat),",
            "        'p_value': float(p_value),",
            "        'mean': float(np.mean(vals)),",
            "        'std': float(np.std(vals)),",
            "    }",
            "    results['p_value'] = float(p_value)",
            "    results['effect_size'] = float(np.mean(vals) / np.std(vals)) if np.std(vals) > 0 else 0.0",
            "",
            "# Generate publication-quality figure",
            "from kosmos.analysis.visualization import PublicationVisualizer",
            "viz = PublicationVisualizer()",
            "",
            "if 'figure_path' in dir() and figure_path and len(numeric_cols) >= 2:",
            "    viz.scatter_with_regression(",
            "        x=df[numeric_cols[0]].values,",
            "        y=df[numeric_cols[1]].values,",
            "        x_label=numeric_cols[0],",
            "        y_label=numeric_cols[1],",
            f"        title={protocol.name!r},",
            "        output_path=str(figure_path)",
            "    )",
            "    results['figure_path'] = str(figure_path)",
            "",
            "# Assumption checks",
            "results['assumption_checks'] = {",
            "    'normality_tested': True,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Print summary",
            "print(f'Analysis complete: {len(results)} result keys')",
            "if 'p_value' in results:",
            "    print(f'Primary p-value: {results[\"p_value\"]:.6f}')",
            "if 'effect_size' in results:",
            "    print(f'Effect size: {results[\"effect_size\"]:.4f}')",
        ]

        return "\n".join(code_lines)


# Evidence that generated code actually reaches its data. `data_path`,
# `datasets` and `__data_files__` are injected by the executor; the reader
# calls are the ordinary vocabulary of opening a table, kept deliberately broad
# because the check exists to catch code that opens NOTHING, not to police how.
# Completion budget for code generation specifically. The client default is
# sized for prose answers; a 10-step protocol does not fit in it, and what
# arrives is a script cut off mid-statement -- "unterminated string literal
# (detected at line 54)". That reads like a model that cannot write Python,
# when it is a model that ran out of room.
_CODEGEN_MAX_TOKENS = int(os.environ.get("KOSMOS_CODEGEN_MAX_TOKENS", "16384"))

# Syntax errors that mean "the response stopped early" rather than "the model
# wrote invalid Python". Only these are worth a retry: a genuine syntax error
# reproduces, and retrying it just spends another call.
_TRUNCATION_MARKERS = (
    "unterminated string literal",
    "unterminated triple-quoted string literal",
    "unexpected eof while parsing",
    "was never closed",
)


# Errors that mean "your filter matched nothing", whatever they say on the
# surface. sklearn reports an empty frame as a train/test split problem, pandas
# as a KeyError on an aggregation -- neither names the cause, so the model reads
# the message and adjusts the split instead of the filter.
_EMPTY_RESULT_MARKERS = (
    "n_samples=0",
    "with n_samples=0",
    "empty dataframe",
    "no rows",
    "resulting train set will be empty",
    "zero-size array",
    "cannot reshape array of size 0",
)


def _looks_like_an_empty_frame(error_text: str) -> bool:
    text = (error_text or "").lower()
    return any(marker in text for marker in _EMPTY_RESULT_MARKERS)


def _empty_frame_advice() -> str:
    """What an empty frame almost always means here.

    Observed three times on one dataset: the model wrote
    `df['readmitted'].map({'<30': 1, 'NO': 0, '>30': 0})` -- the encoding the
    ORIGINAL UCI release uses -- against a column this copy stores as integers
    0/1. Every row became NaN, the frame emptied, and sklearn reported
    `With n_samples=0 ... the resulting train set will be empty`. Nothing in
    that message points at the mapping, so the next draft repeated it.
    """
    return (
        " An empty frame after filtering or mapping almost always means a "
        "VALUE you compared against does not occur in this data -- not that "
        "the split or the sample size is wrong. Re-read the value sets listed "
        "in the dataset summary and use those exact values; a dataset you "
        "recognise may be encoded differently here (integers 0/1 where the "
        "original release used strings like '<30'). Print the value counts of "
        "any column you filter on before filtering."
    )


def _looks_truncated(error: Exception) -> bool:
    """Did generation stop mid-statement?"""
    text = str(error).lower()
    return any(marker in text for marker in _TRUNCATION_MARKERS)


_DATA_ACCESS_NAMES = frozenset({"data_path", "datasets", "__data_files__"})
_DATA_READER_CALLS = frozenset({
    "open", "connect", "execute", "load", "loads", "loadtxt", "loadmat",
    "load_npz", "read_h5ad", "read_hdf", "from_csv", "scan_csv",
    "scan_parquet", "get_dataset", "load_dataset",
})


def _called_name(node: ast.Call) -> str:
    """The bare function name of a call: `pd.read_csv(...)` -> `read_csv`."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _reads_data(tree: ast.AST) -> bool:
    """Does this code reference an injected dataset or call a reader?"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _DATA_ACCESS_NAMES:
            return True
        if isinstance(node, ast.Call):
            name = _called_name(node)
            if name.startswith("read_") or name in _DATA_READER_CALLS:
                return True
    return False


# Injected into the sandbox globals by the executor, so they are defined at run
# time without appearing anywhere in the script.
_INJECTED_NAMES = frozenset({
    "data_path", "datasets", "__data_files__", "__name__", "__file__", "__doc__",
    # Supplied by the figure manager when an experiment saves a plot. The
    # shipped templates guard it with `if 'figure_path' in dir():`, so it is
    # legitimately used without ever being assigned in the script.
    "figure_path",
})

_UNDEFINED_MARKER = "calls undefined name"


# What the sandbox image actually has installed, from
# `docker/sandbox/requirements.txt`. The model was left to guess and guessed
# wrong: `from scipy.stats import multipletests` -- a real function, in
# statsmodels, which IS installed. An import error kills the script on line 61
# before any analysis runs, so naming the package set is cheap next to what it
# prevents.
SANDBOX_PACKAGES = "biopython, gseapy, h5py, jupyter_client, matplotlib, nbconvert, nbformat, networkx, numpy, openpyxl, pandas, plotly, pwlf, pyarrow, pydantic, python-dateutil, pytz, scikit-learn, scipy, seaborn, shap, statsmodels, sympy, xlrd"


def _environment_text() -> str:
    """The prompt section stating what may be imported."""
    return (
        "AVAILABLE PACKAGES (the sandbox has these and nothing else): "
        f"{SANDBOX_PACKAGES}. Import only from these. Multiple-testing "
        "correction lives in statsmodels (statsmodels.stats.multitest."
        "multipletests), NOT in scipy.stats. There is no colocalization or "
        "Mendelian-randomization package -- implement those from the summary "
        "statistics directly with numpy/scipy. The container has NO network "
        "access, so nothing can be downloaded or pip-installed at runtime. Its "
        "FILESYSTEM IS READ-ONLY except /tmp: `df.to_csv('variant_set.csv')` "
        "raises OSError [Errno 30] and kills the script. Keep intermediates in "
        "memory; if you genuinely must write a file, write it under /tmp. The "
        "script is run as `python3 experiment.py` with NO command-line "
        "arguments and no environment variables: `sys.argv` has length 1, so "
        "an argv-based entrypoint always takes its else-branch. Do not write "
        "one -- read the names the executor already defined. "
        # A run that had finished its statistics lost every one of them to a
        # boxplot: the analysis computed probe rankings and an ECM score, then
        # called seaborn on an all-NaN column and died with
        # `UnboundLocalError: boxprops` (a real seaborn 0.13.2 bug, raised
        # before anything was returned). Figures are decoration; the numbers
        # are the experiment. Ordering them the other way makes every plotting
        # library defect a total loss of the science.
        "FIGURES ARE OPTIONAL AND MUST NEVER VOID THE ANALYSIS: assign the "
        "complete `results` dict at module level BEFORE drawing anything, and "
        "wrap every plotting call in `try/except Exception` so a figure that "
        "fails cannot discard numbers you already computed. Drop all-NaN "
        "columns before plotting -- seaborn raises an obscure UnboundLocalError "
        "on them rather than an empty-data message."
    )


def _analyzer_methods() -> list[str]:
    """The methods `DataAnalyzer` actually has, read off the class."""
    try:
        from kosmos.execution.data_analysis import DataAnalyzer
    except Exception:
        return []
    return sorted(m for m in dir(DataAnalyzer) if not m.startswith("_"))


def _analyzer_api_text() -> str:
    """The prompt section listing what `analyzer` can be asked to do.

    The prompt used to say only "Use DataAnalyzer for statistical tests",
    naming a class without naming its API. A model told to use an object it
    cannot inspect invents plausible methods on it: `analyzer.chi_square(table)`
    reached the sandbox and raised `AttributeError: 'DataAnalyzer' object has no
    attribute 'chi_square'`, after which the executor retried the identical code
    three times.

    Generated from the class rather than written out, so the list cannot drift
    away from the object the sandbox imports -- a stale hand-written list would
    reintroduce exactly this bug.
    """
    methods = _analyzer_methods()
    if not methods:
        return "Use scipy.stats and numpy for statistical tests"
    listed = "\n".join(f"  - analyzer.{m}(...)" for m in methods)
    return (
        "Use kosmos.execution.data_analysis.DataAnalyzer for statistical tests. "
        "It has EXACTLY these methods and no others:\n"
        f"{listed}\n"
        "For any test NOT in that list -- chi-square, Fisher, binomial, "
        "colocalization, Mendelian randomization, permutation -- use "
        "scipy.stats/numpy directly. Calling a method DataAnalyzer does not "
        "have raises AttributeError and the experiment produces nothing."
    )


def _invented_analyzer_calls(tree: ast.AST) -> list[str]:
    """Methods called on a `DataAnalyzer()` instance that do not exist.

    Narrow on purpose: it only follows names assigned directly from a
    `DataAnalyzer()` construction, so it cannot misjudge an attribute on some
    other object. That is enough to catch the observed failure while leaving
    every other attribute access alone.
    """
    real = set(_analyzer_methods())
    if not real:
        return []

    instances = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "DataAnalyzer"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if not instances:
        return []

    return sorted({
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in instances
        and node.attr not in real
        and not node.attr.startswith("_")
    })


def _clobbers_injected_data(tree: ast.AST) -> list[str]:
    """Assignments to the executor's injected names, and reads of `sys.argv`.

    The executor writes `data_path = '<a path inside the container>'` (and
    `datasets = {...}`) on the line above the model's first, so generated code
    that assigns either one is overwriting a real path with something else, and
    what a model reaches for when it does is the thing that is not there. The
    myocardial-fibrosis run on GSE2240 died at `ValueError: Expected data_path
    argument` from an argv entrypoint the model wrote itself: the sandbox runs
    `python3 experiment.py` with no arguments, so that branch is taken every
    time, while the injected path sits in scope unread for all 364 lines.

    A gate rather than a warning because neither pattern has a form that works
    inside this sandbox -- there is no argv to read and no reason to rebind a
    name that is already correct. Function PARAMETERS named `data_path` are
    untouched: `def load(data_path)` called as `load(data_path)` is the good
    pattern, not the bad one.
    """
    offences: list[str] = []
    injected = {"data_path", "datasets"}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in injected:
                offences.append(f"assigns `{target.id}`")
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "argv"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            offences.append("reads `sys.argv`")
    return sorted(set(offences))


def _undefined_names(tree: ast.AST) -> list[str]:
    """Names the script reads but never binds, imports or receives.

    Deliberately whole-module rather than per-scope: a name bound ANYWHERE
    counts as defined. That under-reports (a local used in the wrong function
    slips through) and it is the right trade -- a false positive here throws
    away a correct multi-dataset analysis, while the error this catches is a
    guaranteed crash. `NameError: name '_validate_cols' is not defined. Did you
    mean: '_validate_columns'?` reached the sandbox and was retried three times
    against the same code, which no amount of re-execution could fix.
    """
    bound: set[str] = set()
    loaded: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else loaded).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # `ast.Lambda` belongs here and was missed: a lambda's parameters
            # are bound by its `args` exactly as a def's are, but a Lambda is
            # not a FunctionDef. Leaving it out flagged the parameter of every
            # `lambda r: ...` and `key=lambda x: x[1]` as undefined -- and
            # single-letter lambda parameters are ubiquitous in analysis code,
            # so the check rejected working scripts and cost a whole
            # multi-dataset run its analysis.
            if not isinstance(node, ast.Lambda):
                bound.add(node.name)
            a = node.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                if arg is not None:
                    bound.add(arg.arg)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    # A star import defines an unknowable set of names, so the
                    # check cannot be sound. Skip rather than guess.
                    return []
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)

    known = bound | _INJECTED_NAMES | set(dir(builtins))
    return sorted(loaded - known)


def _module_level_bindings(tree: ast.AST) -> set:
    """Names bound in MODULE scope, where the sandbox's capture can see them.

    Descends through `if`/`try`/`with`/`for` at the top level -- those execute
    in module scope -- but never into a function, lambda or class body, whose
    locals are gone by the time the capture reads `globals()`.
    """
    bound: set = set()

    def _targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            return [node.target]
        return []

    def _visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Lambda)):
                continue
            for target in _targets(child):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
                elif (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                ):
                    bound.add(target.value.id)
            _visit(child)

    _visit(tree)
    return bound


def _binds_name(tree: ast.AST, name: str) -> bool:
    """Is `name` assigned anywhere -- plainly, annotated, augmented, or by key?

    Assignment inside a function counts. Code that builds `results` in a
    `main()` and calls it is a real experiment; only code that never mentions
    the name as a target is not.
    """
    def _targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            return [node.target]
        return []

    for node in ast.walk(tree):
        for target in _targets(node):
            if isinstance(target, ast.Name) and target.id == name:
                return True
            # `results['x'] = ...` binds nothing new, but it can only appear in
            # code that already built `results`, so it is equally good evidence.
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
            ):
                return True
    return False


class ExperimentCodeGenerator:
    """
    Generates executable Python code from experiment protocols.

    Uses hybrid approach:
    1. Template matching for common patterns
    2. LLM generation for novel experiments
    3. Optional LLM enhancement of templates
    """

    def __init__(
        self,
        use_templates: bool = True,
        use_llm: bool = True,
        llm_enhance_templates: bool = False,
        llm_client: Optional[ClaudeClient] = None
    ):
        """
        Initialize code generator.

        Args:
            use_templates: If True, try template matching first
            use_llm: If True, use LLM for novel cases or fallback
            llm_enhance_templates: If True, enhance template code with LLM
            llm_client: Optional Claude client (created if not provided)
        """
        self.use_templates = use_templates
        self.use_llm = use_llm
        self.llm_enhance_templates = llm_enhance_templates
        # Datasets this experiment may open: {dataset_name: host path}. Empty or
        # one entry reproduces today's behaviour exactly; more than one switches
        # on the multi-table path in `generate()` (Step 0 there explains why it
        # has to come before template matching).
        self.datasets: Dict[str, str] = {}
        # Per-dataset column and shape descriptions, so the model writes code
        # against real column names rather than guessing them.
        self.dataset_context: Optional[str] = None

        # Initialize LLM client with error handling. Prefer the configured
        # provider system (LLM_PROVIDER, e.g. OpenRouter/deepseek) so code
        # generation uses the same model as the rest of the pipeline instead of
        # hardcoding Claude (which needs ANTHROPIC_API_KEY and otherwise dies).
        if use_llm and llm_client is None:
            try:
                from kosmos.core.llm import get_client
                self.llm_client = get_client(use_provider_system=True)
            except Exception as e:
                logger.warning(f"Provider-system client failed: {e}. Trying ClaudeClient.")
                try:
                    self.llm_client = ClaudeClient()
                except Exception as e2:
                    logger.warning(f"ClaudeClient also failed: {e2}. LLM generation disabled.")
                    self.llm_client = None
                    self.use_llm = False
        else:
            self.llm_client = llm_client if use_llm else None

        # Initialize templates
        self.templates: List[CodeTemplate] = []
        if use_templates:
            self._register_templates()

    def _register_templates(self):
        """Register all available code templates."""
        self.templates = [
            TTestComparisonCodeTemplate(),
            # MR/IVW is a specific method; match it before the correlation/regression
            # and generic catch-all templates so an MR protocol that happens to
            # mention "regression" isn't grabbed by CorrelationAnalysis first.
            MendelianRandomizationCodeTemplate(),
            CorrelationAnalysisCodeTemplate(),
            LogLogScalingCodeTemplate(),
            MLExperimentCodeTemplate(),
            GenericComputationalCodeTemplate(),  # Catch-all for COMPUTATIONAL + DATA_ANALYSIS
        ]

        logger.info(f"Registered {len(self.templates)} code templates")

    def generate(
        self,
        protocol: ExperimentProtocol,
        runtime_error: Optional[str] = None,
    ) -> str:
        """
        Generate code from protocol using hybrid approach.

        Args:
            protocol: Experiment protocol

        Returns:
            Generated Python code as string
        """
        code = None

        # Step 0: Multi-dataset experiments BYPASS template matching entirely.
        #
        # This has to come first, and the reason is not a preference. Every
        # shipped template loads exactly one table from `data_path`, and the
        # matcher is guaranteed to find one: `GenericComputationalCodeTemplate`
        # is registered last and matches any COMPUTATIONAL or DATA_ANALYSIS
        # protocol, which is what `_select_experiment_type` produces by default.
        # So `_match_template` never returns None in practice, `code` is never
        # None at Step 2, and `_generate_with_llm` is unreachable on this path.
        # Leaving multi-dataset generation to Step 2 would mean mounting N files
        # that nothing ever opens -- the experiment would silently analyse the
        # primary alone and report as though it had seen them all.
        # What actually produced the code, recorded for the caller. A fallback
        # narrows the analysis silently otherwise: the protocol still describes
        # the multi-dataset experiment that was DESIGNED, and a report built
        # from the protocol then narrates work that never ran.
        self.last_generation = {"path": None, "template": None, "fallback_reason": None}

        # A regeneration after a runtime failure ALSO bypasses the templates,
        # whatever the dataset count.
        #
        # Without this, a single-dataset retry was a no-op: Step 1 matches a
        # template, the template ignores `runtime_error` entirely, and it
        # re-emits byte-identical code -- which the caller then skips as
        # unchanged. So the run logged "regenerating once" and generated
        # nothing. Observed on the breast-cancer run: the ML template fed the
        # string label column to StandardScaler, `could not convert string to
        # float: 'B'`, and the second draft that would have fixed it never
        # happened. A template that just failed at runtime cannot be the
        # remedy for its own failure; only the model, shown the traceback, can
        # produce something different.
        _bypass_templates = (
            (len(self.datasets) > 1 or runtime_error is not None)
            and self.use_llm
            and self.llm_client
        )
        if _bypass_templates:
            logger.info(
                "Generating with the LLM, bypassing single-table templates "
                "(%s)",
                f"{len(self.datasets)} datasets" if len(self.datasets) > 1
                else "regenerating after a runtime failure",
            )
            try:
                # The generation call belongs INSIDE the retryable try, not
                # before it. Wrapping only the validation meant a provider
                # failure -- now raised as ValueError rather than returning None
                # -- flew straight past the retry to the fallback, which is the
                # exact case a second draw fixes.
                try:
                    code = self._generate_with_llm(
                        protocol,
                        datasets=self.datasets if len(self.datasets) > 1 else None,
                        fix_hint=runtime_error,
                    )
                    # A fragment that opens no data is a failed generation, and
                    # must fall back rather than execute.
                    self._validate_generated(code)
                except ValueError as first_error:
                    # One retry for ANY rejection.
                    #
                    # This used to exclude "real" syntax errors on the reasoning
                    # that they reproduce. That is wrong for a sampler: the
                    # model does not systematically write unbalanced parens, and
                    # `unmatched ')' at line 40` is a sampling accident a second
                    # draw is unlikely to repeat. Excluding it cost a whole
                    # multi-dataset MR, which fell back to a single-table
                    # template and reported a correlation between chromosome
                    # number and genomic position instead.
                    #
                    # The economics are lopsided in one direction: a retry costs
                    # one call, and not retrying costs the entire cross-dataset
                    # analysis the run exists to perform. So retry once,
                    # whatever the rejection, and fall back only if the second
                    # draw fails too.
                    truncated = _looks_truncated(first_error)
                    logger.warning(
                        "Generated code rejected (%s); retrying once", first_error,
                    )
                    code = self._generate_with_llm(
                        protocol,
                        datasets=self.datasets if len(self.datasets) > 1 else None,
                        brief=truncated,
                        fix_hint=None if truncated else str(first_error),
                    )
                    self._validate_generated(code)
                # A second failure of any kind falls through to the outer
                # handler and the single-table fallback: one retry, not a loop.
                self.last_generation["path"] = "multi_dataset_llm"
            except Exception as e:
                self.last_generation["fallback_reason"] = (
                    f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                )
                # Falling back is better than failing the experiment: the
                # single-table path still produces a valid analysis of the
                # primary dataset. Logged loudly because the result answers a
                # narrower question than the protocol asked.
                logger.warning(
                    f"Multi-dataset code generation failed ({e}); falling back "
                    f"to a single-table template over the primary dataset only"
                )
                code = None

        # Step 1: Try template matching
        if code is None and self.use_templates:
            template = self._match_template(protocol)
            if template:
                logger.info(f"Using template: {template.name}")
                code = template.generate(protocol)
                self.last_generation["path"] = "template"
                self.last_generation["template"] = template.name

                # Optionally enhance with LLM
                if self.llm_enhance_templates and self.llm_client:
                    code = self._enhance_with_llm(code, protocol)

        # Step 2: Fall back to LLM generation
        if code is None and self.use_llm:
            logger.info("No template matched, using LLM generation")
            try:
                try:
                    code = self._generate_with_llm(protocol, fix_hint=runtime_error)
                    self._validate_generated(code)
                except ValueError as first_error:
                    # One retry, showing the model WHY it was rejected -- the
                    # same bargain the multi-dataset path already makes. This
                    # branch used to fall straight to the basic template on any
                    # rejection, so a single fixable defect (an argv entrypoint,
                    # one invented analyzer method) cost the whole generated
                    # analysis and replaced it with a summary. The rejection
                    # text is the one thing that makes the second draw
                    # different from the first; without it a retry is just
                    # another sample.
                    truncated = _looks_truncated(first_error)
                    logger.warning(
                        "Generated code rejected (%s); retrying once", first_error,
                    )
                    code = self._generate_with_llm(
                        protocol,
                        brief=truncated,
                        fix_hint=None if truncated else str(first_error),
                    )
                    self._validate_generated(code)
                self.last_generation["path"] = "llm"
            except Exception as e:
                logger.warning(
                    f"LLM generation produced nothing usable ({e}); "
                    f"falling back to the basic template"
                )
                code = None

        # Step 3: Fallback to basic template
        if code is None:
            logger.warning("No code generated, using basic template")
            code = self._generate_basic_template(protocol)
            self.last_generation["path"] = "basic_template"

        # Validate syntax
        self._validate_syntax(code)

        return code

    def _data_access_instructions(
        self, datasets: Optional[Dict[str, str]] = None
    ) -> str:
        """How the generated code should reach its data: one table, or several.

        Returned as one block so the two cases can never both appear in a
        prompt. The single-dataset text is byte-identical to what was inlined
        before, because every existing run takes that branch.
        """
        if not datasets or len(datasets) <= 1:
            # The file's real columns, when the caller knows them. Without this
            # the single-table prompt describes no schema at all, and a model
            # asked to analyse an unseen file invents one: a wide expression
            # matrix was read as a tidy `gene/sample/expression` table and the
            # container died on the generated code's own column check.
            details = (
                f"\n\n**Dataset details:**\n{self.dataset_context}\n"
                if self.dataset_context else ""
            )
            return details + """Generate complete, executable Python code that:
1. Loads data from the `data_path` variable (already defined by executor)
2. Implements each protocol step
3. Performs the specified statistical tests
4. Assigns a non-empty dict to `results` AT MODULE LEVEL (not only inside a
   function -- a local is gone before it can be collected). An empty `results`
   is the same as no output at all: if nothing survives a threshold, put the
   counts and a one-line explanation in `results` -- a null finding is a
   finding, silence is not.

IMPORTANT: Use `data_path` variable for loading data, e.g., `pd.read_csv(data_path)`
Do NOT hardcode 'data.csv' - use the data_path variable instead.

`data_path` is ALREADY ASSIGNED, on the line above your first one, to a path
that exists inside the container. Do not assign it yourself, do not guard it
with `if 'data_path' in globals()`, and do not look for it in `sys.argv` or
`os.environ` -- it is in scope from the first line to the last. Rebinding it
overwrites a real path with one that is not there.

CRITICAL - REAL DATA ONLY: Use ONLY the real dataset at `data_path`. NEVER generate,
fabricate, simulate, or synthesize data - no np.random data generation,
sklearn.datasets.make_*, synthetic/dummy/placeholder data, or synthetic fallbacks. If the
required columns or data are missing/insufficient, RAISE an exception (e.g.
`raise ValueError("required column X not found in dataset")`). Do NOT substitute any
made-up data under any circumstances."""

        listing = "\n".join(f"  - {name!r}" for name in sorted(datasets))
        context = (
            f"\n\n**Dataset details:**\n{self.dataset_context}"
            if self.dataset_context else ""
        )
        return f"""**THIS EXPERIMENT HAS {len(datasets)} DATASETS.** A dict named `datasets` is
already defined by the executor, mapping each dataset name to its file path:
{listing}

Generate complete, executable Python code that:
1. Loads EACH dataset it needs with `pd.read_csv(datasets['<name>'])`
2. Implements each protocol step
3. Performs the specified statistical tests
4. Assigns a non-empty dict to `results` AT MODULE LEVEL (not inside a function;
   a local is gone before it can be collected). An empty `results` is the same
   as no output at all: if nothing survives a threshold or a join, put the
   counts and a one-line explanation in `results` -- a null finding is a
   finding, silence is not.

IMPORTANT: Load every dataset the protocol refers to, by name, from `datasets`.
`data_path` also exists but points at the PRIMARY dataset only -- prefer
`datasets[...]` so it is explicit which table each step reads. Do NOT hardcode
any file path.

`datasets` and `data_path` are ALREADY ASSIGNED above your first line. Do not
assign either yourself, and do not look for them in `sys.argv` or `os.environ`
-- the script receives no arguments.

HOW THESE DATASETS MAY BE COMBINED: compare or correlate FINDINGS across them --
an effect estimated separately in each and then compared, or a per-feature
statistic joined on a shared FEATURE identifier (a gene, a variant, a
transcript, a cell line). Join only on such feature identifiers. Do NOT link
rows representing individual subjects between datasets, and do not build a
combined per-subject table. If a protocol step appears to require linking
individuals, RAISE an exception saying so rather than approximating it.

CRITICAL - REAL DATA ONLY: Use ONLY the real datasets named above. NEVER generate,
fabricate, simulate, or synthesize data - no np.random data generation,
sklearn.datasets.make_*, synthetic/dummy/placeholder data, or synthetic fallbacks. If
the required columns or data are missing/insufficient, RAISE an exception (e.g.
`raise ValueError("required column X not found in dataset Y")`). Do NOT substitute
any made-up data under any circumstances.

DO NOT FILTER TO A HAND-WRITTEN LIST OF ENTITIES. Never restrict the analysis to a
literal set of gene, protein, variant or sample names written from your own
knowledge. The released data defines which entities are testable; a curated list
written from memory usually does not intersect it, leaving an EMPTY result whose
first symptom is a KeyError on a column of an empty frame. Select entities by a
property computable FROM the data (a significance threshold, an effect size, the
presence of a shared identifier), and annotate afterwards if a category matters.

CHECK EVERY FILTER AND JOIN: after each step that can drop rows, verify the result
is non-empty and RAISE with the counts if it is not (e.g.
`raise ValueError(f"no rows after merging X and Y: {{len(a)}} x {{len(b)}}")`). An
empty intermediate is a finding about the analysis, not a crash to discover 200
lines later.{context}"""

    def _match_template(self, protocol: ExperimentProtocol) -> Optional[CodeTemplate]:
        """The template that both matches AND can perform the declared tests.

        Two rules beyond the old "first template whose matches() is True":

        1. A template that cannot satisfy the protocol's declared statistical
           tests is skipped. It would otherwise substitute its own analysis for
           the designed one and report success.
        2. When only the CATCH-ALL matches, prefer generated code if an LLM is
           available. `generic_computational` claims every COMPUTATIONAL and
           DATA_ANALYSIS protocol, so reaching it means no template recognised
           the experiment -- which is the case for code generation, not for a
           generic descriptive fallback. It stays the fallback when there is no
           LLM to generate with.
        """
        catch_all = None
        for template in self.templates:
            if not template.matches(protocol):
                continue
            if not template.can_satisfy(protocol):
                logger.info(
                    "Template %s matches but cannot perform the protocol's "
                    "declared tests (%s); skipping it",
                    template.name,
                    ", ".join(
                        getattr(t.test_type, "value", str(t.test_type))
                        for t in protocol.statistical_tests or []
                    ) or "none",
                )
                continue
            if template.name == "generic_computational":
                catch_all = template
                continue
            return template

        if catch_all is not None:
            if self.use_llm and self.llm_client:
                logger.info(
                    "Only the catch-all template matches; generating code for "
                    "this protocol instead"
                )
                return None
            return catch_all
        return None

    def _generate_with_llm(
        self,
        protocol: ExperimentProtocol,
        datasets: Optional[Dict[str, str]] = None,
        brief: bool = False,
        fix_hint: Optional[str] = None,
    ) -> str:
        """Generate code using the configured LLM.

        `datasets` switches the prompt from the single-table instructions to the
        multi-table ones. It is a parameter rather than read off `self` so the
        single-dataset prompt stays byte-identical when it is absent.
        """
        prompt = self._create_code_generation_prompt(protocol, datasets=datasets)
        if brief:
            prompt += (
                "\n\nLENGTH LIMIT: your previous answer was cut off before it "
                "finished. Write the SHORTEST complete script that implements "
                "the protocol: no docstrings, no explanatory comments, no "
                "progress printing, no defensive try/except around every step. "
                "A complete short script is worth far more than a thorough one "
                "that stops mid-line -- an unfinished script cannot run at all."
            )

        if fix_hint:
            prompt += (
                f"\n\nCORRECTION: your previous answer failed -- {fix_hint}."
                + (_empty_frame_advice() if _looks_like_an_empty_frame(fix_hint) else "")
                + f" "
                f"Every function and name you call must be defined in the "
                f"script or imported by it, and every column you read must "
                f"exist in the frame you read it from. Note that pandas adds "
                f"merge suffixes ONLY to columns whose names collide, so a "
                f"column you expected to be renamed may not have been: rename "
                f"explicitly before merging, and assert the columns you need "
                f"exist straight after the merge. Re-emit the COMPLETE "
                f"corrected script."
            )

        try:
            # Ask for the budget explicitly. Left to the client default, this
            # call inherits a prose-sized allowance that a multi-step analysis
            # script overruns.
            try:
                response = self.llm_client.generate(
                    prompt, max_tokens=_CODEGEN_MAX_TOKENS
                )
            except TypeError:
                # A client whose generate() takes no max_tokens.
                response = self.llm_client.generate(prompt)
            # Provider-system clients return an LLMResponse; ClaudeClient returns
            # a str. Normalize to text before extracting code.
            text = response if isinstance(response, str) else getattr(response, "content", str(response))

            # Extract code from response (may be in code blocks)
            code = self._extract_code_from_response(text)

            return code

        except Exception as e:
            # Raise, do not return None. `None` reached `_validate_generated`,
            # where `ast.parse(None)` raises TypeError -- NOT the ValueError the
            # retry block catches -- so the ONE failure a second draw actually
            # fixes (a timeout, a 429) was the only one that got no retry. It
            # fell straight to the single-table template, and the report then
            # blamed "compile() arg 1 must be a string" for what was an API
            # timeout.
            logger.error(f"LLM code generation failed: {e}")
            raise ValueError(f"LLM code generation failed: {e}") from e

    def _create_code_generation_prompt(
        self,
        protocol: ExperimentProtocol,
        datasets: Optional[Dict[str, str]] = None
    ) -> str:
        """Create prompt for LLM code generation.

        With `datasets`, the data-access section is REPLACED rather than added
        to. Appending would leave the prompt contradicting itself -- the
        single-table block says "use ONLY the real dataset at data_path", which
        instructs the model to ignore every other dataset it was just handed.
        """
        steps_text = "\n".join([
            f"{i+1}. {step.title}: {step.action}"
            for i, step in enumerate(protocol.steps)
        ])

        variables_text = "\n".join([
            f"- {name} ({var.type.value}): {var.description}"
            for name, var in protocol.variables.items()
        ])

        tests_text = "\n".join([
            f"- {test.test_type}: {test.description}"
            for test in protocol.statistical_tests
        ])

        data_access_text = self._data_access_instructions(datasets)
        analyzer_api = _analyzer_api_text()
        environment = _environment_text()

        prompt = f"""Generate executable Python code for this experiment:

**Experiment:** {protocol.name}
**Type:** {protocol.experiment_type.value}
**Description:** {protocol.description}

**Steps:**
{steps_text}

**Variables:**
{variables_text}

**Statistical Tests:**
{tests_text}

{data_access_text}

Use these libraries: pandas, numpy, scipy.stats
{environment}
{analyzer_api}
Include comments explaining each section

Return ONLY the Python code, no explanations."""

        return prompt

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            # Extract from python code block
            start = response.find("```python") + 9
            end = response.find("```", start)
            code = response[start:end].strip()
        elif "```" in response:
            # Extract from generic code block
            start = response.find("```") + 3
            end = response.find("```", start)
            code = response[start:end].strip()
        else:
            # Assume entire response is code
            code = response.strip()

        return code

    def _enhance_with_llm(self, template_code: str, protocol: ExperimentProtocol) -> str:
        """Enhance template code with LLM additions."""
        prompt = f"""Enhance this experiment code for better results:

**Protocol:** {protocol.name}
**Description:** {protocol.description}

**Current Code:**
```python
{template_code}
```

Enhance the code to:
1. Add any domain-specific preprocessing
2. Add robustness checks
3. Add additional relevant statistics
4. Keep the same structure

CRITICAL - REAL DATA ONLY: Use ONLY the real dataset loaded from `data_path`. Never add
synthetic, random, simulated, or fabricated data, and never add a synthetic-data fallback.
If the data is insufficient for a step, raise an exception instead of substituting data.

Return the enhanced Python code only."""

        try:
            response = self.llm_client.generate(prompt)
            text = response if isinstance(response, str) else getattr(response, "content", str(response))
            enhanced_code = self._extract_code_from_response(text)
            # An enhancement that lost the data load or the results assignment
            # is a regression on working template code, not an improvement.
            self._validate_generated(enhanced_code)
            return enhanced_code
        except Exception as e:
            logger.warning(f"LLM enhancement failed, using original template: {e}")
            return template_code

    def _generate_basic_template(self, protocol: ExperimentProtocol) -> str:
        """Generate basic fallback template."""
        code_lines = [
            "# Basic Experiment Template",
            "# Minimal fallback when no specific template matches",
            "",
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Load data (data_path variable is provided by executor)",
            "df = pd.read_csv(data_path)",
            "# Missingness BEFORE cleaning: the cleaning step below drops rows",
            "# with nulls, so counting afterwards always reports zero and hides",
            "# the very thing worth reporting.",
            "_missing_before = {c: int(n) for c, n in df.isna().sum().items() if n}",
            "_rows_before = int(len(df))",
            "",
            "# Process data",
            # `CodeTemplate.` explicitly, not `self.`: this method lives on
            # ExperimentCodeGenerator, which is NOT a CodeTemplate subclass
            # (`class ExperimentCodeGenerator:`), so `self.clean_data_lines()`
            # raised AttributeError on every run that reached the basic
            # fallback -- i.e. exactly when neither a template nor the LLM is
            # available and this is the only code left to generate.
            *CodeTemplate.clean_data_lines(),
            "",
            "print(f\"Loaded {len(df)} samples\")",
            "print(f\"Columns: {list(df.columns)}\")",
            "",
            "# Basic statistics",
            "print(df.describe())",
            "",
            "# Return a SUMMARY, never the frame itself.",
            "#",
            "# `df.to_dict()` on an 81,410 x 151 table serialised 12 million",
            "# cells into the results payload: a 170 MB row in the database,",
            "# rendered in the report as `data: 151 entries`. A dump is not a",
            "# finding, and it passed every emptiness check because it is",
            "# enormous.",
            "_numeric = df.select_dtypes('number')",
            "results = {",
            "    'n_rows': int(len(df)),",
            "    'n_columns': int(df.shape[1]),",
            "    'columns': list(df.columns)[:200],",
            "    'rows_before_cleaning': _rows_before,",
            "    'missing_by_column': _missing_before,",
            "    'numeric_summary': (",
            "        _numeric.describe().round(4).to_dict() if not _numeric.empty else {}",
            "    ),",
            "    'note': (",
            "        'Produced by the basic fallback template: no template matched "
            "and code generation did not yield a usable script, so this is a "
            "description of the data, not a test of the hypothesis.'",
            "    ),",
            "}"
        ]

        return "\n".join(code_lines)

    def generation_note(self) -> Optional[str]:
        """One sentence naming the analysis that ACTUALLY ran, or None.

        None when the code matches what the protocol describes -- the common
        case, where a note would be noise. A sentence whenever the run fell
        back, because the protocol text then overstates what happened: it
        describes a join across several datasets while the code that ran read
        the primary table alone.
        """
        info = getattr(self, "last_generation", None) or {}
        path = info.get("path")
        if path in (None, "multi_dataset_llm", "llm"):
            return None

        # A SINGLE-dataset template run is not a fallback -- templates are the
        # designed first step there -- but the reader still needs to know what
        # produced the numbers. The protocol text is written for this
        # experiment; the template is generic, so the two can describe
        # different studies. Observed: a design promising a nested logistic
        # regression with a likelihood-ratio test and an odds ratio for
        # `number_inpatient`, under findings that were the ml_experiment
        # template's accuracy, ROC-AUC and 5-fold CV over all 150 columns.
        # Nothing in the report said they were not the same thing.
        if len(getattr(self, "datasets", {}) or {}) <= 1:
            if path == "basic_template":
                # The path that runs when NOTHING else worked: no template
                # matched and code generation produced nothing usable. It
                # described the data and was reported as a successful
                # experiment with no provenance line at all, because it carries
                # no template name to put in one. A multi-dataset run says
                # something stronger below -- that N datasets went unused.
                return (
                    "Produced by the BASIC FALLBACK template, which runs only "
                    "when no template matched and code generation produced "
                    "nothing usable. It describes the dataset; it does not test "
                    "the hypothesis or perform the analysis in the Design above."
                )
            named = info.get("template")
            if not named:
                return None
            return (
                f"Produced by the generic {named!r} template, not by code "
                f"written for this protocol. The Design above describes the "
                f"experiment that was DESIGNED; the template runs its own "
                f"standard analysis over the dataset and reports its own "
                f"metrics. Read the two together before taking these numbers "
                f"as an answer to the question."
            )

        what = (
            f"the single-table template {info['template']!r}"
            if path == "template" and info.get("template")
            else "a generic single-table fallback"
        )
        reason = info.get("fallback_reason") or "generation failed"
        return (
            f"NOT the designed analysis: {len(self.datasets)} datasets were "
            f"mounted, but code generation fell back to {what}, which reads "
            f"only the primary dataset. Any cross-dataset claim in the protocol "
            f"above was not tested. Fallback reason: {reason}."
        )

    @staticmethod
    def _validate_syntax(code: str) -> None:
        """Validate Python syntax of generated code."""
        try:
            ast.parse(code)
            # `ast.parse` stops after parsing. A second class of SyntaxError is
            # raised later, when the compiler builds the symbol table, and
            # `ast.parse` accepts all of it: `global x` after x is assigned,
            # duplicate argument names, `return` outside a function, `await`
            # outside async. One of those -- "name 'p12' is assigned to before
            # global declaration" -- passed this gate and killed the container
            # at line 250 in 117ms. compile() runs both passes and executes
            # nothing.
            compile(code, "<generated>", "exec")
            logger.info("Code syntax validation passed")
        except SyntaxError as e:
            # Quote the offending line. The retry below feeds this message back
            # to the model, and "unmatched ')' at line 40" is far more use to it
            # than the line number alone -- it cannot see the file.
            offending = ""
            lines = code.splitlines()
            if e.lineno and 0 < e.lineno <= len(lines):
                offending = f" -- the line reads: {lines[e.lineno - 1].strip()!r}"
            logger.error(f"Generated code has syntax error: {e}")
            raise ValueError(
                f"Invalid Python syntax in generated code: {e}{offending}"
            )

    @staticmethod
    def _validate_substance(code: str) -> None:
        """Reject code that parses cleanly but analyses nothing.

        Syntax is not a sufficient gate, and the gap is not theoretical. When a
        provider spends its whole output budget on reasoning tokens it returns
        `finish_reason=length` with no content; the extractor then scrapes
        fragments out of the prompt, and what reached the sandbox on the
        myocardial-fibrosis run was `analyzer = DataAnalyzer()` -- 81
        characters, valid Python, no imports, no data, no results. It executed,
        raised NameError, and the run still reported COMPLETED with `data: {}`.
        A fragment is indistinguishable from an experiment by `ast.parse`
        alone, so two further properties are required:

          * the code opens the data it was handed -- `data_path` / `datasets`,
            or any reader call, and
          * it binds `results`, which is what the executor collects.

        These are cheap structural checks, not a proof the analysis is
        correct. They only separate "something ran" from "nothing could have
        run" -- which is exactly the distinction the run above got wrong.
        """
        tree = ast.parse(code)
        if not _reads_data(tree):
            raise ValueError(
                "generated code never opens a dataset: no data_path/datasets "
                "reference and no reader call"
            )
        from kosmos.execution.executor import RESULT_CAPTURE_NAMES

        # Module scope specifically. The sandbox reads `globals()` after the
        # script finishes, so a `results` built inside `main()` and never
        # assigned at the top level is invisible to it -- the script runs
        # cleanly, returns nothing, and the run reports an experiment that
        # produced no output. Accepting any of the capture's names also stops
        # this gate rejecting `result = ...`, which the capture takes happily.
        if not (_module_level_bindings(tree) & set(RESULT_CAPTURE_NAMES)):
            raise ValueError(
                "generated code assigns none of "
                f"{', '.join(RESULT_CAPTURE_NAMES)} at module level, so the "
                "sandbox can collect nothing from it"
            )
        clobbered = _clobbers_injected_data(tree)
        if clobbered:
            raise ValueError(
                "generated code " + " and ".join(clobbered) + "; the executor "
                "assigns `data_path` (and `datasets`) above the first generated "
                "line and the sandbox passes no command-line arguments, so read "
                "those names as given and never rebind them"
            )
        undefined = _undefined_names(tree)
        if undefined:
            raise ValueError(
                f"generated code {_UNDEFINED_MARKER}(s): {', '.join(undefined)}"
            )
        invented = _invented_analyzer_calls(tree)
        if invented:
            raise ValueError(
                f"generated code {_UNDEFINED_MARKER}(s) on DataAnalyzer: "
                f"{', '.join(invented)}; it has only "
                f"{', '.join(_analyzer_methods())}"
            )

    @classmethod
    def _validate_generated(cls, code: str) -> None:
        """Full gate for model-written code: syntax, then substance."""
        cls._validate_syntax(code)
        cls._validate_substance(code)

    def save_code(self, code: str, file_path: str) -> None:
        """Save generated code to file."""
        with open(file_path, 'w') as f:
            f.write(code)
        logger.info(f"Saved generated code to {file_path}")
