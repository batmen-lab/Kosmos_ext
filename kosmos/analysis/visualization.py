"""
Publication-quality visualization templates from kosmos-figures.

Matches exact formatting standards from kosmos-figures repository:
- Arial font (TrueType for editability)
- Specific color scheme (#d7191c red, #0072B2 blue, #abd9e9 neutral)
- Publication-quality DPI (300-600)
- Proper spine/grid formatting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from pathlib import Path
import logging

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Imported for typing only. This module is EXEC'd into the experiment sandbox,
# where the kosmos package does not exist -- a runtime import of kosmos.models
# there raises ImportError and the whole module falls back to a no-op stub,
# which is why figures silently never appeared. `select_plot_types` is the only
# host-side entry point and is the only thing that needs the real class.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from kosmos.models.result import ExperimentResult

logger = logging.getLogger(__name__)


# kosmos-figures color scheme (EXACT colors from repository)
COLORS = {
    'red': '#d7191c',  # Negative/decreased
    'blue_dark': '#2c7bb6',  # Positive/increased (dark)
    'blue': '#0072B2',  # Positive/increased (standard)
    'neutral': '#abd9e9',  # Data points
    'gray': '#808080',  # Non-significant
    'black': '#000000'  # Reference lines
}


class PublicationVisualizer:
    """
    Publication-quality visualization templates from kosmos-figures.

    All plots match exact formatting standards:
    - Arial font family
    - TrueType fonts (pdf.fonttype=42)
    - kosmos-figures color scheme
    - DPI 300 (standard) or 600 (panels)
    - Removed top/right spines
    - bbox_inches='tight'

    Example:
        ```python
        viz = PublicationVisualizer()

        # Volcano plot
        viz.volcano_plot(
            log2fc=log2_fold_changes,
            p_values=p_values,
            labels=gene_names,
            output_path="volcano.png"
        )

        # Scatter with regression
        viz.scatter_with_regression(
            x=independent_var,
            y=dependent_var,
            x_label="Independent Variable",
            y_label="Dependent Variable",
            title="Correlation Analysis",
            output_path="scatter.png"
        )
        ```
    """

    def __init__(self):
        """Initialize visualizer with kosmos-figures formatting standards."""
        # Set publication standards from kosmos-figures
        plt.rcParams.update({
            # sans-serif with a fallback chain rather than 'Arial' outright:
            # the sandbox image ships no Arial, so every text object logged
            # "Font family 'Arial' not found" -- hundreds of lines per figure,
            # drowning the stderr an experiment is judged on. The figures
            # always fell back to DejaVu anyway; this just names the fallback.
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
            'font.size': 10,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'pdf.fonttype': 42,  # TrueType for editability
            'ps.fonttype': 42,
            'figure.dpi': 100,  # Display DPI
            'savefig.dpi': 300,  # Save DPI (default, can override)
            'savefig.bbox': 'tight'
        })

        logger.info("PublicationVisualizer initialized with kosmos-figures formatting")

    # ========================================================================
    # CORE PLOT TYPES (from integration-plan.md templates)
    # ========================================================================

    def volcano_plot(
        self,
        log2fc: np.ndarray,
        p_values: np.ndarray,
        labels: Optional[np.ndarray] = None,
        fc_threshold: float = 0.5,
        p_threshold: float = 0.05,
        title: str = "Volcano Plot",
        x_label: str = "Log2 Fold Change",
        y_label: str = "-log10(p-value)",
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Create volcano plot: -log10(p) vs log2(fold change).

        Pattern from: Figure_2_hypothermia_nucleotide_salvage

        Args:
            log2fc: Log2 fold change values
            p_values: P-values
            labels: Optional labels for significant points
            fc_threshold: Fold change threshold for significance
            p_threshold: P-value threshold
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Path to save figure (None = don't save)
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate -log10(p)
        log_p = -np.log10(p_values + 1e-300)  # Add tiny value to avoid log(0)

        # Color by significance
        colors = [
            COLORS['red'] if (abs(fc) > fc_threshold and p < p_threshold) else COLORS['gray']
            for fc, p in zip(log2fc, p_values)
        ]

        # Scatter plot
        ax.scatter(log2fc, log_p, c=colors, alpha=0.7, s=60, edgecolors='none')

        # Threshold lines
        ax.axhline(
            y=-np.log10(p_threshold),
            color=COLORS['black'],
            linestyle='--',
            alpha=0.7,
            linewidth=1.5,
            label=f'p={p_threshold}'
        )
        ax.axvline(x=0, color=COLORS['black'], linestyle='-', alpha=0.3, linewidth=1)

        # Labels
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=12, pad=20)
        ax.legend(fontsize=11)

        # Grid (kosmos-figures style - light grid for volcano plots)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate significant points if labels provided
        if labels is not None:
            for i, (fc, p, label) in enumerate(zip(log2fc, p_values, labels)):
                if abs(fc) > fc_threshold and p < p_threshold:
                    ax.annotate(
                        label,
                        (fc, -np.log10(p + 1e-300)),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        ha='left',
                        alpha=0.8
                    )

        plt.tight_layout()

        # Save
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved volcano plot to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    def custom_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Heatmap",
        cmap: str = 'RdBu_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        annot: bool = True,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Create custom heatmap with publication formatting.

        Pattern from: Figure_2_hypothermia_nucleotide_salvage

        Args:
            data: 2D array of values
            row_labels: Row labels
            col_labels: Column labels
            title: Plot title
            cmap: Colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            annot: Whether to annotate cells with values
            output_path: Path to save figure
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        # Set ticks and labels
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=12, rotation=45, ha='right')
        ax.set_yticklabels(row_labels, fontsize=12)
        ax.set_title(title, pad=12, fontsize=14)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value', rotation=270, labelpad=20, fontsize=12)

        # Add text annotations if requested
        if annot:
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    value = data[i, j]
                    text = f'{value:.2f}' if not np.isnan(value) else 'N/A'

                    # Determine text color for visibility
                    if vmax is not None:
                        threshold = vmax * 0.5
                    else:
                        threshold = np.nanmax(np.abs(data)) * 0.5

                    color = 'white' if abs(value) > threshold else 'black'

                    ax.text(
                        j, i, text,
                        ha='center', va='center',
                        color=color, fontsize=10, weight='bold'
                    )

        plt.tight_layout()

        # Save
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    def scatter_with_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Scatter plot with linear regression fit.

        Pattern from: Figure_3_perovskite_solar_cell

        Args:
            x: X values
            y: Y values
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            output_path: Path to save figure
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        if not HAS_SCIPY:
            logger.error("scipy is required for regression. Install with: pip install scipy")
            raise ImportError("scipy required")

        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter plot - using kosmos-figures color scheme
        ax.scatter(
            x, y,
            alpha=0.7,
            s=60,
            color=COLORS['neutral'],
            edgecolors='black',
            linewidth=0.5
        )

        # Linear fit - using kosmos-figures red color
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = slope * x_trend + intercept

        ax.plot(
            x_trend, y_trend,
            color=COLORS['red'],
            linestyle='--',
            alpha=0.8,
            linewidth=2,
            label=f'Linear fit (r = {r_value:.3f}, p = {p_value:.4f})'
        )

        # Formatting (kosmos-figures style)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=12, pad=20)
        ax.legend(fontsize=11)

        # Remove grid and top/right spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scatter plot to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    def log_log_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
        color: str = COLORS['blue'],
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Log-log scatter plot for power law relationships.

        Pattern from: Figure_4_neural_network

        Args:
            x: X values
            y: Y values
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            color: Point color
            output_path: Path to save figure
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # Log-log scatter
        ax.scatter(x, y, alpha=0.6, s=40, color=color, edgecolors='none')

        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Labels with large fonts (kosmos-figures style for panels)
        ax.set_xlabel(x_label, fontsize=24)
        ax.set_ylabel(y_label, fontsize=24)
        ax.set_title(title, fontsize=24)
        ax.tick_params(labelsize=20, width=2, length=7)

        # Thicker spines (kosmos-figures panel style)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # No grid
        ax.grid(False)

        plt.tight_layout()

        # Save with high DPI for panels
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
            logger.info(f"Saved log-log plot to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    # ========================================================================
    # ADDITIONAL STATISTICAL PLOTS
    # ========================================================================

    def box_plot_with_points(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Box Plot",
        y_label: str = "Value",
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Box plot with overlaid individual data points.

        Args:
            data: Dictionary of {label: values}
            title: Plot title
            y_label: Y-axis label
            output_path: Path to save figure
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create box plot
        positions = list(range(len(data)))
        bp = ax.boxplot(
            data.values(),
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False  # We'll overlay individual points
        )

        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['neutral'])
            patch.set_alpha(0.7)

        # Overlay individual points with jitter
        for i, (label, values) in enumerate(data.items()):
            # Add jitter to x-positions
            x_jitter = np.random.normal(positions[i], 0.04, size=len(values))
            ax.scatter(
                x_jitter, values,
                alpha=0.5,
                s=30,
                color=COLORS['blue_dark'],
                edgecolors='black',
                linewidth=0.5
            )

        # Labels
        ax.set_xticks(positions)
        ax.set_xticklabels(data.keys(), fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=12, pad=20)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

        plt.tight_layout()

        # Save
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved box plot to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    def forest_plot(
        self,
        labels: List[str],
        estimates: np.ndarray,
        ci_low: np.ndarray,
        ci_high: np.ndarray,
        x_label: str = "Effect estimate",
        title: str = "Forest Plot",
        null_value: float = 0.0,
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """One row per estimate with its confidence interval.

        The figure an MR or meta-analysis wants and the one this class did not
        have: per-protein effects with CIs, a reference line at the null, and
        the intervals that exclude it picked out. Added because the analyses
        this agent actually runs -- instrumental-variable MR across proteins,
        random-effects meta-analysis across cohorts -- produced exactly this
        shape of result and had no way to draw it.

        Args:
            labels: one name per row (protein, cohort, cell type)
            estimates: the point estimates
            ci_low, ci_high: interval bounds, same length as estimates
            null_value: where the reference line goes (0 for a difference,
                1 for a ratio)
            output_path: where to save; returns '' when not given

        Returns:
            The path written, or '' if nothing was saved.
        """
        estimates = np.asarray(estimates, dtype=float).ravel()
        ci_low = np.asarray(ci_low, dtype=float).ravel()
        ci_high = np.asarray(ci_high, dtype=float).ravel()
        labels = [str(x) for x in labels]

        if not (len(labels) == len(estimates) == len(ci_low) == len(ci_high)):
            raise ValueError(
                f"forest_plot needs equal lengths, got labels={len(labels)}, "
                f"estimates={len(estimates)}, ci_low={len(ci_low)}, "
                f"ci_high={len(ci_high)}"
            )
        if len(estimates) == 0:
            raise ValueError("forest_plot needs at least one estimate")

        # A forest plot of 400 rows is a black smear. Keep the largest
        # absolute effects, which is what such a plot is read for.
        if len(estimates) > 40:
            order = np.argsort(-np.abs(estimates - null_value))[:40]
            order = np.sort(order)
            labels = [labels[i] for i in order]
            estimates, ci_low, ci_high = estimates[order], ci_low[order], ci_high[order]
            logger.warning("forest_plot truncated to the 40 largest effects")

        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(estimates) + 1.2)))
        y = np.arange(len(estimates))[::-1]
        excludes_null = (ci_low > null_value) | (ci_high < null_value)
        for yi, est, lo, hi, sig in zip(y, estimates, ci_low, ci_high, excludes_null):
            colour = COLORS['red'] if sig else COLORS['gray']
            ax.plot([lo, hi], [yi, yi], color=colour, linewidth=1.6, solid_capstyle='butt')
            ax.plot([est], [yi], marker='s', markersize=6, color=colour, linestyle='none')

        ax.axvline(null_value, color=COLORS['black'], linestyle='--', linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved forest plot to {output_path}")
        if show_plot:
            plt.show()
        plt.close(fig)
        return output_path if output_path else ""

    def violin_plot(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Violin Plot",
        y_label: str = "Value",
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Violin plot for distribution visualization.

        Args:
            data: Dictionary of {label: values}
            title: Plot title
            y_label: Y-axis label
            output_path: Path to save figure
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Prepare data for seaborn
        positions = list(range(len(data)))

        # Create violin plot using matplotlib
        parts = ax.violinplot(
            data.values(),
            positions=positions,
            widths=0.7,
            showmeans=True,
            showmedians=True
        )

        # Color violins
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['neutral'])
            pc.set_alpha(0.7)

        # Labels
        ax.set_xticks(positions)
        ax.set_xticklabels(data.keys(), fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=12, pad=20)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

        plt.tight_layout()

        # Save
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved violin plot to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    def qq_plot(
        self,
        data: np.ndarray,
        title: str = "Q-Q Plot",
        output_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Quantile-quantile plot for normality checking.

        Args:
            data: Data array to check for normality
            title: Plot title
            output_path: Path to save figure
            show_plot: Whether to display plot

        Returns:
            str: Path to saved figure
        """
        if not HAS_SCIPY:
            logger.error("scipy is required for Q-Q plot. Install with: pip install scipy")
            raise ImportError("scipy required")

        fig, ax = plt.subplots(figsize=(6, 6))

        # Calculate theoretical quantiles
        (osm, osr), (slope, intercept, r) = sp_stats.probplot(data, dist="norm")

        # Plot
        ax.scatter(osm, osr, alpha=0.7, s=40, color=COLORS['blue'], edgecolors='none')

        # Reference line
        ax.plot(osm, slope * osm + intercept, color=COLORS['red'], linestyle='--', linewidth=2,
                label=f'Normal (R²={r**2:.3f})')

        # Labels
        ax.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax.set_ylabel('Sample Quantiles', fontsize=12)
        ax.set_title(title, fontsize=12, pad=20)
        ax.legend(fontsize=11)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        plt.tight_layout()

        # Save
        if output_path:
            output_path = str(output_path)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Q-Q plot to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path if output_path else ""

    # ========================================================================
    # AUTOMATIC PLOT SELECTION
    # ========================================================================

    def select_plot_types(self, result: "ExperimentResult") -> List[Dict[str, Any]]:
        """
        Automatically select appropriate plot types based on experiment result.

        Args:
            result: ExperimentResult object

        Returns:
            list: List of plot specifications with parameters
        """
        plots = []

        # Check what statistical tests were performed
        if result.statistical_tests:
            primary_test = result.primary_test.lower() if result.primary_test else ""

            # T-test or ANOVA → Box plot or violin plot
            if 't-test' in primary_test or 'anova' in primary_test:
                plots.append({
                    'type': 'box_plot_with_points',
                    'description': 'Box plot for group comparison'
                })

            # Correlation → Scatter with regression
            if 'correlation' in primary_test or 'regression' in primary_test:
                plots.append({
                    'type': 'scatter_with_regression',
                    'description': 'Scatter plot with linear regression'
                })

            # Multiple tests → Volcano plot (if fold changes available)
            if len(result.statistical_tests) >= 3:
                plots.append({
                    'type': 'volcano_plot',
                    'description': 'Volcano plot for multiple comparisons'
                })

        # Check for multiple variables → Heatmap
        if result.variable_results and len(result.variable_results) >= 3:
            plots.append({
                'type': 'custom_heatmap',
                'description': 'Heatmap for multi-variable comparison'
            })

        # Default: If no specific plots selected, suggest basic visualizations
        if not plots:
            plots.append({
                'type': 'box_plot_with_points',
                'description': 'Default: Box plot for data visualization'
            })

        logger.info(f"Selected {len(plots)} plot types for experiment {result.experiment_id}")
        return plots
