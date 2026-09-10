"""
Rich theme configuration for Kosmos CLI.

Provides consistent styling, colors, and visual elements across all CLI commands.
"""

from rich.theme import Theme
from rich.box import Box, ROUNDED, DOUBLE, HEAVY, SIMPLE
from rich.style import Style


# Color palette for Kosmos CLI
COLORS = {
    # Status colors
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "cyan",
    "muted": "bright_black",

    # Semantic colors
    "primary": "bright_blue",
    "secondary": "magenta",
    "accent": "bright_cyan",

    # Domain colors
    "biology": "green",
    "neuroscience": "magenta",
    "materials": "cyan",
    "physics": "blue",
    "chemistry": "yellow",
    "general": "white",

    # State colors
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "pending": "bright_black",
    "paused": "yellow",

    # Metric colors
    "high": "green",
    "medium": "yellow",
    "low": "red",
}


# Rich theme with custom styles
KOSMOS_THEME = Theme({
    # Status styles
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "cyan",
    "muted": "dim",

    # Header styles
    "h1": "bold bright_blue underline",
    "h2": "bold bright_cyan",
    "h3": "bold white",

    # Content styles
    "emphasis": "bold",
    "code": "bold magenta",
    "path": "underline cyan",
    "number": "bold cyan",

    # State styles
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "pending": "bright_black",

    # Domain styles
    "domain.biology": "green",
    "domain.neuroscience": "magenta",
    "domain.materials": "cyan",
    "domain.physics": "blue",
    "domain.chemistry": "yellow",
    "domain.general": "white",

    # Table styles
    "table.header": "bold bright_white on blue",
    "table.caption": "italic bright_black",
    "table.row_even": "",
    "table.row_odd": "on grey11",

    # Progress styles
    "bar.complete": "green",
    "bar.finished": "bright_green",
    "bar.pulse": "bright_cyan",

    # Metric styles
    "metric.high": "bold green",
    "metric.medium": "bold yellow",
    "metric.low": "bold red",
})


# Box styles for panels
BOX_STYLES = {
    "default": ROUNDED,
    "emphasis": DOUBLE,
    "simple": SIMPLE,
    "heavy": HEAVY,
}


def get_domain_color(domain: str) -> str:
    """Get color for a domain."""
    domain_lower = (domain or "").lower()
    return COLORS.get(domain_lower, COLORS["general"])


def get_state_color(state: str) -> str:
    """Get color for a state."""
    state_lower = state.lower()
    return COLORS.get(state_lower, COLORS["muted"])


def get_metric_color(value: float, thresholds: dict = None) -> str:
    """
    Get color for a metric value based on thresholds.

    Args:
        value: Metric value
        thresholds: Dict with 'high' and 'medium' threshold values
                   Default: {'high': 0.7, 'medium': 0.4}

    Returns:
        Color name for the metric
    """
    if thresholds is None:
        thresholds = {"high": 0.7, "medium": 0.4}

    if value >= thresholds.get("high", 0.7):
        return COLORS["high"]
    elif value >= thresholds.get("medium", 0.4):
        return COLORS["medium"]
    else:
        return COLORS["low"]


def get_box_style(style_name: str = "default") -> Box:
    """Get box style by name."""
    return BOX_STYLES.get(style_name, BOX_STYLES["default"])


# Icon/symbol mappings for visual clarity
ICONS = {
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "running": "⟳",
    "pending": "○",
    "completed": "●",
    "arrow": "→",
    "bullet": "•",
    "sparkle": "✨",
    "rocket": "🚀",
    "magnifying_glass": "🔍",
    "book": "📚",
    "flask": "🧪",
    "brain": "🧠",
    "atom": "⚛",
}


def get_icon(icon_name: str) -> str:
    """Get icon/symbol by name."""
    return ICONS.get(icon_name, "")


# Status icon mappings
STATUS_ICONS = {
    "CREATED": "○",
    "STARTING": "⟳",
    "RUNNING": "▶",
    "IDLE": "⏸",
    "WORKING": "⚙",
    "PAUSED": "⏸",
    "STOPPED": "■",
    "ERROR": "✗",
    "COMPLETED": "✓",
}


def get_status_icon(status: str) -> str:
    """Get icon for agent/workflow status."""
    return STATUS_ICONS.get(status.upper(), "?")
