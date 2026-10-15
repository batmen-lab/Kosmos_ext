"""Stage 1 for perturbation questions, built around datasets and samples.

    survey   question -> spec -> searches -> per-sample metadata -> judged
             candidates (gold / supplementary), with sizes; downloads nothing
    fetch    (next) download the chosen samples' own files
    load     (next) read every format into one standard object and validate it

The per-cell stage 1 (`scripts/auto_task_run.py`) treats every file as a table
with a label column; a screen is a matrix plus per-cell perturbation labels
spread over several files, and fitting it into that model failed on almost
every new question.
"""

from .spec import ScreenSpec, extract_spec
from .survey import run_survey

__all__ = ["ScreenSpec", "extract_spec", "run_survey"]
