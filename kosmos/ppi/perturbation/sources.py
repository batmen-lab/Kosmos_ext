"""Where real perturbation screens live, and which pairs are compatible.

The auxiliary co-expression graph is built from **control** cells, so a
supplementary source only has to be the same cell line/tissue and measure the
same genes -- its perturbations are irrelevant. That makes the useful set much
larger than "another copy of the gold", and it is why these datasets are
registered here with their cell line rather than with their perturbation list.

Every entry is a screen published with GEARS (Harvard Dataverse), which is also
where the processed `.h5ad` and a per-dataset GO graph come from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationSource:
    name: str
    url: str
    cell_line: str
    perturbation: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "url": self.url,
            "cell_line": self.cell_line,
            "perturbation": self.perturbation,
            "notes": self.notes,
        }


#: Harvard Dataverse file ids from GEARS' `pertdata.py`.
SOURCES: dict[str, PerturbationSource] = {
    "norman": PerturbationSource(
        "norman",
        "https://dataverse.harvard.edu/api/access/datafile/6154020",
        "K562",
        "CRISPRa (sgRNA)",
        "the screen GEARS' `norman` loader uses; single and paired perturbations",
    ),
    "adamson": PerturbationSource(
        "adamson",
        "https://dataverse.harvard.edu/api/access/datafile/6154417",
        "K562",
        "CRISPRi (sgRNA)",
        "one perturbation per cell, many guides",
    ),
    "dixit": PerturbationSource(
        "dixit",
        "https://dataverse.harvard.edu/api/access/datafile/6154416",
        "K562",
        "CRISPRi / CRISPRa",
        "Perturb-seq with a transcription-factor library",
    ),
    "replogle_k562_essential": PerturbationSource(
        "replogle_k562_essential",
        "https://dataverse.harvard.edu/api/access/datafile/7458695",
        "K562",
        "CRISPRi (genome-wide, filtered)",
        "filtered release; a good *control-cell* source for a K562 gold",
    ),
    "replogle_rpe1_essential": PerturbationSource(
        "replogle_rpe1_essential",
        "https://dataverse.harvard.edu/api/access/datafile/7458694",
        "RPE1",
        "CRISPRi (genome-wide, filtered)",
        "RPE1 only: not a control source for a K562 gold",
    ),
}


def default_pair(cell_line: str = "K562") -> tuple[PerturbationSource, list[PerturbationSource]]:
    """A gold and the supplementary sources that share its cell line.

    The gold is the screen with paired perturbations (the harder prediction
    task); the supplementary candidate is a *different* screen in the same cell
    line, whose control cells build the auxiliary graph.
    """
    gold = SOURCES["norman"]
    others = [
        source
        for source in SOURCES.values()
        if source.cell_line == cell_line and source.name != gold.name
    ]
    others.sort(key=lambda source: 0 if "replogle" in source.name else 1)
    return gold, others
