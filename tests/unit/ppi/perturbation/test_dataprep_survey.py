"""The perturbation survey's rules: queries, gates, label evidence, estimates."""

from __future__ import annotations

from kosmos.ppi.perturbation.dataprep.spec import ScreenSpec, modality_compatible, modality_of
from kosmos.ppi.perturbation.dataprep.survey import (
    _estimate_cells,
    _queries,
    gate,
    label_evidence,
    render,
    run_survey,
    search,
)

SPEC = ScreenSpec(
    question="q",
    cell_type="K562",
    modality="CRISPRi",
    topic="unfolded protein response",
    assays=["Perturb-seq", "CROP-seq"],
)


def test_queries_anchor_the_topic_for_the_gold_and_the_cell_type_for_evidence():
    queries = _queries(SPEC)
    assert "unfolded protein response Perturb-seq" in queries["gold"]
    assert "unfolded protein response K562" in queries["gold"]
    assert all("K562" in q for q in queries["supplementary"])


def test_search_ranks_on_topic_titles_first():
    hits = {
        "unfolded protein response Perturb-seq": [
            {"reference": "geo://GSE90546", "title": "dissection of the unfolded protein response"},
        ],
    }

    def run_query(query):
        return hits.get(query, [{"reference": "hf://x/k562", "title": "x/k562"}])

    found = search(SPEC, run_query=run_query, log=lambda *_: None)
    assert found[0].reference == "geo://GSE90546"
    upr = next(c for c in found if c.reference == "geo://GSE90546")
    assert upr.on_topic and "gold" in upr.pools


def test_a_gold_needs_a_single_cell_screen_of_the_right_cells_with_labels():
    good = {
        "role": "gold", "is_single_cell": True, "is_perturbation_screen": True,
        "cell_type_matches": True, "organism": "human", "modality": "CRISPRi",
        "has_perturbation_labels": False,
    }
    # the model said "no labels"; the sample's own identity file says otherwise
    assert gate(SPEC, good, labels="label file cell_identities.csv.gz") == ("gold", "")
    role, why = gate(SPEC, good, labels="")
    assert role == "supplementary" and "no per-cell perturbation labels" in why
    assert gate(SPEC, {**good, "modality": "CRISPRa"}, labels="x")[0] == "supplementary"
    assert gate(SPEC, {**good, "is_single_cell": False}, labels="x")[0] == "reject"
    assert gate(SPEC, {**good, "cell_type_matches": False}, labels="x")[0] == "reject"


def test_supplementary_needs_single_cell_data_of_the_right_cells():
    verdict = {"role": "supplementary", "is_single_cell": True, "cell_type_matches": True}
    assert gate(SPEC, verdict)[0] == "supplementary"
    assert gate(SPEC, {**verdict, "is_single_cell": False})[0] == "reject"


def test_label_evidence_reads_file_names_and_companion_samples():
    sample = {"files": [{"name": "GSM1_10X010_cell_identities.csv.gz"}, {"name": "GSM1_matrix.mtx.gz"}]}
    assert "cell_identities" in label_evidence(sample, [])
    bare = {"files": [{"name": "GSM2_matrix.mtx.gz"}]}
    assert label_evidence(bare, []) == ""
    companion = {"id": "GSM3", "title": "UPR experiment guide barcodes"}
    assert "GSM3" in label_evidence(bare, [companion])


def test_cell_estimate_from_the_barcodes_file():
    files = [{"name": "GSM2406681_10X010_barcodes.tsv.gz", "bytes": 249856}]
    assert 60_000 < _estimate_cells(files) < 70_000  # true count: 65,337
    assert _estimate_cells([{"name": "matrix.mtx.gz", "bytes": 10}]) is None


def test_modality_words():
    assert modality_of("CRISPRi knockdown of UPR genes") == "CRISPRi"
    assert modality_of("CRISPR knockout of regulators") == "CRISPRko"
    assert modality_compatible("CRISPRi", "any") and not modality_compatible("CRISPRi", "CRISPRa")


def test_run_survey_end_to_end_with_fakes(tmp_path, monkeypatch):
    import kosmos.ppi.perturbation.dataprep.survey as survey

    def fake_describe(candidate, config):
        candidate.samples = [
            {"id": "GSM1", "title": "UPR Perturb-seq cells", "bytes": 7e8, "cells_estimate": 65000,
             "raw_barcodes": False, "files": [{"name": "GSM1_cell_identities.csv.gz", "bytes": 8e5}]},
            {"id": "GSM2", "title": "atlas", "bytes": 2e9, "cells_estimate": 150000,
             "raw_barcodes": False, "files": [{"name": "GSM2_matrix.mtx.gz", "bytes": 2e9}]},
        ]
        return candidate

    class Client:
        def generate_structured(self, prompt, schema, **_):
            if "samples" in schema["properties"]:
                base = {"is_single_cell": True, "cell_type_matches": True, "organism": "human",
                        "modality": "CRISPRi", "is_perturbation_screen": True}
                return {"samples": [
                    {**base, "id": "GSM1", "role": "gold", "has_perturbation_labels": False, "topic_match": 1},
                    {**base, "id": "GSM2", "role": "supplementary", "is_perturbation_screen": False},
                ]}
            return {"cell_type": "K562", "modality": "CRISPRi", "topic": "unfolded protein response"}

    monkeypatch.setattr(survey, "describe", fake_describe)
    result = run_survey(
        "q", tmp_path, client=Client(), log=lambda *_: None, scbasecount=False,
        run_query=lambda q: [{"reference": "geo://GSE1", "title": "unfolded protein response screen"}],
    )
    assert [r["sample"] for r in result["gold"]] == ["GSM1"]
    assert [r["sample"] for r in result["supplementary"]] == ["GSM2"]
    assert "GSM1" in (tmp_path / "survey.md").read_text()
    assert "Gold candidates" in render(result)


def test_a_topic_of_generic_words_is_dropped():
    from kosmos.ppi.perturbation.dataprep.spec import specific_topic

    assert specific_topic("transcriptional response prediction") == ""
    assert specific_topic("transcriptional regulators") == ""
    assert specific_topic("unfolded protein response") == "unfolded protein response"
    assert specific_topic("T cell receptor signaling") == "T cell receptor signaling"


def test_series_level_samples_collapse_to_one_row():
    from kosmos.ppi.perturbation.dataprep.survey import _collapse_series_level

    base = {"series": "geo://GSE92872", "series_title": "CROP-seq Jurkat", "role": "gold",
            "series_level": True, "bytes": 19e6}
    rows = [{**base, "sample": f"GSM{i}"} for i in range(12)] + [
        {"series": "geo://GSE1", "sample": "GSM9", "role": "gold", "series_level": False}
    ]
    collapsed = _collapse_series_level(rows)
    assert len(collapsed) == 2
    assert collapsed[0]["sample"] == "GSE92872" and len(collapsed[0]["samples"]) == 12


def test_generic_cell_names_are_not_search_terms():
    from kosmos.ppi.perturbation.dataprep.survey import short_cell_type, specific_cell_terms

    spec = ScreenSpec("q", cell_type="iPSC-derived neurons", cell_type_aliases=["human neurons", "neurons", "iPSC neurons"])
    assert short_cell_type(spec) == "iPSC neurons"
    assert "neurons" not in specific_cell_terms(spec) and "human neurons" not in specific_cell_terms(spec)
    assert all("iPSC" in q for q in _queries(spec)["supplementary"])


def test_hf_repos_must_name_the_cells():
    spec = ScreenSpec("q", cell_type="iPSC-derived neurons", cell_type_aliases=["neurons", "iPSC neurons"])

    def run_query(query):
        return [
            {"reference": "hf://neuralblog/llama3-8b-mlp-neurons", "title": "neuralblog/llama3-8b-mlp-neurons"},
            {"reference": "geo://GSE277602", "title": "single-cell atlas of iPSC neurons"},
        ]

    found = [c.reference for c in search(spec, run_query=run_query, log=lambda *_: None)]
    assert "geo://GSE277602" in found and "hf://neuralblog/llama3-8b-mlp-neurons" not in found


def test_a_lane_takes_its_own_guide_companion():
    sample = {"title": "CROP-seq_Day 7_neuron_10x_lane_2", "files": [{"name": "GSM3_matrix.mtx.gz"}]}
    companions = [
        {"id": "GSM3543620", "title": "CROP-seq_iPSC_sgRNA_enrichment_for_lane1"},
        {"id": "GSM3543624", "title": "CROP-seq_Day 7_sgRNA_enrichment_for_lane1"},
        {"id": "GSM3543625", "title": "CROP-seq_Day 7_sgRNA_enrichment_for_lane2"},
    ]
    assert "GSM3543625" in label_evidence(sample, companions)
    assert "GSM3543620" not in label_evidence(sample, companions)


# -- GEO full text, scBaseCount, fit ---------------------------------------------


def test_a_series_named_only_in_its_summary_is_on_topic():
    spec = ScreenSpec(question="q", cell_type="THP-1", modality="CRISPRko", topic="PD-L1")
    hits = [
        {"reference": "geo://GSE153056", "title": "inhibitory immune checkpoints with multi-modal single-cell screens",
         "summary": "pooled CRISPR screens ... regulate PD-L1 expression"},
        {"reference": "geo://GSE2", "title": "an unrelated screen", "summary": ""},
    ]
    found = search(spec, run_query=lambda q: hits, log=lambda *_: None)
    papalexi = next(c for c in found if c.reference == "geo://GSE153056")
    assert papalexi.on_topic
    assert found[0].reference == "geo://GSE153056"


def test_geo_full_text_ands_the_words_and_widens_screen(monkeypatch):
    import datafetcher.sources.geo as geo

    seen = {}

    def fake(path, params, config):
        if path == "esearch.fcgi":
            seen["term"] = params["term"]
            return {"esearchresult": {"idlist": ["200153056"]}}
        return {"result": {"200153056": {"accession": "GSE153056", "title": "t", "summary": "s"}}}

    monkeypatch.setattr(geo, "_eutils_json", fake)
    hits = geo.search_series("THP-1 CRISPR screen", None, organism="human")
    assert seen["term"] == '(THP-1 AND CRISPR AND screen*) AND gse[ETYP] AND "Homo sapiens"[ORGN]'
    assert hits == [{"reference": "geo://GSE153056", "title": "t", "summary": "s"}]


def _sbc_table():
    import pandas as pd

    def row(uid, srx, line, tissue, pert, cells, chem="3_prime_gex"):
        return {"entrez_id": uid, "srx_accession": srx, "file_path": f"gs://arc-institute-virtual-cell-atlas/x/{srx}.h5ad",
                "obs_count": cells, "tech_10x": chem, "cell_prep": "single_cell", "cell_line": line,
                "tissue": tissue, "perturbation": pert, "czi_collection_name": None}

    return pd.DataFrame([
        row(1, "SRX1", "THP-1", "THP-1 cell line", "CRISPR pooled screen (ECCITE-seq)", 20000, "5_prime_gex"),
        row(2, "SRX2", "THP-1", "THP-1 cell line", "none", 15000, "5_prime_gex"),
        row(3, "SRX3", "THP1", "monocytes", "none", 9000),
        row(4, "SRX4", "THP-1", "macrophages", "LPS", 30000),
        row(5, "SRX5", "Jurkat", "T cells", "none", 50000),
        row(6, "SRX6", "THP-1", "THP-1 cell line", "CRISPR screen", 40000, "5_prime_gex"),
    ])


def test_scbasecount_name_keys_and_prefilter():
    from kosmos.ppi.perturbation.dataprep import scbasecount as sbc

    assert sbc.name_keys(["THP-1"]) == [["thp1"]]
    assert sbc.name_keys(["iPSC-derived neurons"]) == [["ipsc", "neuron"]]
    assert set(sbc.prefilter(_sbc_table(), ["THP-1"])["srx_accession"]) == {"SRX1", "SRX2", "SRX3", "SRX4", "SRX6"}


def test_scbasecount_rows_drop_the_gold_and_put_its_chemistry_first(monkeypatch):
    from kosmos.ppi.perturbation.dataprep import scbasecount as sbc

    monkeypatch.setattr(sbc, "object_sizes", lambda paths: {p: 1000 for p in paths})
    spec = ScreenSpec(question="q", cell_type="THP-1")
    rows = sbc.survey_rows(spec, client=None, exclude_uids=[1, 2], chemistry="5_prime_gex",
                           table=_sbc_table(), log=lambda *_: None)
    # the 5' screen first (same chemistry), then 3' unlabeled before 3' perturbed? by cells
    assert [r["sample"] for r in rows] == ["SRX6", "SRX4", "SRX3"]
    assert rows[0]["screen"] and rows[0]["chemistry"] == "5_prime_gex"
    assert rows[0]["files"][0]["name"].startswith("https://storage.googleapis.com/")
    assert rows[0]["bytes"] == 1000


def test_scbasecount_choice_keeps_what_the_model_names():
    from kosmos.ppi.perturbation.dataprep import scbasecount as sbc

    class Client:
        def generate_structured(self, prompt, schema, **_):
            names = [line.split(": ", 1)[1].split("  (")[0] for line in prompt.splitlines() if line[:1].isdigit()]
            return {"keep": [i for i, n in enumerate(names) if "macrophage" not in n]}

    spec = ScreenSpec(question="q", cell_type="THP-1")
    kept = sbc.choose(spec, sbc.prefilter(_sbc_table(), ["THP-1"]), Client())
    assert "SRX4" not in set(kept["srx_accession"]) and "SRX1" in set(kept["srx_accession"])


def test_run_survey_links_the_gold_to_scbasecount(tmp_path, monkeypatch):
    import kosmos.ppi.perturbation.dataprep.scbasecount as sbc
    import kosmos.ppi.perturbation.dataprep.survey as survey

    monkeypatch.setattr(sbc, "object_sizes", lambda paths: {p: None for p in paths})

    def fake_describe(candidate, config):
        candidate.samples = [
            {"id": "GSM1", "title": "ECCITE-seq GEX", "bytes": 7e8, "cells_estimate": None,
             "raw_barcodes": True, "files": [{"name": "GSM1_cell_identities.csv.gz", "bytes": 8e5}]},
        ]
        return candidate

    class Client:
        def generate_structured(self, prompt, schema, **_):
            if "samples" in schema["properties"]:
                return {"samples": [{"id": "GSM1", "role": "gold", "is_single_cell": True, "cell_type_matches": True,
                                     "organism": "human", "modality": "CRISPRko", "is_perturbation_screen": True,
                                     "has_perturbation_labels": True, "topic_match": 1}]}
            if "keep" in schema["properties"]:
                return {"keep": list(range(10))}
            return {"cell_type": "THP-1", "modality": "CRISPRko", "topic": "PD-L1"}

    monkeypatch.setattr(survey, "describe", fake_describe)
    result = run_survey(
        "q", tmp_path, client=Client(), log=lambda *_: None,
        run_query=lambda q: [{"reference": "geo://GSE153056", "title": "PD-L1 screens"}],
        scbasecount_table=_sbc_table(), sra_links=lambda series: {1: "GSM1", 2: "GSM9"},
    )
    gold = result["gold"][0]
    assert gold["scbasecount"] == {"srx": ["SRX1"], "cells": 20000, "chemistry": "5_prime_gex"}
    supp = [r["sample"] for r in result["supplementary"]]
    assert "SRX1" not in supp and "SRX2" not in supp and "SRX6" in supp
    fit = result["fit"][0]
    assert fit["gold_cells"] == 20000 and fit["supp_cells"] == 40000 + 30000 + 9000
    assert fit["ratio"] == 4.0 and fit["enough"]
    assert "supp / gold" in (tmp_path / "survey.md").read_text()


def test_a_matrix_named_after_its_crispr_library_is_not_label_evidence():
    sample = {"title": "CROP-seq_CRISPRi_10x_lane_1",
              "files": [{"name": "GSM4632022_CRISPRi_filtered_feature_bc_matrix_lib1.h5"}]}
    assert label_evidence(sample, []) == ""
    sample["files"].append({"name": "GSM4632022_sgRNA_calls.csv.gz"})
    assert label_evidence(sample, []).startswith("label file GSM4632022_sgRNA_calls")


def test_an_unverified_h5ad_gold_is_verified_by_its_obs(monkeypatch):
    import kosmos.ppi.perturbation.dataprep.survey as survey

    seen = []
    monkeypatch.setattr(survey, "peek_h5ad_labels", lambda url: seen.append(url) or "obs column 'guide' (90 values)")
    rows = [{"role": "gold", "sample": "GSM8787485", "labels": "unverified (model)",
             "files": [{"name": "GSM8787485_EXP224.h5ad", "bytes": 1e9}]},
            {"role": "gold", "sample": "GSM1", "labels": "label file x", "files": [{"name": "a.h5ad", "bytes": 1}]}]
    survey._verify_h5ad_labels(rows, log=lambda *_: None)
    assert rows[0]["labels"] == "obs column 'guide' (90 values)"
    assert seen == ["https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8787nnn/GSM8787485/suppl/GSM8787485_EXP224.h5ad"]
