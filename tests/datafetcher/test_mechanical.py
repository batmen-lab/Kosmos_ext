"""Round one: the query rule, and the adapters for what repositories return."""

from __future__ import annotations

from datafetcher.mechanical import mechanical_search
from datafetcher.query import keyword_query
from datafetcher.search_tools import Hit, geo_datasets, huggingface, omicsdi


def test_the_query_keeps_domain_words_and_drops_question_furniture():
    query = keyword_query(
        "Can diabetes status be predicted from routine clinical measurements "
        "(glucose, blood pressure, insulin, BMI, age), and does unlabeled data "
        "from another patient cohort improve it?"
    )
    words = query.lower().split()
    assert "diabetes" in words and "glucose" in words
    # `predicted` survived the first version of this rule; the stem check exists
    # because of it.
    assert "predicted" not in words and "measurements" not in words
    assert "improve" not in words and "does" not in words


def test_an_explicit_query_wins():
    assert keyword_query("anything", extra_terms=["heart failure"]) == "heart failure"


def test_the_query_keeps_what_the_question_is_about_not_the_long_words():
    """Ranking by length kept "environmental"/"parameters" and dropped "spin coating".

    The search then returned environmental-claims text and environmental-sound
    audio datasets, and the run ended with no labeled table at all.
    """
    query = keyword_query(
        "Determine how environmental parameters during spin coating and thermal "
        "annealing affect perovskite solar-cell efficiency."
    )
    assert query == (
        "perovskite solar-cell thermal annealing spin coating efficiency"
    )
    assert "environmental" not in query and "parameters" not in query


def test_a_list_of_words_is_not_glued_into_a_phrase():
    query = keyword_query(
        "Can diabetes status be predicted from routine clinical measurements "
        "(glucose, blood pressure, insulin, BMI, age)?"
    )
    assert "blood pressure" in query  # written next to each other
    assert "glucose blood" not in query  # separated by a comma, so a list


def test_a_trailing_period_does_not_reach_the_query():
    assert "efficiency." not in keyword_query("Does annealing affect efficiency.")


def test_omicsdi_maps_arrayexpress_imports_onto_geo():
    """E-GEOD-21321 is ArrayExpress's copy of GSE21321, so the GEO connector routes it."""
    hits = omicsdi(
        {
            "data": {
                "datasets": [
                    {"id": "GSE194122", "source": "geo", "title": "BMMC"},
                    {"id": "E-GEOD-21321", "source": "arrayexpress", "title": "diabetes"},
                    {"id": "PXD055651", "source": "pride", "title": "proteomics"},
                    {"id": "", "source": "geo"},
                ]
            }
        },
        "OmicsDI_search_datasets",
    )
    references = {hit.accession: hit.reference for hit in hits}
    assert references["GSE194122"] == "geo://GSE194122"
    assert references["E-GEOD-21321"] == "geo://GSE21321"
    assert references["PXD055651"] is None  # a lead, honestly labelled
    assert len(hits) == 3


def test_geo_and_huggingface_payloads_are_read_from_their_real_envelopes():
    """Both shapes were probed live: GEO nests under data.datasets, HF under data."""
    geo = geo_datasets(
        {
            "status": "success",
            "data": {
                "total": 710,
                "datasets": [{"accession": "GSE342320", "title": "diabetic kidney disease"}],
            },
        },
        "GEO_search_rnaseq_datasets",
    )
    assert [hit.reference for hit in geo] == ["geo://GSE342320"]

    hf = huggingface(
        {"status": "success", "data": [{"id": "Plashkar/diabetes-predict-db"}]},
        "HuggingFace_search_datasets",
    )
    assert [hit.reference for hit in hf] == ["hf://Plashkar/diabetes-predict-db"]


def test_null_references_are_not_offered_for_download():
    search = mechanical_search("q", downloader=None)
    # No interpreter configured in the test environment: the round reports that
    # rather than pretending it searched.
    assert search.references == []


def test_references_are_deduplicated_and_only_fetchable_ones_listed():
    class S:
        query = "diabetes"
        hits = [
            Hit("GSE1", "geo://GSE1", "geo", "t"),
            Hit("GSE1", "geo://GSE1", "geo", "t"),
            Hit("lead", None, "zenodo", "t"),
        ]

    assert S().hits[0].to_dict()["reference"] == "geo://GSE1"
    from datafetcher.mechanical import MechanicalSearch

    result = MechanicalSearch(query="q", hits=S.hits)
    assert result.references == ["geo://GSE1"]
