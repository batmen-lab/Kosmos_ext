"""Choosing what to download from what it costs, before the bytes move."""

from __future__ import annotations

import json

from kosmos.discovery.preflight import choose_downloads, plan_prompt


class FakeClient:
    """A model that answers with a fixed structured response."""

    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt, schema, max_tokens=0, temperature=0):
        self.prompts.append(prompt)
        return self.response


def listing(reference="geo://GSE194122", **overrides) -> dict:
    payload = {
        "reference": reference,
        "about": "A sandbox for prediction and integration of DNA, RNA, and proteins",
        "why": "bone-marrow mononuclear cells with cell-type annotations",
        "bytes": 3_300_000_000,
        "files": [
            {
                "path": "suppl/GSE194122_cite_BMMC.h5ad.gz",
                "bytes": 615_514_112,
                "size": "587.0 MB",
                "reference": "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz",
            },
            {
                "path": "suppl/GSE194122_multiome_BMMC.h5ad.gz",
                "bytes": 2_900_000_000,
                "size": "2.7 GB",
                "reference": "geo://GSE194122#suppl/GSE194122_multiome_BMMC.h5ad.gz",
            },
        ],
    }
    return {**payload, **overrides}


def test_the_prompt_shows_what_it_is_and_what_it_costs():
    text = plan_prompt("q", [listing()])

    assert "587.0 MB" in text
    assert "2.7 GB" in text
    assert "smallest first" in text
    assert "about:" in text


def test_the_model_may_choose_one_file_out_of_a_series():
    client = FakeClient(
        {
            "downloads": [
                {
                    "reference": "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz",
                    "why": "the smaller of the two carries the annotated cells",
                }
            ]
        }
    )

    chosen = choose_downloads("q", [listing()], client=client)

    assert chosen == [
        {
            "reference": "geo://GSE194122#suppl/GSE194122_cite_BMMC.h5ad.gz",
            "why": "the smaller of the two carries the annotated cells",
        }
    ]
    # The prompt asks for the measurement file, smallest among those.
    assert "Never choose an index of the data" in client.prompts[0]
    assert "choose the smallest" in client.prompts[0]


def test_a_reference_that_was_never_offered_is_ignored():
    client = FakeClient({"downloads": [{"reference": "geo://GSE99999"}]})

    assert choose_downloads("q", [listing()], client=client) == []


def test_an_empty_answer_is_respected():
    """Nothing on offer is a decision, not a failure to decide."""
    client = FakeClient({"downloads": []})
    assert choose_downloads("q", [listing()], client=client) == []


def test_a_failed_model_call_keeps_the_candidates():
    class Broken:
        def generate_structured(self, **_kwargs):
            raise RuntimeError("no provider")

    chosen = choose_downloads("q", [listing()], client=Broken())

    assert [entry["reference"] for entry in chosen] == ["geo://GSE194122"]


def test_a_non_json_answer_keeps_the_candidates():
    client = FakeClient("not json at all")
    chosen = choose_downloads("q", [listing()], client=client)
    assert [entry["reference"] for entry in chosen] == ["geo://GSE194122"]


def test_a_json_string_answer_is_accepted():
    client = FakeClient(json.dumps({"downloads": [{"reference": "geo://GSE194122"}]}))
    chosen = choose_downloads("q", [listing()], client=client)
    assert chosen[0]["reference"] == "geo://GSE194122"
