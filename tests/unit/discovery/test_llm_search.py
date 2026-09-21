"""Retrieval by the model: it proposes datasets, and says so out loud."""

from __future__ import annotations

import pytest

from kosmos.discovery import Proposal, propose_datasets, propose_evidence


class FakeClient:
    def __init__(self, response, raises=None):
        self.response = response
        self.raises = raises
        self.prompts = []

    def generate_structured(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return self.response


class ScriptedClient:
    """One response per call, so a two-round conversation can be tested."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_structured(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else {"datasets": []}


def test_a_search_call_is_run_and_its_hits_are_handed_back_to_the_model():
    """The model used a ToolUniverse search tool; that lead used to be dropped.

    In the perovskite run it asked `tu://HuggingFace_search_datasets` for exactly
    the right thing, and the pipeline threw the call away.
    """
    call = 'tu://HuggingFace_search_datasets#{"query": "perovskite solar cell"}'
    client = ScriptedClient(
        [
            {"datasets": [{"identifier": call, "why": "search the Hub"}]},
            {
                "datasets": [
                    {
                        "identifier": "hf://owner/perovskite-db",
                        "contains": "42k devices with fabrication conditions",
                        "why": "the search hit",
                        "confidence": 0.7,
                    }
                ]
            },
        ]
    )
    printed: list[str] = []
    asked: list[str] = []

    def run_search(identifier: str) -> str:
        asked.append(identifier)
        return "  PDB -> hf://owner/perovskite-db  42k devices"

    proposals = propose_datasets(
        "perovskite efficiency",
        client=client,
        max_items=3,
        log=printed.append,
        run_search=run_search,
    )

    assert asked == [call]
    assert [p.identifier for p in proposals] == ["hf://owner/perovskite-db"]
    assert len(client.prompts) == 2
    assert "What you have already looked up" in client.prompts[1]
    assert any("hop 1/" in line for line in printed)


def test_a_search_call_is_dropped_with_a_reason_when_nothing_can_run_it():
    client = FakeClient(
        {"datasets": [{"identifier": "tu://Zenodo_search_records#{}", "why": "x"}]}
    )
    printed: list[str] = []
    assert propose_datasets("q", client=client, log=printed.append) == []
    assert any("nothing can run a search" in line for line in printed)


def test_the_model_can_search_more_than_once():
    """One hop was not enough: with the hits in hand it searches again, then names."""
    first = 'tu://HuggingFace_search_datasets#{"query": "chronic kidney"}'
    second = 'tu://HuggingFace_search_datasets#{"query": "kidney disease UCI"}'
    client = ScriptedClient(
        [
            {"datasets": [{"identifier": first, "why": "search"}]},
            {"datasets": [{"identifier": second, "why": "refine"}]},
            {
                "datasets": [
                    {
                        "identifier": "hf://owner/ckd",
                        "contains": "ckd cohort",
                        "why": "the search hit",
                        "confidence": 0.9,
                    }
                ]
            },
        ]
    )
    printed: list[str] = []
    asked: list[str] = []
    proposals = propose_datasets(
        "chronic kidney disease",
        client=client,
        max_items=3,
        log=printed.append,
        run_search=lambda call: (asked.append(call), f"  hit for {call}")[1],
        max_hops=3,
    )

    assert asked == [first, second]
    assert [p.identifier for p in proposals] == ["hf://owner/ckd"]
    text = "\n".join(printed)
    assert "hop 1/3" in text and "hop 2/3" in text
    assert "What you have already looked up" in client.prompts[2]


def test_the_loop_stops_at_the_hop_limit():
    """A model that only ever searches is stopped, and the run keeps what it has."""
    call = 'tu://HuggingFace_search_datasets#{"query": "anything"}'

    class AlwaysSearching(ScriptedClient):
        def generate_structured(self, prompt, schema, **kwargs):
            self.prompts.append(prompt)
            return {"datasets": [{"identifier": f"{call}#hop{len(self.prompts)}", "why": "x"}]}

    client = AlwaysSearching([])
    printed: list[str] = []
    proposals = propose_datasets(
        "q",
        client=client,
        max_items=3,
        log=printed.append,
        run_search=lambda call: "  a hit",
        max_hops=2,
    )
    assert proposals == []
    assert len(client.prompts) == 2  # hop 1 and hop 2, then it stops


def test_the_evidence_round_is_told_which_columns_to_match():
    """Round two can only aim once round one has a schema to aim at."""
    client = ScriptedClient(
        [
            {
                "datasets": [
                    {
                        "identifier": "hf://owner/other-cohort",
                        "contains": "the same measurements, another cohort",
                        "why": "shares bill_length_mm and flipper_length_mm",
                        "confidence": 0.7,
                    }
                ]
            }
        ]
    )
    printed: list[str] = []

    proposals = propose_evidence(
        "Can species be told from measurements?",
        columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
        gold_reference="hf://owner/penguins",
        sibling_files=[{"path": "test.csv", "bytes": 1000}],
        search_hits="  GSE1 -> hf://owner/other-cohort  another cohort",
        client=client,
        max_items=2,
        log=printed.append,
    )

    prompt = client.prompts[0]
    assert [p.identifier for p in proposals] == ["hf://owner/other-cohort"]
    assert "bill_length_mm" in prompt
    assert "hf://owner/penguins" in prompt
    assert "test.csv" in prompt  # the repository's other files were listed
    assert "another cohort" in prompt  # the column-name search hits
    assert "do NOT name a mirror" in prompt
    assert any("asking the model for tables with these 3 column(s)" in line for line in printed)


def test_the_model_names_datasets_and_every_step_is_printed():
    client = FakeClient(
        {
            "datasets": [
                {
                    "identifier": "geo://GSE194122",
                    "contains": "CITE-seq BMMC",
                    "why": "matches the tissue and assay",
                    "confidence": 0.8,
                },
                {
                    "identifier": "hf://codesignal/wine-quality",
                    "contains": "wine chemistry",
                    "why": "the measurement in question",
                    "confidence": 0.6,
                },
            ]
        }
    )
    printed: list[str] = []
    proposals = propose_datasets(
        "Can wine quality be predicted?", client=client, max_items=3, log=printed.append
    )

    assert [p.identifier for p in proposals] == [
        "geo://GSE194122",
        "hf://codesignal/wine-quality",
    ]
    assert all(isinstance(p, Proposal) for p in proposals)
    text = "\n".join(printed)
    assert "asking the model for candidate datasets" in text
    assert "geo://GSE194122" in text and "matches the tissue and assay" in text
    assert "hf://codesignal/wine-quality" in text


def test_unfetchable_identifiers_are_dropped_with_a_printed_reason():
    client = FakeClient(
        {
            "datasets": [
                {"identifier": "PMID:12345678", "why": "a paper"},
                {"identifier": "s3://bucket/key", "why": "no downloader for this scheme"},
                {"identifier": "geo://GSE1", "why": "fine"},
                {"identifier": "geo://GSE1", "why": "duplicate"},
            ]
        }
    )
    printed: list[str] = []
    proposals = propose_datasets("q", client=client, log=printed.append)
    assert [p.identifier for p in proposals] == ["geo://GSE1"]
    text = "\n".join(printed)
    assert "dropped 'PMID:12345678'" in text
    assert "dropped 's3://bucket/key'" in text


def test_no_client_is_a_hard_stop_not_an_empty_search():
    """There is no query to fall back to when the model is choosing datasets."""
    with pytest.raises(ValueError, match="needs the model"):
        propose_datasets("q", client=None)


def test_a_failed_model_call_is_raised_after_being_printed():
    client = FakeClient(None, raises=RuntimeError("provider down"))
    printed: list[str] = []
    with pytest.raises(RuntimeError):
        propose_datasets("q", client=client, log=printed.append)
    assert any("model call failed" in line for line in printed)


def test_an_empty_proposal_list_says_so():
    printed: list[str] = []
    proposals = propose_datasets("q", client=FakeClient({"datasets": []}), log=printed.append)
    assert proposals == []
    assert any("proposed no fetchable dataset" in line for line in printed)


def test_hints_and_context_reach_the_prompt():
    client = FakeClient({"datasets": []})
    propose_datasets(
        "q", client=client, hints=["bmmc", "cite-seq"], context="protocol says BMMC"
    )
    prompt = client.prompts[0]
    assert "bmmc" in prompt and "cite-seq" in prompt and "protocol says BMMC" in prompt
    # The instruction that costs downloads if ignored: name real, fetchable ids.
    assert "geo://GSE194122" in prompt
