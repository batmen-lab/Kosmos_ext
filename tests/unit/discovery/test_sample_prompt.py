"""The review packet as the model reads it, for a table wider than the prompt."""

from __future__ import annotations

from kosmos.discovery.sample_prompt import render_packet


def test_the_fetchers_rendering_is_used_verbatim():
    """The fetcher owns the readers, so it owns the head-and-tail rendering."""
    rendered = "file: cells.csv\n  GENE1, ... [10 more],cell_type"
    packet = {"path": "x/cells.csv", "rendered": rendered}

    assert render_packet(packet) == rendered


def test_a_packet_without_a_rendering_shows_both_ends_of_a_wide_table():
    columns = [f"GENE{i}" for i in range(200)] + ["cell_type", "batch"]
    packet = {
        "path": "x/cells.csv",
        "raw_head": ["GENE0,GENE1,...", "1.0,2.0,..."],
        "columns": columns,
        "kinds": dict.fromkeys(columns, "numeric"),
        "null_fraction": {},
        "delimiter": ",",
        "rows_sampled": 5,
        "notes": [],
    }

    rendered = render_packet(packet, max_columns=40)

    assert "cell_type" in rendered
    assert "batch" in rendered
    assert "first and last 20 of 202" in rendered
