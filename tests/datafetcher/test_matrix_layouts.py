"""Matrix-market layouts as GEO publishes them, and the cache/preflight fixes."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datafetcher.fetch import _metadata_beside
from datafetcher.singlecell import SingleCellSelection, companion_kind, convert_single_cell


def _gz(path: Path, text: str) -> Path:
    with gzip.open(path, "wt") as handle:
        handle.write(text)
    return path


def _matrix(path: Path, genes: int, cells: int) -> Path:
    lines = ["%%MatrixMarket matrix coordinate real general", "%", f"{genes} {cells} {cells}"]
    lines += [f"{(cell % genes) + 1} {cell + 1} {cell % 5 + 1}" for cell in range(cells)]
    return _gz(path, "\n".join(lines) + "\n")


def test_a_geo_prefixed_triplet_and_its_cell_identities_become_a_labeled_table(tmp_path):
    """GSE133344: `<prefix>_matrix.mtx.gz` + `<prefix>_barcodes/genes.tsv.gz` + identities."""
    cells, genes = 30, 6
    matrix = _matrix(tmp_path / "GSE1_filtered_matrix.mtx.gz", genes, cells)
    _gz(tmp_path / "GSE1_filtered_barcodes.tsv.gz", "\n".join(f"AAAC{c}-1" for c in range(cells)))
    _gz(tmp_path / "GSE1_filtered_genes.tsv.gz", "\n".join(f"ENSG{g}\tGENE{g}" for g in range(genes)))
    _gz(tmp_path / "GSE1_raw_cell_identities.csv.gz", "cell_barcode,guide_identity\nX-1,junk\n")
    _gz(
        tmp_path / "GSE1_filtered_cell_identities.csv.gz",
        "cell_barcode,guide_identity,UMI_count\n"
        + "\n".join(f"AAAC{c}-1,{'CBL_NegCtrl0__CBL_NegCtrl0' if c % 2 else 'NegCtrl0_NegCtrl0__NegCtrl0_NegCtrl0'},{c}" for c in range(cells)),
    )
    assert companion_kind("GSE1_filtered_barcodes.tsv.gz", matrix.name) == "barcodes"
    metadata = _metadata_beside(matrix)
    assert metadata.name == "GSE1_filtered_cell_identities.csv.gz"  # not raw, not a companion
    table, _ = convert_single_cell(matrix, selection=SingleCellSelection(max_cells=100), metadata=metadata)
    assert len(table) == cells and {"GENE0", "guide_identity", "UMI_count", "barcode"} <= set(table.columns)
    assert table["guide_identity"].notna().all()


def test_a_dixit_layout_with_an_inverted_guide_map_is_read(tmp_path):
    """GSE90063: `X.mtx.txt.gz`, `X_cellnames.csv.gz`, `X_genenames.csv.gz`, `X_cbc_gbc_dict*.csv.gz`."""
    cells, genes = 12, 4
    matrix = _matrix(tmp_path / "GSM1_k562.mtx.txt.gz", genes, cells)
    _gz(tmp_path / "GSM1_k562_cellnames.csv.gz", ",0\n" + "\n".join(f"{c},CELL{c}_p7" for c in range(cells)))
    _gz(tmp_path / "GSM1_k562_genenames.csv.gz", ",0\n" + "\n".join(f"{g},ENSG000{g}_SYM{g}" for g in range(genes)))
    _gz(tmp_path / "GSM1_k562_cbc_gbc_dict_lenient.csv.gz", 'p_sgA,"CELL0_p7"\n')
    _gz(
        tmp_path / "GSM1_k562_cbc_gbc_dict_strict.csv.gz",
        'p_sgA_1,"CELL0_p7, CELL1_p7, CELL2_p7"\np_sgB_2,"CELL2_p7, CELL3_p7"\n',
    )
    metadata = _metadata_beside(matrix)
    assert metadata.name.endswith("strict.csv.gz")
    table, _ = convert_single_cell(matrix, selection=SingleCellSelection(max_cells=100), metadata=metadata)
    assert {"SYM0", "SYM3"} <= set(table.columns)
    guides = dict(zip(table["barcode"], table["guide"]))
    assert guides["CELL0_p7"] == "p_sgA_1"
    assert guides["CELL2_p7"] == "p_sgA_1;p_sgB_2"  # a cell with two guides keeps both


def test_a_zero_byte_staged_file_is_never_reused(tmp_path):
    from datafetcher.models import FetchResult, FileRecord
    from datafetcher.store import _still_on_disk

    (tmp_path / "empty.csv.gz").write_bytes(b"")
    record = FetchResult.model_construct(
        directory=str(tmp_path),
        files=[FileRecord(path="empty.csv.gz", bytes=0, sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", derived_from=None)],
    )
    assert _still_on_disk(record, verify=True) is None


def test_an_empty_download_is_an_error_not_a_file(tmp_path, monkeypatch):
    import io

    import datafetcher.sources.net as net
    from datafetcher.config import DataFetcherConfig
    from datafetcher.errors import SourceError

    class Empty(io.BytesIO):
        headers = {"Content-Length": "0"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(net, "_open", lambda url, config: Empty(b""))
    monkeypatch.setattr(net, "_check_offline", lambda config, url: None)
    with pytest.raises(SourceError, match="cut short"):
        net.download("https://example.org/x.csv.gz", tmp_path / "x.csv.gz", DataFetcherConfig(root=tmp_path))
    assert not (tmp_path / "x.csv.gz").exists()


def test_preflight_keeps_a_matrix_with_its_companions_and_labels():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from auto_task_run import _keep_matrix_companions

    class Log:
        def event(self, *args, **kwargs):
            pass

    def entry(name, size=10):
        return {"path": name, "reference": f"geo://GSE1#suppl/{name}", "bytes": size}

    listing = {
        "files": [
            entry("GSE1_filtered_matrix.mtx.gz", 10**9),
            entry("GSE1_filtered_barcodes.tsv.gz"),
            entry("GSE1_filtered_genes.tsv.gz"),
            entry("GSE1_filtered_cell_identities.csv.gz"),
            entry("GSE1_raw_matrix.mtx.gz", 10**9),
            entry("GSE1_raw_barcodes.tsv.gz"),
            entry("GSE1_RAW.tar", 10**10),
        ]
    }
    chosen = [{"reference": "geo://GSE1#suppl/GSE1_filtered_matrix.mtx.gz"}]
    kept = {Path(e["reference"]).name for e in _keep_matrix_companions(listing, chosen, Log())}
    assert kept == {
        "GSE1_filtered_matrix.mtx.gz",
        "GSE1_filtered_barcodes.tsv.gz",
        "GSE1_filtered_genes.tsv.gz",
        "GSE1_filtered_cell_identities.csv.gz",
    }


def test_nested_sample_archives_are_unpacked_and_their_tables_kept_apart(tmp_path):
    """GEO RAW.tar -> GSM*_matrix.tar.gz -> matrix.mtx: both levels are
    unpacked, and two samples' `matrix.mtx` do not overwrite one table."""
    import io
    import tarfile

    from datafetcher.fetch import FetchResult, FileRecord, _table_label, _unpack_archives

    def tar_bytes(files: dict[str, bytes], gz: bool) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz" if gz else "w") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    inner = {name: tar_bytes({"matrix.mtx": b"x", "barcodes.tsv": b"b"}, gz=True)
             for name in ("GSM1_A_matrix.tar.gz", "GSM2_B_matrix.tar.gz")}
    raw = tmp_path / "GSE1_RAW.tar"
    raw.write_bytes(tar_bytes({**inner, "notes.xlsx": b"PK\x03\x04"}, gz=False))
    result = FetchResult(
        reference="geo://GSE1", scheme="geo", locator="GSE1", directory=str(tmp_path),
        files=[FileRecord(path=raw.name, bytes=raw.stat().st_size, sha256="0" * 64)],
    )
    _unpack_archives(result)
    paths = {record.path for record in result.files}
    assert "GSM1_A_matrix.tar.gz-contents/matrix.mtx" in paths
    assert "GSM2_B_matrix.tar.gz-contents/matrix.mtx" in paths
    assert not any("notes.xlsx-contents" in path for path in paths)
    first = tmp_path / "GSM1_A_matrix.tar.gz-contents" / "matrix.mtx"
    assert _table_label(first, tmp_path, {"matrix.mtx"}) == "GSM1_A_matrix__matrix.mtx"
    assert _table_label(first, tmp_path, set()) == "matrix.mtx"


def test_a_matrix_fetched_before_its_companions_is_converted_when_they_arrive(tmp_path):
    """Each GEO supplementary file is its own fetch: the matrix can land first."""
    from scipy import io as sio
    from scipy import sparse

    from datafetcher.config import DataFetcherConfig
    from datafetcher.fetch import FetchResult, FileRecord, _convert_single_cell

    matrix = sparse.random(5, 4, density=0.8, format="coo", random_state=0) * 10
    sio.mmwrite(tmp_path / "GSE1_filtered_matrix.mtx", matrix.astype(int))
    config = DataFetcherConfig(root=tmp_path)

    def fetched(name):
        return FetchResult(
            reference=f"geo://GSE1#suppl/{name}", scheme="geo", locator="GSE1",
            directory=str(tmp_path),
            files=[FileRecord(path=name, bytes=(tmp_path / name).stat().st_size, sha256="0" * 64)],
        )

    first = fetched("GSE1_filtered_matrix.mtx")
    _convert_single_cell(first, config)
    assert not (tmp_path / "GSE1_filtered_matrix.mtx-table.csv").exists()
    (tmp_path / "GSE1_filtered_features.tsv").write_text(
        "".join(f"ENSG{i}\tG{i}\tGene Expression\n" for i in range(5))
    )
    (tmp_path / "GSE1_filtered_barcodes.tsv").write_text("".join(f"C{i}-1\n" for i in range(4)))
    last = fetched("GSE1_filtered_barcodes.tsv")
    _convert_single_cell(last, config)
    assert (tmp_path / "GSE1_filtered_matrix.mtx-table.csv").exists()
    assert any(record.path.endswith("-table.csv") for record in last.files)


def _write_10x_v2(path, genes, barcodes, dense):
    """A CellRanger v2 `.h5`: genome group, CSC over cells (genes x cells)."""
    import h5py
    from scipy import sparse

    matrix = sparse.csc_matrix(np.asarray(dense, dtype=np.int32).T)  # genes x cells
    with h5py.File(path, "w") as handle:
        group = handle.create_group("hg38_spiked_lib")
        group["data"] = matrix.data
        group["indices"] = matrix.indices
        group["indptr"] = matrix.indptr
        group["shape"] = np.array(matrix.shape)
        group["barcodes"] = np.array([b.encode() for b in barcodes])
        group["gene_names"] = np.array([g.encode() for g in genes])
        group["genes"] = np.array([g.encode() for g in genes])


def test_a_cropseq_10x_h5_becomes_a_labeled_table(tmp_path):
    """GSE137554: guides are matrix features (`Lib_GENE_n_gene`), the matrix is
    unfiltered, and the annotation beside it names the real cells of two samples."""
    import numpy as np

    from datafetcher.singlecell import SingleCellSelection, convert_single_cell, is_single_cell

    guides = [f"Tcrlibrary_{g}_{n}_gene" for g in ("LCK", "ZAP70", "LAT") for n in (1, 2)]
    genes = ["ACTB", "CD3E", "IL2"] + guides
    # cells: LCK guide / ZAP70 guide / no guide / two genes' guides, the same
    # barcode body in both samples, and many empty droplets (an unfiltered matrix)
    barcodes = ["AAA-1", "CCC-1", "GGG-1", "TTT-1", "AAA-2"] + [f"E{i}-1" for i in range(20)]
    dense = np.zeros((len(barcodes), len(genes)), dtype=int)
    dense[:5, :3] = 5
    dense[0, 3] = 2      # LCK_1
    dense[1, 5] = 1      # ZAP70_1
    dense[3, [3, 7]] = 1  # LCK_1 + LAT_1
    dense[4, 4] = 3      # LCK_2, other sample
    path = tmp_path / "GSE1_raw_gene_bc_matrices_h5.h5"
    _write_10x_v2(path, genes, barcodes, dense)
    annotation = tmp_path / "GSE1_CellAnnotation.tsv"
    annotation.write_text(
        "Cell barcode\tSample\nAAA-1\tnaive\nCCC-1\tnaive\nGGG-1\tnaive\nTTT-1\tnaive\nAAA-2\tstim\n"
    )

    assert is_single_cell(path)
    table, selection = convert_single_cell(path, selection=SingleCellSelection(), metadata=annotation)
    by_cell = table.set_index("barcode")
    assert set(by_cell.index) == {"AAA-1", "CCC-1", "GGG-1", "TTT-1", "AAA-2"}  # droplets dropped
    assert by_cell.loc["AAA-1", "guide"] == "LCK" and by_cell.loc["AAA-2", "guide"] == "LCK"
    assert by_cell.loc["CCC-1", "guide"] == "ZAP70"
    assert by_cell.loc["GGG-1", "guide"] == "no_guide"
    assert by_cell.loc["TTT-1", "guide"] == "multiple"
    assert by_cell.loc["AAA-2", "Sample"] == "stim"  # two samples' `AAA` kept apart
    assert not any(str(c).startswith("Tcrlibrary_") for c in table.columns)
    assert {"ACTB", "CD3E", "IL2"} <= set(table.columns)


def test_gene_symbols_with_underscores_are_not_a_guide_library():
    from datafetcher.singlecell import guide_features

    assert guide_features(["HLA_A_1", "ACTB", "CD3E"]) == {}
    library = [f"Lib_G{i}_1_gene" for i in range(6)]
    assert len(guide_features(["ACTB", *library])) == 6
    typed = guide_features(["ACTB", "LCK-1"], ["Gene Expression", "CRISPR Guide Capture"])
    assert typed == {1: "LCK"}
