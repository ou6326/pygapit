"""Correctness checks for the manual PCA store-I/O benchmark."""

from benchmarks.benchmark_pca_store_io import run_pca_store_io_benchmark


def test_pca_store_io_benchmark_records_dense_and_sparse_reads() -> None:
    report = run_pca_store_io_benchmark(
        individuals=40,
        parent_markers=40,
        retained_markers=10,
        components=3,
        marker_workspace_mib=0.001,
        seed=20260912,
        warmups=0,
        repeats=1,
        backends=("numpy",),
    )

    dense, interleaved = report.measurements
    assert dense.pattern == "dense"
    assert interleaved.pattern == "interleaved"
    assert interleaved.parent_read_calls > dense.parent_read_calls
    assert interleaved.parent_cells_read == dense.parent_cells_read
