"""Contract tests for the Manhattan preparation benchmark."""

from benchmarks.benchmark_plot_preparation import run_plot_preparation_benchmark


def test_plot_preparation_benchmark_reports_separate_stages() -> None:
    report = run_plot_preparation_benchmark(
        n_markers=100,
        n_chromosomes=5,
        warmups=0,
        repeats=1,
    )

    assert report["workload"] == {
        "markers": 100,
        "chromosomes": 5,
        "warmups": 0,
        "repeats": 1,
    }
    measurements = report["measurements"]
    assert [measurement["name"] for measurement in measurements] == [
        "genomic_axis",
        "manhattan_data",
    ]
