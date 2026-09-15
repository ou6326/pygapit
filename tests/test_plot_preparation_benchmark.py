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
        "render_backends": [],
    }
    measurements = report["measurements"]
    assert [measurement["name"] for measurement in measurements] == [
        "genomic_axis",
        "manhattan_data",
        "manhattan_points_object",
        "manhattan_aggregate_object",
    ]


def test_plot_preparation_benchmark_can_measure_renderer_stage() -> None:
    report = run_plot_preparation_benchmark(
        n_markers=20,
        n_chromosomes=2,
        warmups=0,
        repeats=1,
        render_backends=("matplotlib",),
    )

    assert report["workload"]["render_backends"] == ["matplotlib"]
    assert [measurement["name"] for measurement in report["measurements"]][-2:] == [
        "manhattan_points_render_matplotlib",
        "manhattan_aggregate_render_matplotlib",
    ]
