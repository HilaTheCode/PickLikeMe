import time

from picklikeme.profiling import PipelineProfiler, StageTimer


def test_stage_timer_records_and_formats_a_report() -> None:
    timer = StageTimer(enabled=True)

    with timer.stage("sample stage"):
        time.sleep(0.01)

    report = timer.report("Sample report", images=1)

    assert "Sample report" in report
    assert "sample stage" in report
    assert "1" in report


def test_pipeline_profiler_records_stage_totals_and_report() -> None:
    profiler = PipelineProfiler(images=2, decode_workers=2, device="cpu")

    with profiler.stage("RAW loading"):
        time.sleep(0.01)

    profiler.record_stage("CSV generation", 0.02, count=1)

    report = profiler.build_report(total_runtime=0.05)

    assert "RANKING PERFORMANCE REPORT" in report
    assert "RAW loading" in report
    assert "CSV generation" in report
    assert "Images processed: 2" in report
