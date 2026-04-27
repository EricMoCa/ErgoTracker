import pytest
from pathlib import Path
from reports import ReportGenerator


def test_generate_creates_file(tmp_path, sample_report):
    gen = ReportGenerator()
    output = str(tmp_path / "report.pdf")
    result = gen.generate(sample_report, output)
    assert Path(result).exists()


def test_generate_file_not_empty(tmp_path, sample_report):
    gen = ReportGenerator()
    output = str(tmp_path / "report.pdf")
    result = gen.generate(sample_report, output)
    assert Path(result).stat().st_size > 1024


def test_generate_with_empty_frames(tmp_path, empty_report):
    gen = ReportGenerator()
    output = str(tmp_path / "empty_report.pdf")
    result = gen.generate(empty_report, output)
    assert Path(result).exists()


def test_generate_creates_parent_dirs(tmp_path, sample_report):
    gen = ReportGenerator()
    output = str(tmp_path / "subdir" / "nested" / "report.pdf")
    result = gen.generate(sample_report, output)
    assert Path(result).exists()


def test_get_top_violations_empty(sample_report):
    gen = ReportGenerator()
    violations = gen._get_top_violations([])
    assert violations == []


def test_get_top_violations_no_violations(sample_report):
    gen = ReportGenerator()
    violations = gen._get_top_violations(sample_report.frame_scores)
    assert violations == []


def test_generate_returns_string_path(tmp_path, sample_report):
    gen = ReportGenerator()
    output = str(tmp_path / "report.pdf")
    result = gen.generate(sample_report, output)
    assert isinstance(result, str)
