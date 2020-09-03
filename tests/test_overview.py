import pytest

from rasalit.apps.overview.common import read_reports, mk_viewable


@pytest.mark.parametrize("report", ["intent", "entity"])
def test_read_reports_lines(report):
    assert len(read_reports("tests/gridresults", report=report)) == 18


@pytest.mark.parametrize("report", ["intent", "entity"])
def test_read_reports_fix(report):
    assert len(read_reports("tests/gridresults", report=report).pipe(mk_viewable)) == 6
