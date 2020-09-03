import pytest

from rasalit.apps.overview.common import read_reports


@pytest.mark.parametrize("report", ["intent", "entity"])
def test_read_reports_lines(report):
    assert len(read_reports("tests/gridresults", report=report)) == 6
