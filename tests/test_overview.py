import pytest

from rasalit.apps.overview.common import read_reports, mk_viewable, create_altair_chart


@pytest.mark.parametrize("report", ["intent", "entity"])
def test_read_reports_lines(report):
    assert len(read_reports("tests/gridresults", report=report)) == 18


@pytest.mark.parametrize("report", ["intent", "entity"])
def test_read_reports_fix(report):
    assert len(read_reports("tests/gridresults", report=report).pipe(mk_viewable)) == 6


@pytest.mark.parametrize("report", ["intent", "entity"])
def test_altair_plot(report):
    d = (
        read_reports("tests/gridresults", report=report)
        .pipe(create_altair_chart)
        .to_dict()
    )

    key = list(d["datasets"].keys())[0]

    # Confirm correct number of drawn points
    assert len(d["datasets"][key]) == 18

    # Confirm correct columns picked up
    assert all([v in d["datasets"][key][0] for v in ["config", "variable", "value"]])
