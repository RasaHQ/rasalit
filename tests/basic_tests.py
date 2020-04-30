import pytest
from rasalit.__main__ import app_path


@pytest.mark.parametrize("f", ["viewresults.py", "html/diet"])
def test_can_find_files(f):
    app_path(f)
