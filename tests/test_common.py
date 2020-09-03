import pytest
from rasalit.common import app_path


@pytest.mark.parametrize("f", ["html/diet"])
def test_can_find_files(f):
    app_path(f)
