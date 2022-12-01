# -*- coding: utf-8 -*-
"""
    conftest.py for mruns.
"""

import pathlib
import sys

root = pathlib.Path(".").parent.parent
sys.path.append(str(root / "src"))
sys.path.append(str(root.parent / "mreports" / "src"))
print(sys.path)
import mbf
import mreports
import mruns
from mruns.util import filter_function, read_toml
from mruns.base import analysis
from pathlib import Path
import pytest

from pypipegraph.testing.fixtures import (  # noqa:F401
    new_pipegraph,
    both_ppg_and_no_ppg,
    no_pipegraph,
    pytest_runtest_makereport,
)

data_folder = Path(__file__).parent / "data"
toml_file = data_folder / "run.toml"


@pytest.fixture
@pytest.mark.usefixtures("new_pipegraph_no_qc")
def ana():
    ans = analysis(toml_file)
    return ans


@pytest.fixture
def toml_input():
    return read_toml(toml_file)
