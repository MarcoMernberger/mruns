# -*- coding: utf-8 -*-
"""
    conftest.py for mruns.
"""

import pathlib
import sys

root = pathlib.Path(".").parent.parent
sys.path.append(str(root / "src"))
sys.path.append(str(root.parent / "mreports" / "src"))
import mbf
import mreports
import mruns
from mruns.util import filter_function, read_toml
from mruns.base import analysis
from pathlib import Path
import pytest
import pypipegraph2 as ppg2  # noqa: F401
import pypipegraph as ppg  # noqa: F401

ppg2.replace_ppg1()

from pypipegraph.testing.fixtures import (  # noqa:F401
    new_pipegraph,
)

data_folder = Path(__file__).parent / "data"
toml_file = data_folder / "run.toml"
toml_file_pype = data_folder / "run.pypipegraph.toml"

root = pathlib.Path(".").parent.parent
sys.path.append(str(root / "src"))


@pytest.fixture
def new_pipegraph_no_qc(new_pipegraph):
    ppg.util.global_pipegraph._qc_keep_function = False
    return new_pipegraph


@pytest.fixture
def ana():
    ans = analysis(toml_file)
    return ans


@pytest.fixture
@pytest.mark.usefixtures("new_pipegraph_no_qc")
def ana_pypipe():
    ans = analysis(toml_file_pype)
    return ans


@pytest.fixture
def toml_input():
    return read_toml(toml_file)
