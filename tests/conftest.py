# -*- coding: utf-8 -*-
"""
    conftest.py for mruns.
"""
from pathlib import Path
from mruns.base import get_analysis_from_toml
import pytest


@pytest.fixture
def analysis():
    data_folder = Path(__file__).parent / "data"
    toml_file = data_folder / "run.toml"
    analysis = get_analysis_from_toml(toml_file)
    return analysis