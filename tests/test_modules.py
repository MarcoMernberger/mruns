# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from mbf import align, genomics
import mbf
import pytest
import mock
import unittest
import mreports
import mdataframe
import mpathways
import pypipegraph2 as ppg2
import mruns
import numpy as np
from mruns.runner import Runner
from mruns.base import Analysis
from mbf.genomes import EnsemblGenome
from mock import MagicMock
from pandas import DataFrame
from mdataframe.differential import DESeq2Unpaired
from mdataframe import Filter
from mruns.modules import Module, InputHandler, VolcanoModule, PCAModule

__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


data_folder = Path(__file__).parent / "data"


class MockModule(Module):
    def __init__(self, name, inputs, caller=lambda: 5, **kẃargs):
        super().__init__(name, inputs, **kẃargs)
        self.name = name
        self._outputs = [f"file_for_{name}"]
        self.caller = caller

    def call(self):
        return self.caller()

    def check_inputs(self):
        pass

    def create_outputs(self):
        pass


def call_value():
    return "data"


class Test_Module:
    @pytest.fixture
    def inputs(self):
        mockinput = MockModule("InputMock", {}, lambda: 3)
        return {
            "path": data_folder / "combinations.tsv",
            "str": str(data_folder / "combinations.tsv"),
            "callable": call_value,
            "module": mockinput,
        }

    def test_module_init(self, inputs):
        testmodule = MockModule("BaseModule", inputs, parameter=1, some_other_parameter=2)
        assert hasattr(testmodule, "parameters")
        assert testmodule.parameters["parameter"] == 1
        assert testmodule.parameters["some_other_parameter"] == 2
        assert hasattr(testmodule, "inputs")
        assert hasattr(testmodule, "outputs")
        assert testmodule._outputs == ["file_for_BaseModule"]
        for input_name in testmodule.inputs:
            assert input_name in inputs
            assert callable(testmodule._inputs[input_name])

    def test_load_input(self, inputs):
        testmodule = MockModule("BaseModule", inputs, parameter=1, some_other_parameter=2)
        testmodule.load_input("path")
        assert isinstance(testmodule.path, DataFrame)
        testmodule.load_input("str")
        assert isinstance(testmodule.str, DataFrame)
        testmodule.load_input("callable")
        assert testmodule.callable == "data"
        testmodule.load_input("module")
        assert testmodule.module == 3
        inputs["not_readable"] = "somepath.clc"
        with pytest.raises(NotImplementedError) as info:
            testmodule = MockModule("BaseModule", inputs, parameter=1, some_other_parameter=2)
            assert "Need a method to load file type .clc" in str(info)
        inputs["notworking"] = 0.1
        with pytest.raises(NotImplementedError) as info:
            testmodule = MockModule("BaseModule", inputs, parameter=1, some_other_parameter=2)
            assert "Don't know how to read from type" in str(info)


class Test_VolcanoModule:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "logFC": [2, 1, -9, 0],
                "fdr": [0.1, 0.001, 0.4, 0.2],
                "p": [0.001, 0.003, 0.01, 0.01],
                "someother": [1, 2, 3, 4],
            },
            index=["A", "B", "C", "D"],
        )

    @pytest.fixture
    def module(self, frame, tmp_path):
        def caller_func():
            return frame

        outfile = tmp_path / "volcano.png"
        return VolcanoModule(outfile, {"df": caller_func})

    def test_verify_arguments(self, tmp_path):
        outfile = tmp_path / "volcano.png"
        with pytest.raises(ValueError) as info:
            VolcanoModule(outfile, {})
            assert "Module VolcanoModule inputs must contain a field named df." in str(info)

    def test_init(self, module, tmp_path, frame):
        outfile = tmp_path / "volcano.png"
        assert module.name == str(outfile)
        assert module.outputs == [
            outfile,
            outfile.with_suffix(".tsv"),
            outfile.with_suffix(".pdf"),
            outfile.with_suffix(".svg"),
        ]
        assert hasattr(module, "parameters")
        print(module.parameters)
        for col in ["fc_threshold", "logFC", "p", "fdr"]:
            assert col in module.parameters
        assert hasattr(module, "columns")

    def test_prepare_input(self, module):
        module.load()
        df = module.df
        wrong_cols = df.columns.difference(["logFC", "fdr", "p"])
        assert len(wrong_cols) == 0

    def test_outputs(self, module, tmp_path):
        module.run()
        for outfile in module._outputs:
            assert outfile.exists()

    def test_check_input(self, frame, tmp_path):
        del frame["logFC"]

        def caller_func():
            return frame

        outfile = tmp_path / "volcano.png"
        with pytest.raises(ValueError) as info:
            module = VolcanoModule(outfile, {"df": caller_func})
            module()
            print(str(info))
            assert "VolcanoModule Dataframe is missing the following columns: ['logFC']." in str(
                info
            )


class Test_PCAModule:
    @pytest.fixture
    def data(self):
        return pd.DataFrame(
            {
                "s1": [2, 1, -9, 0],
                "s2": [3, 4, 0.4, 0.2],
                "s3": [-3, 2, 2, 1],
                "s4": [1, 2, 3, 1],
            },
            index=["A", "B", "C", "D"],
            dtype=float,
        )

    @pytest.fixture
    def samples(self):
        return pd.DataFrame(
            {
                "Group": ["A", "A", "B", "B"],
            },
            index=["s1", "s2", "s3", "s4"],
        )

    @pytest.fixture
    def module(self, data, samples, tmp_path):
        def caller_func_data():
            return data

        def caller_func_samples():
            return samples

        outfile = tmp_path / "pca.png"
        module = PCAModule(outfile, {"df": caller_func_data, "df_samples": caller_func_samples})
        return module

    def test_init(self, tmp_path, data, samples, module):
        outfile = tmp_path / "pca.png"
        assert module.name == str(outfile)
        assert module.outputs == [
            outfile,
            outfile.with_suffix(".tsv"),
            outfile.with_suffix(".pdf"),
            outfile.with_suffix(".svg"),
        ]
        assert hasattr(module, "parameters")
        for col in ["n_components"]:
            assert col in module.parameters

    def test_prepare_input(self, data, tmp_path, module):
        module.load()
        df = module.df
        np.testing.assert_almost_equal(df.std(axis="columns"), np.ones(len(df)), decimal=0)
        np.testing.assert_almost_equal(df.mean(axis="columns"), np.zeros(len(df)))

    def test_outputs(self, data, tmp_path, module):
        outfile = tmp_path / "pca.png"
        module.run()
        for outfile in module._outputs:
            assert outfile.exists()

    def test_check_input(self, module, data, tmp_path):
        data.iloc[(0, 0)] = "wrong"
        module._inputs["df"] = lambda: data
        with pytest.raises(ValueError) as info:
            module.load()
            assert "PCA Dataframe contains non-float types." in str(info)
