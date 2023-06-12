# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from pandas import DataFrame
from mruns.modules import Module, InputHandler, VolcanoModule, PCAModule, HeatmapModule


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
        return {
            "path": data_folder / "combinations.tsv",
            "str": str(data_folder / "combinations.tsv"),
            "callable": call_value,
            "module": MockModule("InputMock", {}, lambda: 3),
        }

    def test_module_init(self, inputs):
        testmodule = MockModule("BaseModule", inputs, parameter=1, some_other_parameter=2)
        assert hasattr(testmodule, "caller")
        assert hasattr(testmodule, "outputs")
        assert hasattr(testmodule, "parameters")
        assert testmodule.parameters["parameter"] == 1
        assert testmodule.parameters["some_other_parameter"] == 2
        assert hasattr(testmodule, "inputs")
        assert "file_for_BaseModule" in testmodule.outputs
        for input_name in testmodule.inputs:
            assert input_name in inputs
            assert callable(InputHandler.get_load_callable(testmodule.sources[input_name]))

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
            testmodule.load_input("not_readable")
            assert "Need a method to load file type .clc" in str(info)
        inputs["notworking"] = 0.1
        with pytest.raises(NotImplementedError) as info:
            testmodule = MockModule("BaseModule", inputs, parameter=1, some_other_parameter=2)
            testmodule.load_input("not_readable")
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
        for outfile in module.outputs:
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
                "s1": [2, 1, -9, 0, 1],
                "s2": [3, 4, 0.4, 0.2, 2],
                "s3": [-3, 2, 2, 1, 3],
                "s4": [1, 2, 3, 1, 4],
            },
            index=["A", "B", "C", "D", "E"],
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

    def test_init(self, tmp_path, module):
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

    def test_prepare_input(self, module):
        module.load()
        df = module.df
        print(df.shape)
        np.testing.assert_almost_equal(df.std(axis="columns"), np.ones(df.shape[0]), decimal=0)
        np.testing.assert_almost_equal(df.mean(axis="columns"), np.zeros(df.shape[0]))

    def test_outputs(self, tmp_path, module):
        outfile = tmp_path / "pca.png"
        module.run()
        for outfile in module.outputs:
            assert outfile.exists()

    def test_check_input(self, module, data):
        data.iloc[(0, 0)] = "wrong"
        module.sources = {"df": lambda: data}
        with pytest.raises(ValueError) as info:
            module.load()
            assert "PCA Dataframe contains non-float types." in str(info)


class Test_HeatmapModule:
    @pytest.fixture
    def data(self):
        return pd.DataFrame(
            {
                "s1": [2, 5, 6, 1, 2],
                "s2": [3, 4, 3, 2, 3],
                "s3": [-3, 2, 2, -4, -3],
                "s4": [-1, 2, 3, -1, -1],
            },
            index=["A", "B", "C", "D", "E"],
            dtype=float,
        )

    @pytest.fixture
    def module(self, data, tmp_path):
        def caller_func_data():
            return data

        outfile = tmp_path / "map.png"
        module = HeatmapModule(
            outfile,
            {"df": caller_func_data},
            add=False,
            sort=True,
            cluster_params={"n_clusters": 2},
        )
        return module

    def test_init(self, tmp_path, module):
        outfile = tmp_path / "map.png"
        assert module.name == str(outfile)
        assert module.outputs == [
            outfile,
            outfile.with_suffix(".tsv"),
            outfile.with_suffix(".pdf"),
            outfile.with_suffix(".svg"),
        ]
        assert hasattr(module, "parameters")
        for col in ["add", "sort"]:
            assert col in module.parameters

    def test_prepare_input(self, module):
        """assert the data is z scaled"""
        module.load()
        df = module.df
        print(df)
        np.testing.assert_almost_equal(df.std(axis="columns"), np.ones(df.shape[0]), decimal=0)
        np.testing.assert_almost_equal(df.mean(axis="columns"), np.zeros(df.shape[0]))

    def test_outputs(self, tmp_path, module):
        outfile = tmp_path / "map.png"
        module.run()
        for outfile in module.outputs:
            assert outfile.exists()
        df = pd.read_csv(outfile.with_suffix(".tsv"), sep="\t", index_col=0)
        assert df.index.name == "Sample"
        assert df.index.difference(["A", "B", "C", "D", "E"]).empty

    def test_check_input(self, module, data):
        data.iloc[(0, 0)] = "wrong"
        module.sources = {"df": lambda: data}
        with pytest.raises(ValueError) as info:
            module.load()
            assert "Heatmap Dataframe contains non-float types." in str(info)
