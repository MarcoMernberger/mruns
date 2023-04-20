# -*- coding: utf-8 -*-

import pandas as pd
import pytest
import mbf
import unittest
import mreports
import mdataframe
import mpathways
import pypipegraph2 as ppg2
import mruns
import mock
from pathlib import Path
from pandas import DataFrame
from mbf import align, genomics
from mruns.runner import Runner
from mruns.base import Analysis
from mruns.mbf_compliance import GenesWrapper, GenesCollection
from mruns.modules import Module
from mbf.genomes import EnsemblGenome
from mbf.genomics.genes import Genes
from mbf.genomics import DelayedDataFrame
from mdataframe.differential import DESeq2Unpaired
from mdataframe import Filter
from conftest import mockgenome, MockGenome
from test_runner import just_write

__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


class MockModule(Module):
    def __init__(self, name, inputs):
        super().__init__(name, inputs)
        self.name = name
        self._outputs = [Path(f"file_for_{name}")]

    def check_inputs(self):
        pass

    def call(self):
        return pd.DataFrame({"1": [1, 2], "a": ["1", "2"]})

    def create_outputs(self, df, *args):
        df.to_csv(self._outputs[0])


genes = mbf.genomics.genes.Genes(MockGenome())


@pytest.fixture
def gw(tmp_path):
    return GenesWrapper(
        genes,
        tmp_path,
        tags=["main"],
        name="genes_used",
    )


@pytest.fixture
def gw2():
    genes2 = genes.filter("genes_filtered", lambda df: df.index)
    return GenesWrapper(
        genes2,
        Path("another"),
        tags=["filtered", "some_other"],
    )


@pytest.fixture
def gw3():
    genes3 = genes.filter("genes_filtered2", lambda df: df.index)
    return GenesWrapper(
        genes3,
        Path("another"),
        tags=["filtered"],
    )


@pytest.fixture
def module():
    return MockModule(name="MockModule", inputs={})


class Test_GenesWrapper:
    def test_geneswrapper_init(self, tmp_path, gw, gw2):
        assert gw.genes == genes
        assert gw.genes_name == "Genes_MockGenome"
        assert gw.name == "genes_used"
        assert "main" in gw.tags
        assert genes.name in gw.tags
        assert gw.path == tmp_path
        assert hasattr(gw, "modules")
        assert hasattr(gw, "dependencies")
        assert gw2.name == "GW_genes_filtered"
        assert gw2.tags == ["filtered", "some_other", gw2.genes_name]

    def test_geneswrapper_add_tag(self, gw):
        gw.add_tag("another tag")
        assert "another tag" in gw.tags

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_genes_df(self, gw):
        print(type(gw.genes_df()))
        assert isinstance(gw.genes_df(), DataFrame)

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_write(self, gw, tmp_path):
        gw.path = tmp_path
        ppg2.new()
        with mock.patch.object(
            mbf.genomics.delayeddataframe.DelayedDataFrame, "write", new=just_write
        ):
            gw.write()
        ppg2.run()
        assert (gw.path / f"{gw.genes.name}.tsv").exists()

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_get_df_caller_func(self, gw):
        columns = ["chr", "tss"]
        caller = gw.get_df_caller_func(columns=columns)
        df = caller()
        assert len(df.columns.difference(columns)) == 0
        caller = gw.get_df_caller_func()
        df = caller()
        assert "tes" in df.columns

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_register_default_module(self, gw, module):
        module.sources = {"df": ["chr"]}
        with mock.patch.object(
            mbf.genomics.genes.Genes, "add_annotator", new=lambda self, anno: anno
        ):
            gw.register_default_module(module, annotators=["mock"], dependencies=["job"])
        fixed_module = gw.modules["MockModule"]
        assert fixed_module.outputs[0].parent == gw.path
        assert fixed_module.outputs[0].name == f"{gw.genes_name}.file_for_MockModule"
        assert callable(fixed_module.sources["df"])
        df = fixed_module.sources["df"]()
        assert df.columns == ["chr"]
        assert len(gw.dependencies["MockModule"]) == 2
        module.sources = {"df": None}
        gw.register_default_module(module, dependencies=[])
        fixed_module = gw.modules["MockModule"]
        assert callable(fixed_module.sources["df"])
        df = fixed_module.sources["df"]()
        assert "tss" in df.columns

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_register_module(self, gw, module):
        dependencies = [ppg2.FileGeneratingJob("outfile", lambda _: 3)]
        gw.register_module(module, dependencies)
        assert gw.modules["MockModule"] == module
        assert gw.dependencies["MockModule"] == dependencies

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_jobify_module(self, gw, module, tmp_path):
        module._outputs = [tmp_path / Path(f"file_for_mock.csv")]
        dependencies = [ppg2.FileGeneratingJob("outfile", lambda _: 3)]
        gw.register_module(module, dependencies)
        mod_job = gw.jobify_module("MockModule")
        assert isinstance(mod_job, ppg2.Job)
        jobs = gw.jobify()
        assert jobs[0] == mod_job
        assert len(jobs) == 1
        jobs = gw.jobs()
        assert len(jobs) == 3

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_geneswrapper_jobify_module_create(self, gw, module, tmp_path):
        module._outputs = [tmp_path / Path(f"file_for_mock.csv")]
        ppg2.new()
        gw.register_module(module, dependencies=[])
        mod_job = gw.jobify_module("MockModule")
        assert isinstance(mod_job, ppg2.Job)
        ppg2.run()
        assert (module._outputs[0]).exists()


class Test_GenesCollection:
    def test_genes_collection(self, gw, gw2, gw3):
        collection = GenesCollection()
        collection2 = GenesCollection()
        collection["first"] = gw
        collection2["second"] = gw2
        collection2["third"] = gw3
        collection.update(collection2)
        assert collection["first"] == gw
        assert collection["second"] == gw2
        assert collection["third"] == gw3
        filtered = collection.genes_by_tag("filtered")
        assert len(filtered) == 2
        assert gw2 in filtered
        assert gw3 in filtered
        main = collection.genes_by_tag("main")
        assert len(main) == 1
        assert main[0] == gw

    def test_genes_collection_add(self, gw):
        collection = GenesCollection()
        assert len(collection) == 0
        collection.add(gw)
        assert collection[gw.name] == gw
        main = collection.genes_by_tag("main")
        assert main == [gw]

    def test_genes_collection_init(self, gw):
        gw.add_tag("filtered")
        collection = GenesCollection({gw.name: gw})
        assert "filtered" in collection.tags
        assert gw.genes.name in collection.tags
        assert "genes_used" in collection.tags_to_names["filtered"]

    @pytest.mark.usefixtures("new_pipegraph_no_qc")
    def test_collection_register_module(self, gw, module):
        dependencies = [ppg2.FileGeneratingJob("outfile", lambda _: 3)]
        collection = GenesCollection()
        collection[gw.name] = gw
        collection.register_module(
            module=module, genewrapper_name=gw.name, dependencies=dependencies
        )
        gw_reg = collection[gw.name]
        assert gw_reg.modules["MockModule"] == module
        assert gw_reg.dependencies["MockModule"] == dependencies

    def test_register_module_for_tag(self, gw, gw2, gw3, module):
        collection = GenesCollection()
        collection["first"] = gw
        collection["second"] = gw2
        collection["third"] = gw3
        collection.register_module_for_tag(module=module, tag="filtered", dependencies=["mock"])
        for gw_reg in collection.genes_by_tag("filtered"):
            assert gw_reg.modules["MockModule"] == module
            assert "mock" in gw_reg.dependencies["MockModule"]

    def register_default_module_for_tag(self, gw, gw2, gw3, module):
        collection = GenesCollection()
        collection["first"] = gw
        collection["second"] = gw2
        collection["third"] = gw3
        collection.register_default_module_for_tag(
            module=module, tag="filtered", dependencies=["mock"]
        )
        for gw_reg in collection.genes_by_tag("filtered"):
            assert gw_reg.modules["MockModule"] == module
            assert "mock" in gw_reg.dependencies["MockModule"]
