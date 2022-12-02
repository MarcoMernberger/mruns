# -*- coding: utf-8 -*-

from pathlib import Path
from pandas import DataFrame
import pytest
import mbf
import mock
import unittest
from mruns.base import Analysis


__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


def test_analysis_parser(ana):
    assert ana.name == "Test"
    assert ana.analysis_type == "RNAseq"
    assert ana.main_incoming == Path("tests/data/").resolve()
    assert isinstance(ana.incoming, Path)
    assert str(ana.incoming) == "tests/data"
    assert isinstance(ana.path_to_samples_df, Path)
    assert ana.samples["df_samples"] == "samples.tsv"
    assert str(ana.path_to_samples_df) == "tests/data/samples.tsv"
    assert not ana.samples["reverse_reads"]
    assert ana.samples["kit"] == "QuantSeq"
    assert isinstance(ana.fastq_processor(), mbf.align.fastq2.UMIExtractAndTrim)
    assert ana.alignment["species"] == "Homo_sapiens"
    assert ana.alignment["revision"] == 98
    with mock.patch.object(
        mbf.externals.ExternalAlgorithm, "get_version_cached", new=lambda *args: "mocked"
    ):
        assert isinstance(ana.aligner()[0], mbf.externals.aligners.STAR)
    assert ana.genes["filter"]["canonical"]
    assert hasattr(ana.genes["filter"]["biotypes"], "__iter__")
    assert "protein_coding" in ana.genes["filter"]["biotypes"]
    assert ana.genes["filter"]["cpm_threshold"] == 1.0
    assert "ABpairs" in ana.comparison
    assert ana.comparison["ABpairs"]["method"] == "DESeq2Unpaired"
    assert ana.comparison["ABpairs"]["type"] == "ab"
    assert ana.comparison["ABpairs"]["file"] == "group.tsv"
    assert ana.comparison["ABpairs"]["method"] == "DESeq2Unpaired"
    assert isinstance(ana.comparison["ABpairs"]["filter_expressions"], list)
    assert [["FDR", "<=", "0.05"], ["log2FC", "|>", "1"]] in ana.comparison["ABpairs"][
        "filter_expressions"
    ]
    assert str(ana.path_to_combination_df) == "tests/data/combinations.tsv"
    assert isinstance(ana.comparison["ABpairs"]["parameters"], dict)
    assert hasattr(ana, "pathway_analysis")
    assert "ora" in ana.pathway_analysis
    assert "gsea" in ana.pathway_analysis
    collections = ["c6", "c7", "c2", "c5", "h", "ipa", "ipa_reg"]
    assert hasattr(ana.pathway_analysis["ora"]["collections"], "__iter__")
    unittest.TestCase().assertListEqual(
        list(ana.pathway_analysis["ora"]["collections"]), collections
    )
    unittest.TestCase().assertListEqual(
        list(ana.pathway_analysis["gsea"]["collections"]), collections[:-1]
    )
    assert ana.reports["name"] == "run_report"


def test_analysis_incoming(ana):
    assert isinstance(ana.incoming, Path)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_set_genome(ana_pypipe):
    ana = ana_pypipe
    assert hasattr(ana_pypipe, "genome")
    assert ana.genome.species == "Homo_sapiens"
    assert ana.genome.revision == "98"
    mouse = {"species": "Mus_musculus", "revision": 98, "aligner": "STAR"}
    ana2 = Analysis(
        "",
        ana.project,
        ana.samples,
        mouse,
        ana.genes,
        ana.comparison,
        ana.pathway_analysis,
        ana.reports,
        None,
    )
    assert ana2.genome.species == "Mus_musculus"
    assert ana2.genome.revision == "98"
    rat = {"species": "Rattus_norvegicus", "revision": 98, "aligner": "STAR"}
    ana2 = Analysis(
        "",
        ana.project,
        ana.samples,
        rat,
        ana.genes,
        ana.comparison,
        ana.pathway_analysis,
        ana.reports,
        None,
    )
    assert ana2.genome.species == "Rattus_norvegicus"
    with pytest.raises(ValueError):
        fail = {"species": "Biggusdickus", "revision": 98, "aligner": "STAR"}
        ana3 = Analysis(
            "",
            ana.project,
            ana.samples,
            fail,
            ana.genes,
            ana.comparison,
            ana.pathway_analysis,
            ana.reports,
            None,
        )
        ana3.set_genome()
        assert ana3.genome.species == "Biggusdickus"


def test_verfiy(ana):
    ana.samples["kit"] = "None"
    with pytest.raises(ValueError):
        ana._verify()
    ana.samples["df_samples"] = "notfound"
    with pytest.raises(FileNotFoundError):
        ana._verify()
    ana.project["run_ids"] = ["notfound"]
    with pytest.raises(FileNotFoundError):
        ana._verify()
    with pytest.raises(ValueError):
        del ana.project["analysis_type"]
        ana._verify()
    with pytest.raises(ValueError):
        del ana.project
        ana._verify()


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_in_pypipegraph(ana_pypipe):
    with mock.patch.object(
        mbf.externals.ExternalAlgorithm, "get_version_cached", new=lambda *args: "mocked"
    ):
        assert ana_pypipe.aligner()[1] == {}


def test_aligner(ana):
    with mock.patch.object(
        mbf.externals.ExternalAlgorithm, "get_version_cached", new=lambda *args: "mocked"
    ):
        assert ana.aligner()[1]["some"] == "parameter"
        ana.alignment["aligner"] = "None"
        with pytest.raises(ValueError):
            ana.aligner()


def test_combination_df(ana):
    assert isinstance(ana.combination_df(), DataFrame)
    del ana.combination["file"]
    assert ana.combination_df() is None


def test_verify_samples(ana):
    df = DataFrame(
        {
            "number": [1, 2],
            "sample": ["s1", "s2"],
            "comment": ["", ""],
            "prefix": ["a", "a"],
            "ABpairs": ["ABpairs", "ABpairs"],
            "vid": ["MM01", "MM02"],
        }
    )
    
    with mock.patch.object(pandas, "read_csv", new=lambda *args: df):
        with pytest.raises(ValueError) as info:
            ana.verify_samples()
            assert "The groups table" in str(info)
    del df["ABPairs"]
    with mock.patch.object(ana, "sample_df", new=lambda *args: df):
        with pytest.raises(FileNotFoundError) as info:
            ana.verify_samples()



    with mock.patch.object(ana, "sample_df", new=lambda *args: df):
        with pytest.raises(ValueError) as info:
            ana.verify_samples()
            assert "vids where assigned twice" in str(info)
    with mock.patch.object(ana, "sample_df", new=lambda *args: DataFrame({"a": [123]})):
        with pytest.raises(ValueError) as info:

            ana.verify_samples()
