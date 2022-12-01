# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import mbf
import mock
import unittest

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
