# -*- coding: utf-8 -*-

from pathlib import Path
import pandas
from pandas import DataFrame
from mbf import align, genomics
import pytest
import mbf
import mock
import unittest
import mreports
from mruns.base import Analysis
from mruns import util

__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


def test_analysis_parser(ana):
    assert ana.name == "Test"
    assert ana.analysis_type == "RNAseq"
    assert ana.main_incoming == Path("tests/data/incoming").resolve()
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
    assert [["FDR", "<=", 0.05], ["log2FC", "|>", 1]] in ana.comparison["ABpairs"][
        "filter_expressions"
    ]
    assert str(ana.path_to_combination_df) == "tests/data/combinations.tsv"
    assert isinstance(ana.comparison["ABpairs"]["parameters"], dict)
    assert hasattr(ana, "pathways")
    assert "ora" in ana.pathways
    assert "gsea" in ana.pathways
    collections = ["c6", "c7", "c2", "c5", "h", "ipa", "ipa_reg"]
    assert hasattr(ana.pathways["ora"]["collections"], "__iter__")
    unittest.TestCase().assertListEqual(list(ana.pathways["ora"]["collections"]), collections)
    unittest.TestCase().assertListEqual(list(ana.pathways["gsea"]["collections"]), collections[:-1])
    assert ana.reports["name"] == "run_report"


def test_analysis_incoming(ana):
    assert isinstance(ana.incoming, Path)


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
            "ABpairs": ["groupA", "groupB"],
            "vid": ["MM01", "MM02"],
        }
    )
    with mock.patch.object(ana, "sample_df", new=lambda *__, **_: df):
        ana.verify_samples()
    with mock.patch.object(ana, "sample_df", new=lambda *args: df):
        ana.comparison["nonexistent"] = {"file": "group_nonexistent.tsv"}
        with pytest.raises(ValueError) as info:
            ana.verify_samples()
            print(str(info))
            assert "No factors and no grouping column" in str(info)
        ana.comparison["nonexistent"]["factors"] = ["condition"]
        with pytest.raises(ValueError) as info:
            ana.verify_samples()
            print(str(info))
            assert "is missing the following factors: ['condition']" in str(info)
        del ana.comparison["nonexistent"]
    df["vid"] = ["MM01", "MM01"]
    with mock.patch.object(ana, "sample_df", new=lambda *args: df):
        with pytest.raises(ValueError) as info:
            ana.verify_samples()
        assert "vids where assigned twice" in str(info)
    with mock.patch.object(ana, "sample_df", new=lambda *args: DataFrame({"a": [123]})):
        with pytest.raises(ValueError) as info:
            ana.verify_samples()
            assert "does not contain the following required columns" in str(info)


def fastq_processor(ana):
    processor = ana.fastq_processor()
    assert isinstance(processor, align.fastq2.UMIExtractAndTrim)
    ana.samples["kit"] = "NextSeq"
    processor = ana.fastq_processor()
    assert isinstance(processor, align.fastq2.Straight)
    ana.samples["kit"] = "Unknown kit"
    with pytest.raises(NotImplementedError):
        ana.fastq_processor()


def test_post_processor(ana):
    processor = ana.post_processor()
    assert isinstance(processor, align.post_process.UmiTools_Dedup)
    ana.samples["kit"] = "NextSeq"
    processor = ana.post_processor()
    assert processor is None
    ana.samples["kit"] = "Unknown kit"
    with pytest.raises(NotImplementedError):
        ana.fastq_processor()


def test_raw_counter(ana):
    counter = ana.raw_counter()
    assert "genomics.genes.anno_tag_counts.ExonSmartStrandedRust" in str(counter)
    ana.samples["stranded"] = False
    with pytest.raises(NotImplementedError):
        ana.samples["stranded"] = False
        ana.raw_counter()
    ana.samples["kit"] = "NextSeq"
    ana.samples["stranded"] = True
    counter = ana.raw_counter()
    assert "genomics.genes.anno_tag_counts.ExonSmartStrandedRust" in str(counter)
    with pytest.raises(NotImplementedError):
        ana.samples["stranded"] = False
        ana.raw_counter()
    with pytest.raises(NotImplementedError):
        ana.samples["kit"] = "Unknown"
        ana.samples["stranded"] = True
        ana.raw_counter()


def test_norm_counter(ana):
    with pytest.raises(NotImplementedError):
        ana.samples["stranded"] = False
        ana.norm_counter()
    ana.samples["stranded"] = True
    counter = ana.norm_counter()
    assert "genomics.genes.anno_tag_counts.NormalizationCPM" in str(counter)
    ana.samples["kit"] = "NextSeq"
    counter = ana.norm_counter()
    assert "genomics.genes.anno_tag_counts.NormalizationTPM" in str(counter)
    with pytest.raises(NotImplementedError):
        ana.samples["kit"] = "Unknown"
        ana.norm_counter()


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_report(ana_pypipe):
    report = ana_pypipe.report()
    assert isinstance(report, mreports.NB)
    assert report.name == "report"
    del ana_pypipe.reports["name"]
    report = ana_pypipe.report()
    assert report.name == "run_report"


def test_has_gene_filter_specified(ana):
    assert ana.has_gene_filter_specified()


def test_genes_filter(ana):
    with mock.patch.object(ana._genome, "get_true_chromosomes", new=lambda *args: [1, 2, 3]):
        assert callable(ana.genes_filter())
        ana.genes["filter"]["canonical"] = False
        assert callable(ana.genes_filter())
        ana.has_gene_filter_specified = lambda *_: False
        with pytest.raises(ValueError):
            ana.genes_filter()


def test_comparison_method(ana):
    deseq, params = ana.comparison_method(group_name="ABpairs", method="DESeq2Unpaired")
    assert isinstance(deseq, mbf.comparisons.methods.DESeq2Unpaired)
    assert isinstance(params, dict)
    with pytest.raises(ValueError):
        ana.comparison_method(group_name="ABpairs", method="Nomethod")


def test_deg_filter_expressions(ana):
    exprs = ana.deg_filter_expressions("ABpairs")
    default = [[["FDR", "<=", 0.05], ["log2FC", "|>", 1]]]
    assert exprs == [
        [["FDR", "<=", 0.05], ["log2FC", "|>", 1]],
        [["FDR", "<=", 0.05], ["log2FC", ">=", 1]],
        [["FDR", "<=", 0.05], ["log2FC", "<=", 1]],
    ]
    del ana.comparison["ABpairs"]["filter_expressions"]
    exprs = ana.deg_filter_expressions("ABpairs")
    assert exprs == default


def test_deg_filter_expression_as_str(ana):
    expr = ana.deg_filter_expression_as_str([["FDR", "<=", 0.05], ["log2FC", "|>", 1]])
    assert expr == "FDR<=0.05_log2FC|>1"


def test_pretty(ana):
    prettystring = ana.pretty()
    assert isinstance(prettystring, str)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_summary_markdown(ana_pypipe):
    with mock.patch.object(
        mbf.externals.ExternalAlgorithm, "get_version_cached", new=lambda *args: "mocked"
    ):
        md = ana_pypipe.summary_markdown()
        assert isinstance(md, str)
        assert "Set operations on comparisons" in md
        ana_pypipe.combination_df = lambda: None
        md = ana_pypipe.summary_markdown()
        assert isinstance(md, str)
        assert "Set operations on comparisons" not in md
        ana_pypipe.comparison["ABpairs"]["type"] = "multi"
        md = ana_pypipe.summary_markdown()
        assert "(multi)" in md
        with pytest.raises(ValueError) as info:
            ana_pypipe.comparison["ABpairs"]["type"] = "notype"
            md = ana_pypipe.summary_markdown()
            assert "Don't know what to do with type notype." in str(info)


def test_incoming(ana):
    del ana.project["incoming"]
    assert ana.incoming == Path("incoming")


def test_main_incoming(ana):
    del ana.project["main_incoming"]
    assert ana.main_incoming == Path("/rose/ffs/incoming")


def test_specification(ana):
    report_spec = ana.specification()
    print(report_spec)
    assert "### Specification" in report_spec


def test_combinations(ana):
    combinations = ana.combinations()
    assert hasattr(combinations, "__iter__")
    generators = []
    for x, y in zip(combinations, ["Union", "Intersection", "Set difference"]):
        assert x[-1] == y
        # generator_fnc
        generators.append(x[3])
    with mock.patch.object(genomics.genes.genes_from, "FromAny", lambda *_, **__: "FromAny called"):
        assert generators[0](None, None) == "FromAny called"
    with mock.patch.object(
        genomics.genes.genes_from, "FromIntersection", lambda *_, **__: "FromIntersection called"
    ):
        assert generators[1](None, None) == "FromIntersection called"
    with mock.patch.object(
        genomics.genes.genes_from, "FromDifference", lambda *_, **__: "FromDifference called"
    ):
        assert generators[2](None, [1, 2]) == "FromDifference called"
    with mock.patch.object(ana, "combination_df", lambda: None):
        ana.df_combinations = None
        combinations = ana.combinations()
        assert hasattr(combinations, "__iter__")
        list = [i for i in combinations]
        assert len(list) == 0


def test_parse_single_comparisons(ana):
    df = pandas.read_csv("tests/data/group.tsv", sep="\t")
    df["comparison_name"] = [df["comparison_name"].values[0]] * len(df)
    with mock.patch.object(ana, "get_comparison_group_table", lambda *_, **__: df):
        with pytest.raises(ValueError):
            ana.parse_single_comparisons("ABpairs", "DESeq2Unpaired", "ab", "mockpath")


def test_parse_comparisons(ana):
    comp_type = "this is no type"
    ana.comparison["ABpairs"]["type"] = comp_type
    with pytest.raises(ValueError) as info:
        ana.parse_comparisons()
        assert f"Don't know what to do with type {comp_type}." in str(info)
    ana.comparison["ABpairs"]["type"] = "multi"
    with mock.patch.object(
        ana,
        "parse_multi_comparisons",
        lambda *_, **__: {"msg": "parse_multi_comparisons called"},
    ):
        comparisons_to_do = ana.parse_comparisons()
        assert comparisons_to_do["ABpairs"]["msg"] == "parse_multi_comparisons called"


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_console_print(ana_pypipe):
    with mock.patch.object(
        mbf.externals.ExternalAlgorithm, "get_version_cached", new=lambda *args: "mocked"
    ):
        ana_pypipe.display_summary()
    # with mock.patch.object(console, "print", lambda *args: args):
    #     """
    #     Displays the summary of analysis on console.
    #     """
    #     md = self.summary_markdown()
    #     console.print(Markdown(md))
