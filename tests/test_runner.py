# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from pandas import DataFrame
from mbf import align, genomics
import pytest
import mbf
import mock
import logging
import mreports
import mdataframe
import mpathways
import pypipegraph2 as ppg
import mruns
import matplotlib
import matplotlib.pyplot as plt
from testfixtures import LogCapture
from mruns.runner import Runner
from mruns.base import Analysis
from mruns.mbf_compliance import DifferentialWrapper, GenesCollection
from mbf.genomes import EnsemblGenome
from mock import MagicMock
from mdataframe.differential import DESeq2Unpaired
from mdataframe import Filter
from conftest import mockgenome, MockGenome
from mruns.modules import VolcanoModule, PCAModule


__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


data_folder = Path(__file__).parent / "data"


def make_html(path):
    with path.open("w") as op:
        op.write(
            """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>title</title>
    <link rel="stylesheet" href="style.css">
    <script src="script.js"></script>
  </head>
  <body>
    <!-- page content -->
  </body>
</html>
"""
        )


class MockRawLane:
    def __init__(self, sample):
        self.name = sample

    def align(self, aligner, genome, aligner_parameters):
        return MockAligned(aligner, genome, aligner_parameters, self.name)


def make_plot(name, *args, **kwargs):
    plt.figure()
    plt.plot([1, 2, 3])
    plt.savefig(name)
    assert name.exists()


class MockAligned:
    def __init__(self, aligner, genome, params, name=None):
        self.aligner = aligner
        self.genome = genome
        self.params = params
        self.processed = False
        self.name = f"{name} aligned" if name is not None else "Mock aligned"

    def post_process(self, *args):
        self.processed = True
        return self


def patch_align(_, aligner, genome, params):
    return MockAligned(aligner, genome, params)


class MockCounter(mbf.genomics.annotator.Annotator):
    def __init__(self, col, val, *args, **kwargs):
        self.columns = [col]
        self.val = val

    def calc(self, ddf):
        return pd.Series(self.val)


class MockCounterRaw(MockCounter):
    def __init__(self, *args, **kwargs):
        sample = args[0].name.replace(" aligned", "")
        self.name = sample + " Counter"
        super().__init__(sample + " raw", [1, 15, 12, 2, 2, 2], *args, **kwargs)


class MockCounterNorm(MockCounter):
    def __init__(self, *args, **kwargs):
        sample = args[0].name.replace(" Counter", "")
        self.name = sample + " NormCounter"
        super().__init__(sample + " CPM", [0, 14.3, 5.6, 5, 5, 5], *args, **kwargs)


class MockDEseq:
    def __init__(self, columns_a, columns_b, comparison_name):
        self.name = f"{comparison_name} MockDEseq"
        self.hash = self.__freeze__()
        self.columns = [
            f"log2FC {self.name}",
            f"p {self.name}",
            f"FDR {self.name}",
            f"baseMean {self.name}",
            f"lfcSE {self.name}",
            f"stat {self.name}",
        ]
        self.FDR = f"FDR {self.name}"
        self.logFC = f"log2FC {self.name}"
        self.P = f"p {self.name}"

    def __freeze__(self):
        return self.name

    def __call__(self, df):
        return pd.DataFrame(0, columns=self.columns, index=df.index)


class MockDESeq2Timeseries:
    def __init__(
        self,
        **kwargs,
    ):
        self.columns = ["MockDSEseqTS"]
        self.name = "MockDSEseqTS"
        self.hash = "MockDEseqTS"

    def __freeze__(self):
        return "MockDEseqTS"

    def __call__(self, df):
        return pd.DataFrame(0, columns=self.columns, index=df.index)


class MockGSEA:
    def run_on_counts(self, *args, **kwargs):
        return args, kwargs


def mock_run_ora(arg):
    return arg


def return_mocklane(*args):
    return MockAligned(*args)


def return_mocklane_from_row(sample, row):
    return MockRawLane(sample)


def return_mock_diff(*args, **kwargs):
    return MockDEseq(*args, **kwargs)


@pytest.fixture
def patch_runner(ana):
    ana.alignment["parameters"] = {"myparam": "param"}
    runner = Runner(ana, log_level=logging.WARNING)
    with mock.patch.object(
        mbf.align.strategies.FASTQsJoin, "__call__", new=lambda *_, **__: "FASTQsJoin"
    ):
        with mock.patch.object(
            mbf.align.strategies.FASTQsFromPrefix,
            "__init__",
            new=lambda *_, **__: None,
        ):
            with mock.patch.object(
                mbf.align.raw.Sample, "register_qc", new=lambda *_: "registered"
            ):
                assert not hasattr(runner, "_raw_samples")
                runner.create_samples()
                assert hasattr(runner, "_raw_samples")
                for value in runner.raw_samples.values():
                    assert isinstance(value, mbf.align.raw.Sample)
    return runner


def test_runner_init(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    assert runner.analysis == ana
    assert runner.name == "DefaultRunner"
    assert isinstance(runner.fastq_processor, mbf.align.fastq2.UMIExtractAndTrim)
    with mock.patch.object(
        mbf.externals.ExternalAlgorithm, "get_version_cached", new=lambda *args: "mocked"
    ):
        assert isinstance(runner.aligner, mbf.externals.aligners.STAR)
    assert isinstance(runner.genome, mbf.genomes.ensembl._EnsemblGenome)
    assert isinstance(runner.fastq_processor, align.fastq2.UMIExtractAndTrim)
    assert hasattr(runner, "samples")
    assert not hasattr(runner, "raw_samples")
    assert "genomics.genes.anno_tag_counts.ExonSmartStrandedRust" in str(runner.raw_counter)
    assert isinstance(runner._normalizer, dict)
    assert "genomics.genes.anno_tag_counts.NormalizationCPM" in str(runner._normalizer["CPM"])
    assert "genomics.genes.anno_tag_counts.NormalizationTPM" in str(runner._normalizer["TPM"])
    df_combinations = pd.read_csv(data_folder / "combinations.tsv", sep="\t")
    assert runner.combinations.equals(df_combinations)
    assert isinstance(runner.gsea, mpathways.GSEA)


def test_runner_generate_combinations_none(ana):
    ana.combination = {}
    runner = Runner(ana, log_level=logging.WARNING)
    assert runner.combinations is None
    combined = runner.generate_combinations()
    assert combined == {}


def test_runner_generate_combinations(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    assert runner.combinations is not None
    with mock.patch.object(runner, "combine_genes", new=lambda *_, **__: "gene"):
        combined = runner.generate_combinations()
        print(combined)
        for genes in ["C_vs_D_A_or_B", "C_vs_D_A_and_B", "C_vs_D_A_only"]:
            assert genes in combined


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_runner_create_raw(ana_pypipe):
    runner = Runner(ana_pypipe, log_level=logging.WARNING)
    with mock.patch.object(
        mbf.align.strategies.FASTQsJoin, "__call__", new=lambda *_, **__: "FASTQsJoin"
    ):
        with mock.patch.object(
            mbf.align.strategies.FASTQsFromPrefix,
            "__init__",
            new=lambda *_, **__: None,
        ):
            with mock.patch.object(
                mbf.align.raw.Sample, "register_qc", new=lambda *_: "registered"
            ):
                assert not hasattr(runner, "_raw_samples")
                runner.create_samples()
                assert hasattr(runner, "_raw_samples")
                for value in runner.raw_samples.values():
                    assert isinstance(value, mbf.align.raw.Sample)


def test_runner_align(ana, patch_runner):
    runner_no_samples = Runner(ana, log_level=logging.WARNING)
    with pytest.raises(ValueError) as info:
        runner_no_samples.align()
        assert "No raw samples were defined. Call create_samples first." in str(info)
    with mock.patch.object(mbf.align.raw.Sample, "align", new=patch_align):
        assert not hasattr(patch_runner, "aligned_samples")
        assert patch_runner.postprocessor is not None
        patch_runner.align()
        assert hasattr(patch_runner, "aligned_samples")
        for aligned_lane in patch_runner.aligned_samples.values():
            assert aligned_lane.processed
            assert isinstance(aligned_lane.aligner, mbf.externals.aligners.STAR)
            assert aligned_lane.genome.species == "Homo_sapiens"
            assert "myparam" in aligned_lane.params
    with mock.patch.object(mbf.align.raw.Sample, "align", new=patch_align):
        patch_runner._postprocessor = None
        patch_runner.align()
        assert hasattr(patch_runner, "aligned_samples")
        for aligned_lane in patch_runner.aligned_samples.values():
            assert not aligned_lane.processed


def test_get_fastq_processor(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    assert isinstance(runner.fastq_processor, mbf.align.fastq2.UMIExtractAndTrim)
    ana.samples["kit"] = "NextSeq"
    runner = Runner(ana, log_level=logging.WARNING)
    assert isinstance(runner.fastq_processor, mbf.align.fastq2.Straight)
    with pytest.raises(NotImplementedError):
        ana.samples["kit"] = "None"
        Runner(ana)


def test_set_aligner(ana):
    ana.alignment["parameters"] = {"myparam": "param"}
    runner = Runner(ana, log_level=logging.WARNING)
    assert isinstance(runner.aligner, mbf.externals.aligners.STAR)
    assert "myparam" in runner._aligner_params
    with pytest.raises(ValueError) as info:
        ana.alignment["aligner"] = "Thisisnoaligner"
        Runner(ana, log_level=logging.WARNING)
        assert "No aligner" in str(info)


def test_genes(ana):
    runner = Runner(ana)
    assert isinstance(runner.genes, GenesCollection)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_runner_pype(ana_pypipe):
    runner = Runner(ana_pypipe, log_level=logging.WARNING)
    assert isinstance(runner.aligner, mbf.externals.aligners.STAR)
    assert isinstance(runner.fastq_processor, mbf.align.fastq2.Straight)
    assert isinstance(runner.genome, mbf.genomes.ensembl._EnsemblGenome)


def test_raw_counter(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    counter = runner.get_raw_counter_from_kit()
    assert "genomics.genes.anno_tag_counts.ExonSmartStrandedRust" in str(counter)
    runner.analysis.samples["stranded"] = False
    with pytest.raises(NotImplementedError):
        runner.analysis.samples["stranded"] = False
        runner.get_raw_counter_from_kit()
    runner.analysis.samples["kit"] = "NextSeq"
    runner.analysis.samples["stranded"] = True
    counter = runner.get_raw_counter_from_kit()
    assert "genomics.genes.anno_tag_counts.ExonSmartStrandedRust" in str(counter)
    runner.analysis.samples["stranded"] = False
    counter = runner.get_raw_counter_from_kit()
    assert "mbf.genomics.genes.anno_tag_counts.ExonSmartUnstrandedRust" in str(counter)
    with pytest.raises(NotImplementedError):
        runner.analysis.samples["kit"] = "Unknown"
        runner.analysis.samples["stranded"] = True
        runner.get_raw_counter_from_kit()


def test_norm_counter(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    with pytest.raises(NotImplementedError):
        runner.analysis.samples["stranded"] = False
        runner.get_norm_counter_from_kit()
    runner.analysis.samples["stranded"] = True
    counter = runner.get_norm_counter_from_kit()
    assert "genomics.genes.anno_tag_counts.NormalizationCPM" in str(counter)
    runner.analysis.samples["kit"] = "NextSeq"
    counter = runner.get_norm_counter_from_kit()
    assert "genomics.genes.anno_tag_counts.NormalizationTPM" in str(counter)
    with pytest.raises(NotImplementedError):
        runner.analysis.samples["kit"] = "Unknown"
        runner.get_norm_counter_from_kit()


def just_write(ddf, output_filename=None, mangler_function=None, float_format="%4g"):
    def __write(outfile):
        df = ddf.df.copy()
        df.to_csv(outfile, sep="\t", index=False)

    return ppg.FileGeneratingJob(output_filename, __write).depends_on(ddf.annotate())


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_run(ana_pypipe, tmpdir):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_pypipe, log_level=logging.WARNING)
        runner.create_samples()
        runner.raw_samples["sample"].align = MagicMock(side_effect=return_mocklane)
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(
            mbf.genomics.delayeddataframe.DelayedDataFrame, "write", new=just_write
        ):
            runner.write_genes_by_name("genes_all", tmpdir / "genes_test.tsv")
            runner.write_genes_by_name("genes_used", tmpdir / "genes_used_test.tsv")
        ppg.run()
        df = pd.read_csv(tmpdir / "genes_test.tsv", sep="\t")
        assert "Mock CPM" in df.columns
        assert "Mock raw" in df.columns


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_write_genes(ana_pypipe, tmpdir):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_pypipe, log_level=logging.WARNING)
        runner.create_samples()
        runner.raw_samples["sample"].align = MagicMock(side_effect=return_mocklane)
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(
            mbf.genomics.delayeddataframe.DelayedDataFrame, "write", new=just_write
        ):
            runner.write_genes()
        ppg.run()
        for genes in runner.genes.values():
            assert (genes.path / f"{genes.genes_name}.tsv").exists()


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_run_fail_norm(ana_pypipe):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        with pytest.raises(ValueError) as info:
            runner = Runner(ana_pypipe, log_level=logging.WARNING)
            runner.create_samples()
            runner.align()
            runner.normalize()
            ppg.run()
            assert "Normalization called before counting. Please call count() first" in str(info)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_run_fail_count(ana_pypipe):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        with pytest.raises(ValueError) as info:
            runner = Runner(ana_pypipe, log_level=logging.WARNING)
            runner.count()
            ppg.run()
            assert "Please call count after alignment" in str(info)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_set_genome(ana_pypipe):
    runner = Runner(ana_pypipe, log_level=logging.WARNING)
    assert hasattr(runner, "genome")
    assert runner.genome.species == "Homo_sapiens"
    assert runner.genome.revision == "98"
    mouse = {"species": "Mus_musculus", "revision": 98, "aligner": "STAR"}
    ana2 = Analysis(
        "",
        ana_pypipe.project,
        ana_pypipe.samples,
        mouse,
        ana_pypipe.genes,
        ana_pypipe.comparison,
        ana_pypipe.pathways,
        ana_pypipe.reports,
        {},
        {},
    )
    runner = Runner(ana2, log_level=logging.WARNING)
    assert runner.genome.species == "Mus_musculus"
    assert runner.genome.revision == "98"
    rat = {"species": "Rattus_norvegicus", "revision": 98, "aligner": "STAR"}
    ana2 = Analysis(
        "",
        ana_pypipe.project,
        ana_pypipe.samples,
        rat,
        ana_pypipe.genes,
        ana_pypipe.comparison,
        ana_pypipe.pathways,
        ana_pypipe.reports,
        {},
        {},
    )
    runner = Runner(ana2, log_level=logging.WARNING)
    assert runner.genome.species == "Rattus_norvegicus"
    with pytest.raises(ValueError):
        fail = {"species": "Biggusdickus", "revision": 98, "aligner": "STAR"}
        ana3 = Analysis(
            "",
            ana_pypipe.project,
            ana_pypipe.samples,
            fail,
            ana_pypipe.genes,
            ana_pypipe.comparison,
            ana_pypipe.pathways,
            ana_pypipe.reports,
            {},
            {},
        )
        runner = Runner(ana3)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_prefilter_run(ana_pypipe, tmpdir):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_pypipe, log_level=logging.WARNING)
        runner.create_samples()
        runner.raw_samples["sample"].align = MagicMock(side_effect=return_mocklane)
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(
            mbf.genomics.delayeddataframe.DelayedDataFrame, "write", new=just_write
        ):
            runner.write_genes_by_name("genes_all", tmpdir / "genes_test.tsv")
            runner.write_genes_by_name("genes_used", tmpdir / "genes_used_test.tsv")
        ppg.run()
        df_genes = pd.read_csv(tmpdir / "genes_test.tsv", sep="\t")
        df_genes_used = pd.read_csv(tmpdir / "genes_used_test.tsv", sep="\t")
        df_genes_used = df_genes_used.drop(labels="parent_row", axis="columns")
        assert len(df_genes) > len(df_genes_used)
        assert df_genes.columns.isin(df_genes_used.columns).all()
        assert df_genes_used.columns.isin(df_genes.columns).all()
        assert df_genes_used["gene_stable_id"].isin(["B", "C"]).all()


def test_get_samples_by_factors(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    runner._samples = pd.read_csv(data_folder / "samples.new.tsv", sep="\t")
    factors = ["factor1", "factor2"]
    values = ["condB", "condC"]
    samples = runner.get_samples_by_factors(factors, values)
    assert all(samples == ["S5", "S6"])


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_get_deg(ana_new, tmpdir):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        ts_transformer = MockDESeq2Timeseries()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            with mock.patch.object(
                mdataframe.differential,
                "DESeq2Timeseries",
                new=MagicMock(return_value=ts_transformer),
            ):
                runner.deg()
        with mock.patch.object(
            mbf.genomics.delayeddataframe.DelayedDataFrame, "write", new=just_write
        ):
            runner.write_genes_by_name("genes_used", tmpdir / "genes_used_test.tsv")
        ppg.run()
        df = pd.read_csv(tmpdir / "genes_used_test.tsv", sep="\t")
        for diff_anno in runner.differential.values():
            for column in diff_anno.columns:
                assert column in df
        assert len(df) == 2
        transformer_caller = runner.differential_wrapper_ab
        diffs = runner.get_differential_transformers("ABpairs", transformer_caller)
        names = ["C_vs_D(A)", "C_vs_D(B)"]
        for name, diff in diffs.items():
            assert isinstance(diff, DifferentialWrapper)
            assert name in names
        assert diffs["C_vs_D(A)"].columns == [
            "log2FC (C_vs_D(A))",
            "p (C_vs_D(A))",
            "FDR (C_vs_D(A))",
            "baseMean (C_vs_D(A))",
            "lfcSE (C_vs_D(A))",
            "stat (C_vs_D(A))",
        ]
        ts_diff = runner._differential["Multifactor_comparison"]
        assert ts_diff.columns == ["MockDSEseqTS"]


def test_get_comparison_samples_by_row(ana):
    ana.comparison["ABpairs"]["file"] = "group.new.tsv"
    ana.samples["file"] = "samples.new.tsv"
    runner = Runner(ana)
    df = pd.read_csv(data_folder / "group.new.tsv", sep="\t")
    row = df.loc[0]
    samples_a, samples_b = runner.get_comparison_samples_by_row(row)
    assert samples_a == ["S1", "S2"]
    assert samples_b == ["S3", "S4"]


def test_generate_transformer(ana):
    runner = Runner(ana, log_level=logging.WARNING)
    diff = runner.generate_transformer("ABpairs", [["sample1"], ["sample2"], "comparison"])
    assert isinstance(diff, DESeq2Unpaired)
    assert diff.columns_a == ["sample1"]
    assert diff.columns_b == ["sample2"]


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_gsea(ana_new):
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.analysis._incoming = Path("data")
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(mpathways.gsea.GSEA, "run_on_counts", new=MockGSEA.run_on_counts):
            args_caught = runner.run_gsea()
            for comp_name in [
                "C_vs_D(A)",
                "C_vs_D(B)",
            ]:
                assert comp_name in args_caught
                for collection_name in ["h", "c2", "h,c2"]:
                    assert collection_name in args_caught[comp_name]
            args, kwargs = args_caught["C_vs_D(A)"]["h,c2"]
            assert kwargs["collection"] == ["h", "c2"]
            assert args[0] == runner.genes_used
            assert kwargs["phenotypes"] == ["C", "D"]
            samples_expected = (["S1 CPM", "S2 CPM"], ["S3 CPM", "S4 CPM"])
            assert kwargs["columns_a_b"] == samples_expected
            assert kwargs["genome"] == runner.genome
            assert isinstance(kwargs["annotators"][0], MockCounterNorm)
            assert kwargs["phenotypes"] == ["C", "D"]

        ppg.run()


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_filter(ana_new, tmpdir):
    del ana_new.comparison["timeseries"]
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        ts_transformer = MockDESeq2Timeseries()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            with mock.patch.object(
                mdataframe.differential,
                "DESeq2Timeseries",
                new=MagicMock(return_value=ts_transformer),
            ):
                runner.deg()
                runner.filter()
        ppg.run()


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_ora(ana_new):
    del ana_new.comparison["timeseries"]
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            runner.deg()
        runner.filter()
        runner.combine()
        with mock.patch.object(mruns.defaults.ORA, "run_ora", new=mock_run_ora):
            ora = runner.run_ora(genes_to_select=None)
            assert ora.genes_used == runner.genes_used
            assert ora.analysis == ana_new
            for gw in runner.genes.genes_by_tag("filtered"):
                assert gw in ora.genes_to_analyze
            for gw in runner.genes.genes_by_tag("combined"):
                assert gw in ora.genes_to_analyze
        ppg.run()


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_pathways(ana_new, tmp_path):
    del ana_new.comparison["timeseries"]
    gsea_file = tmp_path / "mockrun.gsea.txt"
    ora_file = tmp_path / "mockrun.ora.txt"

    def mockrun(pathway_analysis):
        if pathway_analysis == "GSEA":
            outfile = gsea_file
        elif pathway_analysis == "ORA":
            outfile = ora_file
        else:
            raise NotImplementedError()

        def run(*args, **kwargs):
            print("run called")
            print(pathway_analysis, outfile)
            with (outfile).open("w") as out:
                out.write(pathway_analysis + "\n")
                for arg in args:
                    out.write(str(arg) + "\n")
                for key, value in kwargs.items():
                    out.write(f"{str(key)}:{str(value)}\n")
            print(outfile.exists())
            return args, kwargs

        return run

    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.analysis._incoming = Path("data")
        runner.logger.setLevel(logging.DEBUG)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            runner.deg()
        runner.filter()
        runner.combine()
        with mock.patch.object(mpathways.gsea.GSEA, "run_on_counts", new=mockrun("GSEA")):
            with mock.patch.object(mpathways.ora.ORAHyper, "run", new=mockrun("ORA")):
                runner.pathways()
        ppg.run()
        assert gsea_file.exists()
        assert ora_file.exists()
        with gsea_file.open("r") as inp:
            out = inp.read()
            assert "GSEA" in out
            assert (
                "Genes(Genes_Homo_sapiens_98_chr=canonical_biotypes_CPM1>1_drop_empty_names)" in out
            )
            assert "comparison_name:C_vs_D(B)" in out
            assert "phenotypes:['C', 'D']" in out
            assert "columns_a_b:(['S5 CPM', 'S6 CPM'], ['S7 CPM', 'S8 CPM'])" in out
            assert "collection:c2" in out
            assert "annotators:[Annotator(S1 CPM)" in out
            assert "median:True" in out
            assert "permutations:1000" in out
        with gsea_file.open("r") as inp:
            out = inp.read()
            assert (
                "Genes(Genes_Homo_sapiens_98_chr=canonical_biotypes_CPM1>1_drop_empty_names)" in out
            )


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_filtered_genes_to_analyze(ana_new):
    del ana_new.comparison["timeseries"]
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            runner.deg()
        with pytest.raises(ValueError) as info:
            runner.filtered_genes_to_analyze()
            assert "No filtered genes to perform ORA on. call filter()" in str(info)
        runner.filter()
        with LogCapture() as captured:
            runner.filtered_genes_to_analyze()
            msg = captured.records[0].getMessage()
            assert (
                "No set combinations to perform on. ORA is performed for filtered genes only."
                in msg
            )
        runner.combine()
        genes_to_analyze = runner.filtered_genes_to_analyze(["genes_used"])
        ppg.run()
        assert len(genes_to_analyze) == 1
        assert genes_to_analyze[0].genes_name == runner.genes_used.name


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_write_filtered(ana_new):
    del ana_new.comparison["timeseries"]
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            runner.deg()
        with pytest.raises(ValueError) as info:
            runner.filtered_genes_to_analyze()
            assert "No filtered genes to perform ORA on. call filter()" in info
        runner.filter()
        runner.combine()
        runner.write_filtered()
        runner.write_combined()
        ppg.run()
        for gw in runner.genes.genes_by_tag("combined"):
            output_filename = gw.path / f"{gw.genes.name}.tsv"
            assert output_filename.exists()
        for gw in runner.genes.genes_by_tag("filtered"):
            output_filename = gw.path / f"{gw.genes.name}.tsv"
            assert output_filename.exists()


def test_get_differential_for_group(ana):
    ana.comparison["ABpairs"]["type"] = "nontype"
    runner = Runner(ana)
    with pytest.raises(NotImplementedError) as info:
        runner.get_differential_for_group("ABpairs")
        assert "Cannot interpret comparison type" in str(info)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_register_volcano(ana_new, tmpdir):
    del ana_new.comparison["timeseries"]
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        ts_transformer = MockDESeq2Timeseries()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            with mock.patch.object(
                mdataframe.differential,
                "DESeq2Timeseries",
                new=MagicMock(return_value=ts_transformer),
            ):
                runner.deg()
                runner.filter()
        runner.register_volcano("filtered")
        ppg.run()
        for genes in runner.genes.genes_by_tag("filtered"):
            for comparison_name in runner.differential:
                module_name = f"{genes.genes_name}.{comparison_name}.volcano"
                assert module_name in genes.modules
                mod = genes.modules[module_name]
                assert isinstance(mod, VolcanoModule)
                assert (genes.path / f"{module_name}.png") in mod.outputs


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_register_pca(ana_new, tmpdir):
    del ana_new.comparison["timeseries"]

    def assert_module(module_name, genes):
        assert module_name in genes.modules
        mod = genes.modules[module_name]
        assert isinstance(mod, PCAModule)
        assert (genes.path / f"{module_name}.png") in mod.outputs

    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        ppg.new()
        runner = Runner(ana_new, log_level=logging.WARNING)
        runner.create_raw = return_mocklane_from_row
        runner.create_samples()
        runner.align()
        runner._raw_counter = MockCounterRaw
        runner.count()
        runner._normalizer = {"CPM": MockCounterNorm}
        runner.normalize()
        runner.prefilter()
        ts_transformer = MockDESeq2Timeseries()
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            with mock.patch.object(
                mdataframe.differential,
                "DESeq2Timeseries",
                new=MagicMock(return_value=ts_transformer),
            ):
                runner.deg()
                runner.filter()
        runner.register_pca("filtered", counter="CPM", comparisons=True)
        ppg.run()
        counter = "CPM"
        for genes in runner.genes.genes_by_tag("filtered"):
            for comparison_name in runner.differential:
                module_name = f"{genes.genes_name}.{comparison_name}.{counter}.pca"

                assert_module(module_name, genes)
            module_name = f"{genes.genes_name}.all.{counter}.pca"
            assert_module(module_name, genes)


@pytest.mark.usefixtures("new_pipegraph_no_qc")
def test_everything(ana_new, tmp_path):
    html = tmp_path / "test.html"
    gsea_job = ppg.FileGeneratingJob(tmp_path / "test.html", make_html)
    del ana_new.comparison["timeseries"]
    with mock.patch.object(Runner, "_get_genome", new=MagicMock(return_value=MockGenome())):
        with mock.patch.object(mdataframe.differential, "DESeq2Unpaired", new=return_mock_diff):
            with mock.patch.object(
                mpathways.gsea.GSEA, "run_on_counts", new=lambda *args, **kwargs: (gsea_job, html)
            ):
                with mock.patch.object(
                    mpathways.ora.ORAHyper,
                    "run",
                    new=lambda *args, **kwargs: ppg.FileGeneratingJob(
                        tmp_path / "path.png", make_plot
                    ),
                ):
                    with mock.patch.object(
                        mbf.genomics.delayeddataframe.DelayedDataFrame, "write", new=just_write
                    ):
                        with mock.patch.object(
                            mreports.htmlmod.GSEAReportPathModifier,
                            "job",
                            new=lambda _, __, deps: deps[0],
                        ):
                            ppg.new()
                            runner = Runner(ana_new, log_level=logging.WARNING)
                            runner.create_raw = return_mocklane_from_row
                            runner._raw_counter = MockCounterRaw
                            runner._normalizer = {"CPM": MockCounterNorm}
                            runner.everything()
                            ppg.run()
                        for genes in runner.genes.values():
                            assert (genes.path / f"{genes.genes_name}.tsv").exists()
