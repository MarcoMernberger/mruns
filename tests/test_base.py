# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import mbf_align
import mbf_externals

__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


def test_analysis_parser(ana):
    assert ana.name == "Test"
    assert ana.analysis_type == "RNAseq"
    assert isinstance(ana.incoming, Path)
    assert str(ana.incoming) == "tests/data"
    assert isinstance(ana.path_to_samples_df, Path)
    assert ana.samples["df_samples"] == "samples.tsv"
    assert str(ana.path_to_samples_df) == "tests/data/samples.tsv"
    assert not ana.samples["reverse_reads"]
    assert ana.samples["kit"] == "QuantSeq"
    assert isinstance(ana.fastq_processor(), mbf_align.fastq2.UMIExtractAndTrim) 
    assert ana.alignment["species"] == "Homo_sapiens"
    assert ana.alignment["revision"] == 98
    assert isinstance(ana.aligner()[0], mbf_externals.aligners.STAR)
    assert ana.genes["filter"]["canonical"]
    assert hasattr(ana.genes["filter"]["biotypes"], "__iter__")
    assert "protein_coding" in ana.genes["filter"]["biotypes"]
    assert ana.genes["filter"]["cpm_threshold"] == 1.0
    assert "DESeq2Unpaired" in ana.comparison 
    assert isinstance(ana.comparison["DESeq2Unpaired"]["filter_expressions"], list)
    assert [["FDR", "<=", "0.05"], ["log2FC", "|>", "1"]] in ana.comparison["DESeq2Unpaired"]["filter_expressions"]
    assert "pathway_analysis" in ana.downstream
    assert "ora" in ana.downstream["pathway_analysis"]
    assert "gsea" in ana.downstream["pathway_analysis"]
    assert ana.downstream["pathway_analysis"]["gsea"]["collections"] == ana.downstream["pathway_analysis"]["ora"]["collections"]
    assert ana.reports["name"] == "run_report"


def test_analysis_incoming(ana):
    assert isinstance(ana.incoming, Path)

