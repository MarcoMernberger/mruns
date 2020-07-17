# -*- coding: utf-8 -*-

from mruns.base import Analysis, get_analysis_from_toml
from pathlib import Path
import pytest

__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


def test_analysis_parser(analysis):
    print(analysis)
    assert analysis.name == "20200520_Test"
    assert analysis.analysis_type == "RNAseq"
    assert isinstance(analysis.sample_file, Path)
    assert str(analysis.sample_file) == "incoming/samples.tsv"
    assert not analysis.reverse
    assert analysis.kit == "QuantSeq"
    assert analysis.species == "Homo_sapiens"
    assert analysis.revision == 98
    assert analysis.aligner == "STAR"
    assert analysis.canonical_chromosomes
    assert hasattr(analysis.biotypes, "__iter__")
    assert "protein_coding" in analysis.biotypes
    assert analysis.cpm_threshold == 1.0
    assert analysis.comparison_method == "DESeq2Unpaired"
    assert isinstance(analysis.filter_expressions, list)
    assert analysis.filter_expressions == [["FDR", "<=", "0.05"], ["log2FC", "|>", "1"]]
    assert analysis.directions == ["up", "down", "all"]
    assert "pathway_analysis" in analysis.downstream
    assert "ora" in analysis.downstream["pathway_analysis"]
    assert "gsea" in analysis.downstream["pathway_analysis"]
    assert analysis.downstream["pathway_analysis"]["gsea"]["collections"] == analysis.downstream["pathway_analysis"]["ora"]["collections"]
    assert analysis.run_report == "run_report"
