# -*- coding: utf-8 -*-
"""
    conftest.py for mruns.
"""

import pathlib
import sys
import pandas as pd
import os
import mbf
import mreports
import pytest
import pypipegraph2 as ppg2  # noqa: F401
import pypipegraph as ppg  # noqa: F401
from mruns.util import filter_function, read_toml
from mruns.base import analysis
from pathlib import Path

root = pathlib.Path(".").parent.parent
sys.path.append(str(root / "src"))
import mruns

sys.path.append(str(root.parent / "mreports" / "src"))

ppg2.replace_ppg1()

from pypipegraph.testing.fixtures import (  # noqa:F401
    new_pipegraph,
)

data_folder = Path(__file__).parent / "data"
toml_file = data_folder / "run.toml"
toml_file_pype = data_folder / "run.pypipegraph.toml"
toml_file_new = data_folder / "run.new.toml"

root = pathlib.Path(".").parent.parent
sys.path.append(str(root / "src"))


@pytest.fixture
def new_pipegraph_no_qc(new_pipegraph):
    ppg.util.global_pipegraph._qc_keep_function = False
    return new_pipegraph


@pytest.fixture
def ana():
    ans = analysis(toml_file)
    return ans


@pytest.fixture
@pytest.mark.usefixtures("new_pipegraph_no_qc")
def ana_new():
    ans = analysis(toml_file_new)
    return ans


@pytest.fixture
@pytest.mark.usefixtures("new_pipegraph_no_qc")
def ana_pypipe():
    ans = analysis(toml_file_pype)
    return ans


@pytest.fixture
def toml_input():
    return read_toml(toml_file)


class MockGenome:
    def __init__(self):
        self.name = "MockGenome"
        self.species = "Homo_sapiens"
        self.revision = 97
        self.df_genes = pd.DataFrame(
            {
                "chr": ["1", "1", "12", "2", "noncanonical", "1"],
                "tss": [1, 1, 3, 3, 1, 1],
                "tes": [14, 14, 12, 12, 14, 13],
                "strand": [1, 1, 1, 1, 14, 1],
                "biotype": [
                    "protein_coding",
                    "protein_coding",
                    "lincRNA",
                    "notype",
                    "protein_coding",
                    "protein_coding",
                ],
                "name": ["genA", "genB", "genC", "genD", "genE", ""],
            },
            pd.Index(data=["A", "B", "C", "D", "E", "F"], name="gene_stable_id"),
        )
        self.fasta = "data/genome.fasta"
        self.gtf = "data/genes.gtf"

    def name_to_gene_ids(self, name):
        return name

    def download_genome(self):
        return []

    def job_genes(self):
        return []

    def job_transcripts(self):
        return []

    def get_chromosome_lengths(self):
        return {"1": 20, "12": 20, "2": 12, "noncanonical": 12}

    def build_index(self, aligner, fasta_to_use=None, gtf_to_use=None):
        job = ppg.FileGeneratingJob("fake_index", lambda *_: Path("fake_index").write_text("hello"))
        job.output_path = "fake_index"
        return job

    def get_true_chromosomes(self):
        return ["1", "2", "12"]

    @property
    def genes(self):
        return self.df_genes


mockgenome = MockGenome()
