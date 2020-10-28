#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Contains some utility functions."""

from typing import Callable, List, Any
from pathlib import Path
from pandas import DataFrame
import shutil
import tomlkit
import numpy as np

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


def filter_function(
    threshold: float = 1,
    at_least: int = 1,
    canonical_chromosomes: List[str] = None,
    biotypes: List[str] = None,
) -> Callable:
    """
    Filter function for RNAseq used to filter genes instances prior to DE analysis.

    Filters for canonical chromosomes and biotypes. In addition, you can set a
    threshold on expression value columns. This is used to prefilter the genes
    before the actual DEG analysis to limit the set of genes we need to consider
    and exclude lowly expressed genes.

    Parameters
    ----------
    threshold : float
        Threshold for expression values to be considered as 'measured'.
    at_least : int
        Minimum number of samples that must meet the expression threshold (normalized).
    canonical_chromosomes : List[str]
        List of canonical chromosomes to consider.
    biotypes : List[str], optional
        Biotypes that are relevant, by default None.

    Returns
    -------
    Callable
        Filter function t be passed to genes.filter.
    """

    def get_filter(column_names: List[str] = None) -> Callable:
        """
        Takes a list of normalized expression column names and returns a filter
        function for genes.

        Parameters
        ----------
        column_names : List[str]
            Expression columns to filter for.

        Returns
        -------
        Callable
            Filter function that takes a DataFrame and returns a bool iterable.
        """

        def __filter(df):
            if threshold is not None:
                if column_names is None:
                    raise ValueError("No expression columns specified.")
                keep = (df[column_names] >= threshold).sum(axis="columns") >= at_least
            else:
                keep = np.ones(len(df), dtype=bool)
            if canonical_chromosomes is not None:
                chromosomes = [x in canonical_chromosomes for x in df["chr"].values]
                keep = keep & chromosomes
            if biotypes is not None:
                bio = [x in biotypes for x in df["biotype"].values]
                keep = keep & bio
            return keep

        return __filter

    return get_filter


def read_toml(req_file: Path):
    """
    Reads a tomlfile and returns a dict-like.

    Parameters
    ----------
    req_file : Path
        Path to toml file.

    Returns
    -------
    [type]
        [description]
    """
    with req_file.open("r") as op:
        p = tomlkit.loads(op.read())
    return p


def df_to_markdown_table(df: DataFrame) -> str:
    """
    Turns a DataFrame into a markdown table format string.

    Parameters
    ----------
    df : DataFrame
        The dataframe to convert.

    Returns
    -------
    str
        String representation of dataframe in markdown format.
    """
    ret = "|" + "|".join(list(df.columns.values)) + "|\n"
    ret += "|" + "|".join(["-" for x in df.columns]) + "|\n"
    for _, row in df.iterrows():
        ret += "|" + "|".join([str(x) for x in row]) + "|\n"
    return ret


def fill_incoming(run_ids):
    target_dirs = [Path(f"incoming/{run_id}") for run_id in run_ids]
    normal_path = Path("/rose/ffs/incoming")
    nextseq_path = Path("/rose/ffs/incoming/NextSeq")
    for path in target_dirs:
        path.mkdir(exist_ok=True, parents=True)
    #  sentinel_files = [target_dir / "sentinel.txt" for target_dir in target_dirs]

    def __check_path(path):
        for filename in path.iterdir():
            if str(filename).endswith("fastq.gz"):
                return True
        return False

    def copy_fastq(run_id: str, sentinelfile: Path):
        path_2_files = None
        run_folder = normal_path / run_id
        if run_folder.exists():
            if __check_path(run_folder / "Unaligned"):
                path_2_files = run_folder / "Unaligned"
            elif __check_path(run_folder / "Data" / "Intensities" / "BaseCalls"):
                path_2_files = run_folder / "Data" / "Intensities" / "BaseCalls"
            else:
                raise ValueError(f"No fastq files in {str(normal_path)}.")
        else:
            run_folder = nextseq_path / run_id
            if run_folder.exists():
                sub = run_folder / run_id
                alignments = []
                for s in sub.iterdir():
                    if s.name.startswith("Alignment"):
                        alignments.append(s)
                for ss in alignments[-1].iterdir():
                    path_2_files = ss / "Fastq"
                    break
                if not __check_path(path_2_files):
                    raise ValueError(f"No fastq files in {str(nextseq_path)}.")
        if path_2_files is None:
            raise ValueError("Did not find the path to the files.")
        # now we know the path to fastq files
        with sentinelfile.open("w") as sentinel:
            sentinel.write("copied:\n")
            for filename in path_2_files.iterdir():
                if filename.name.endswith("fastq.gz"):
                    shutil.copy(
                        filename, Path(f"incoming/{run_id}") / (filename.name + "_tmp"),
                    )
                    shutil.move(
                        f"incoming/{run_id}/" + filename.name + "_tmp",
                        f"incoming/{run_id}/" + filename.name,
                    )
                    sentinel.write(f"{filename.name}\n")

    for run_id in run_ids:
        sentinel_file = Path(f"incoming/{run_id}") / "sentinel.txt"
        if not sentinel_file.exists():
            copy_fastq(run_id, sentinel_file)

    # return ppg.MultiFileGeneratingJob(sentinel_files, copy_fastq)


def assert_uniqueness(list_of_objects: List[Any]):
    try:
        assert len(list_of_objects) == len(set(list_of_objects))
    except AssertionError:
        print(f"Duplicate objects passed: {list_of_objects}.")
        raise
