#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Contains some utility functions."""

from typing import Callable, List, Any, Optional
from pathlib import Path
from pandas import DataFrame
import shutil
import tomlkit
import numpy as np
import logging
import glob


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


MACHINE_IDS = ["NB552003", "M03491", "M00754", "C00113"]


def is_run_id(putative_run_id: str) -> bool:
    """
    Check if a supplied string is a valid sequencer run id.

    Parameters
    ----------
    putative_run_id : str
        the run id to check.

    Returns
    -------
    bool
        True, if it is a valid run id.
    """
    if putative_run_id[0].isdigit():
        for mid in MACHINE_IDS:
            if mid in putative_run_id:
                return True
    return False


def locate_folder(run_id: str, main_incoming: Path) -> Path:
    """
    breadth first search for a run folder

    Parameters
    ----------
    run_id : str
        The run id of the run to find.
    main_incoming : Path
        The main incoming folder on our server.

    Returns
    -------
    Path
        _description_

    Raises
    ------
    ValueError
        If main incoming path with sequencer projects does not exist.
    ValueError
        If no path could be found.
    """
    if not main_incoming.exists():
        raise ValueError(
            f"Main incoming path '{str(main_incoming)}' not available, check server mounts."
        )
    folders_to_check = [main_incoming]
    while len(folders_to_check) > 0:
        folder = folders_to_check.pop(0)
        print(f"checking {folder} ...")
        logging.info(f"checking {folder} ...")
        f = folder / run_id
        print(f, f.exists())
        if f.exists():
            return f
        else:
            for sub in folder.iterdir():
                if sub.is_dir() and not is_run_id(sub.name):
                    folders_to_check.append(sub)
    raise ValueError(f"Path for {run_id} not found.")


def read_toml(req_file: Path) -> dict:
    """
    Reads a tomlfile and returns a dict-like.

    Parameters
    ----------
    req_file : Path
        Path to toml file.

    Returns
    -------
    dict
        dict-like representation of toml.
    """
    with req_file.open("r") as op:
        p = tomlkit.loads(op.read())
    return p


def dir_is_empty(directory: Path) -> bool:
    """
    dir_is_empty checks if a directory is empty.

    Parameters
    ----------
    directory : Path
        pathlib Path to directory.

    Returns
    -------
    bool
        True if any file is located in the directory.
    """
    return not any(directory.iterdir())


def is_fastq_folder(fastq_folder: Path) -> bool:
    """
    Checks if a folder contains fastq.gz files.

    Parameters
    ----------
    fastq_folder : Path
        Folder to be checked.

    Returns
    -------
    bool
        True, if fastq files present.
    """
    return len(glob.glob(str(fastq_folder) + "/*.fastq*")) > 0


def find_fastq_folder(run_folder: Path) -> Path:
    """
    Locates the fastq folder in a run folder.

    Parameters
    ----------
    run_folder : Path
        The run folder on the main server.

    Returns
    -------
    Path
        Path to fastq files.

    Raises
    ------
    ValueError
        if no folder with fastq files could be found.
    """
    # check NextSeq style
    ns_path = run_folder / run_folder.name
    if ns_path.exists():
        fastq_folders = [
            Path(p) for p in sorted(glob.glob(str(ns_path) + "/Alignment_*/*/Fastq"), reverse=True)
        ]
        for fastq_folder in fastq_folders:
            if is_fastq_folder(fastq_folder):
                return fastq_folder

    # check old style
    old_path = run_folder / "Unaligned"
    if old_path.exists() and is_fastq_folder(old_path):
        return old_path
    older_still = run_folder / "Data" / "Intensities" / "BaseCalls"
    if old_path.exists() and is_fastq_folder(older_still):
        return older_still
    raise ValueError(f"No fastq folder found in path: {run_folder}.")


def fill_incoming(run_ids: List[str], main_incoming: Path, incoming: Path):
    """
    Copies all the fastq files in the project folder.

    Parameters
    ----------
    run_ids : List[str]
        Run IDs needed for the project.
    main_incoming : Path
        Main path to sequencer runs.
    """

    def copy_fastq(run_folder: Path, sentinelfile: Path):
        target_path = sentinelfile.parent
        fastq_folder = find_fastq_folder(run_folder)
        files_to_copy = [Path(p) for p in glob.glob(str(fastq_folder / "*.fastq*"))]
        with sentinelfile.open("w") as sentinel:
            sentinel.write("copied:\n")
            for filename in files_to_copy:
                shutil.copy(
                    filename,
                    target_path / (filename.name + "_tmp"),
                )
                shutil.move(
                    target_path / (filename.name + "_tmp"),
                    target_path / filename.name,
                )
                sentinel.write(f"{filename.name}\n")

    ids_to_fetch = []
    import os

    for run_id in run_ids:
        target_path = incoming / run_id
        target_path.mkdir(exist_ok=True, parents=True)
        if dir_is_empty(target_path):
            ids_to_fetch.append(run_id)

    for run_id in ids_to_fetch:
        print(main_incoming)
        print(run_id)
        target_path = incoming / run_id
        sentinel_file = target_path / "info.txt"
        run_folder = locate_folder(run_id, main_incoming)
        if not sentinel_file.exists():
            copy_fastq(run_folder, sentinel_file)


def assert_uniqueness(list_of_objects: List[Any]):
    try:
        assert len(list_of_objects) == len(set(list_of_objects))
    except AssertionError:
        print(f"Duplicate objects passed: {list_of_objects}.")
        raise


def filter_function(
    threshold: float = 1,
    at_least: int = 1,
    canonical_chromosomes: Optional[List[str]] = None,
    biotypes: Optional[List[str]] = None,
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
