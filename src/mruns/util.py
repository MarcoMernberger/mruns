#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Contains some utility functions."""

from typing import Callable, List
from pathlib import Path
import tomlkit
import numpy as np

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


def filter_function(threshold: float = 1, at_least: int = 1, canonical_chromosomes: List[str] = None, biotypes: List[str] = None) -> Callable:
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
