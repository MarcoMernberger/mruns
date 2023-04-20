#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""factory.py: Contains default module factories."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from .mbf_compliance import GenesWrapper
from .modules import Module, PCA

Volcano
import pandas as pd
import pypipegraph as ppg

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class PCAFactory(Module):

    required_inputs = ["df"]
    optional_inputs = ["df_samples"]

    def __init__(self):
        pass

    def load_df(self, genes_wrapped, columns):
        def __load():
            df = genes_wrapped.genes_df()
            return df[columns]

        return __load

    def register_module(
        self, genes_wrapped: GenesWrapper, column_group: str, **parameters
    ) -> Module:
        outfile = genes_wrapped.path / f"{genes_wrapped.genes_name}_{column_group}.png"
        columns = genes_wrapped.columns[column_group]
        inputs = {
            "df": self.load_df(genes_wrapped, columns),
            "df_samples": None,
        }
        module = PCA(outfile=outfile, inputs=inputs, **parameters)
        return module

"""
Genes

    genes.register(PCA, counter_group)

    - PCA
    - Distributions
    - add to report

Genes Differential
    - PCA  -- for all norm groups
    - Distributions
    - Volcano  -- differential columns
    - Heatmap
    - ORA

Genes Combinations
    - PCA
    - Heatmap
    - ORA

GSEA
    - html



