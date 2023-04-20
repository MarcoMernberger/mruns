#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""modules.py: Contains ...."""

import pandas as pd
import sklearn
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from mplots import save_figure
from pandas import DataFrame
from abc import ABC, abstractmethod
from mplots import volcano_plot, volcano_calc
from .inputs import InputHandler
from mplots.scatter import generate_dr_plot
from mdataframe.projection import PCA


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class Module(ABC):
    def __init__(self, name: str, inputs: Dict[str, Union[Callable, Any, str, Path]], **parameters):
        self.name = name
        self._parameters = parameters
        self._inputs = inputs
        self._outputs: List[Path] = []

    @property
    def inputs(self):
        return list(self._inputs.keys())

    @property
    def sources(self):
        return self._inputs

    @sources.setter
    def sources(self, inputs: Dict[str, Union[Callable, Any, str, Path]]):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[Path]):
        self._outputs = outputs

    @property
    def parameters(self):
        return self._parameters

    def verify_inputs(self):
        for attribute_name in self.required_inputs:
            if attribute_name not in self._inputs:
                raise ValueError(
                    f"Module {self.__class__.__name__} inputs must contain a field named {attribute_name}."
                )

    def get_input(self, input_name: str) -> Any:
        input_callable = InputHandler.get_load_callable(self._inputs[input_name])
        return input_callable()

    def load_input(self, input_name: str):
        value = self.get_input(input_name)
        setattr(self, input_name, value)

    def load(self):
        "ensure inputs are there"
        for input_name in self._inputs:
            self.load_input(input_name)
        self.check_inputs()
        if hasattr(self, "prepare_input_frame"):
            self.prepare_input()

    def prepare_input(self):
        self.df = self.prepare_input_frame(self.df)

    def __call__(self):
        self.load()
        return self.call()

    def run(self):
        results = self()
        if not isinstance(results, tuple):
            results = [results]
        self.create_outputs(*results)

    @abstractmethod
    def check_inputs(self):
        pass

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def create_outputs(self, *args):
        pass


class VolcanoModule(Module):

    required_inputs = ["df"]

    def __init__(
        self,
        outfile: Path,
        inputs: Dict[str, Union[Callable, Any]],
        name: Optional[str] = None,
        **parameters,
    ):
        name = str(outfile) if name is None else name
        super().__init__(name, inputs, **parameters)
        self._outputs = [
            outfile,
            outfile.with_suffix(".tsv"),
            outfile.with_suffix(".pdf"),
            outfile.with_suffix(".svg"),
        ]
        self.verify_inputs()
        self.set_default_parameters_parameters()
        self.columns = [self.parameters["logFC"], self.parameters["p"], self.parameters["fdr"]]

    def set_default_parameters_parameters(self):
        default_parameters = {
            "fc_threshold": self.parameters.pop("fc_threshold", 1),
            "alpha": self.parameters.pop("alpha", 0.05),
            "logFC": self.parameters.pop("logFC", "logFC"),
            "p": self.parameters.pop("p", "p"),
            "fdr": self.parameters.pop("fdr", "fdr"),
        }
        self._parameters.update(default_parameters)  # add remaining parameters

    def prepare_input_frame(self, df: DataFrame) -> DataFrame:
        return df[self.columns]

    def call(self):
        parameters = self.parameters
        title = parameters.pop("title", "Volcano")
        df_plot = volcano_calc(self.df, **parameters)
        f = volcano_plot(df_plot, title=title)
        return [f, df_plot]

    def create_outputs(self, f, df_plot):
        folder = self.outputs[0].parent
        filename_stem = self.outputs[0].stem
        save_figure(f, folder, filename_stem)
        df_plot.to_csv(self.outputs[1], index=False, sep="\t")

    def check_inputs(self):
        missing = pd.Index(self.columns).difference(self.df.columns)
        if len(missing) > 0:
            raise ValueError(
                f"Volcano Dataframe is missing the following columns: {missing.values}."
            )


class PCAModule(Module):

    required_inputs = ["df"]
    optional_inputs = ["df_samples"]

    def __init__(
        self,
        outfile: Path,
        inputs: Dict[str, Union[Callable, Any]],
        name: Optional[str] = None,
        **parameters,
    ):
        name = str(outfile) if name is None else name
        super().__init__(name, inputs, **parameters)
        self._outputs = [
            outfile,
            outfile.with_suffix(".tsv"),
            outfile.with_suffix(".pdf"),
            outfile.with_suffix(".svg"),
        ]
        self.verify_inputs()
        self.set_default_parameters_parameters()

    def set_default_parameters_parameters(self):
        default_parameters = {
            "n_components": self.parameters.pop("n_components", 2),
        }
        self.parameters.update(default_parameters)  # add remaining parameters

    def prepare_input_frame(self, df: DataFrame) -> DataFrame:
        df = df.astype(float)
        df = df.transform(sklearn.preprocessing.scale, axis="columns")
        return df

    def call(self):
        n_components = self.parameters["n_components"]
        title = self.parameters.pop("title", "PCA (n_components={n_components})")
        show_names = self.parameters.pop("show_names", True)
        pca = PCA(n_components=n_components)
        df_pca = pca(self.df)
        class_label_column = None
        if hasattr(self, "df_samples"):
            df_pca.index.rename("Sample", inplace=True)
            df_pca = df_pca.join(self.df_samples, how="left")
            class_label_column = "Group"
        f = generate_dr_plot(
            df_pca, class_label_column=class_label_column, title=title, show_names=show_names
        )
        return [f, df_pca]

    def create_outputs(self, f, df_plot):
        folder = self.outputs[0].parent
        filename_stem = self.outputs[0].stem
        save_figure(f, folder, filename_stem)
        df_plot.to_csv(self.outputs[1], index=False, sep="\t")

    def check_inputs(self):
        all_foat = all([dtype == float for dtype in self.df.dtypes])
        if not all_foat:
            raise ValueError("PCA Dataframe contains non-float types.")


class HeatmapModule:
    pass


"""
Genes
    - PCA
    - Distributions
    - add to report

Genes Differential
    - PCA
    - Distributions
    - Volcano
    - Heatmap
    - ORA

Genes Combinations
    - PCA
    - Heatmap
    - ORA

GSEA
    - html


GA --> Wrapper for genes


## requirements

Module --> A class that takes inputs, knows it outputs and generates them
    --> can be plugged into a Job
    --> can be used to create a snake Module


--> Generate tables, plots, any output file should be a model
--> needs to be independent of jobs or snake, just the raw functionality

## for the runner right now
We do analyses with different gene objects at some point
--> Heatmap
--> pCA
--> enrichments
--> Volcano

All these plots need a dataframe to work on as it is. So dataframe massaging occurs before the module.
Ideally, getting the dataframe is another module.

For genes/pypipegraph objects, we can wrap it in another wrapper class
    
GA:
Wrapper for genes
--> takes care of job generation
--> register_odule

Runner: 
- we want to keep track of all the genes in one dictionary to avoid all the for loops
- we want to tag them, to see, what output should be generated for each gene
- we need to keep track of dependencies?
register a single module for all genes
- register_module (class)
--> GA wraps genes


example: PCA
# 3-4 steps
# 1/2.  select the correct columns from the genes and scale ->
# 3. calculate PCA --> new df
# 4. plot

PCAscale(df, columns) -> df_scaled
PCAcalc(df_scaled) -> df_pca
PCAplot(df_pca)

mod = PCAplot(PCAcalc(PCAscale(df, columns)))  # 
self.gr.register_by_tag("filtered", "PCA")  -->


what i need:
GA should know it's genes "relevant" parameters
-> count_columns
-> norm columns
-> if differential: its deseq columns
-> if comparison: its sample names
-> if combined: 
    both sample names
    its ancestor columns
    more differential columns
-> its ancestors?

#
"""
