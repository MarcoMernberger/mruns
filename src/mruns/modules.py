#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""modules.py: Contains ...."""

import pandas as pd
import sklearn
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union
from mplots import save_figure
from pandas import DataFrame
from abc import ABC, abstractmethod
from mplots import volcano_plot, volcano_calc
from .inputs import InputHandler
from mplots.scatter import generate_dr_plot
from mplots.heatmaps import generate_heatmap_simple_figure
from mplots import plot_empty
from mdataframe.projection import PCA
from mdataframe.clustering import KMeans


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class Module(ABC):
    def __init__(
        self,
        name: str,
        inputs: Dict[str, Union[Callable, Any, str, Path]],
        description: Optional[str] = None,
        **parameters,
    ):
        self.name = name
        self._description = description
        if self._description is None:
            self.description = self.name
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
        value = input_callable()
        return value

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
        if not (isinstance(results, tuple) or isinstance(results, list)):
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

    @property
    def is_figure(self):
        for path in self.outputs:
            if path.suffix in [".png", ".svg", ".jpg"]:
                return True
        return False

    def wrap_prepare_input(
        self, functions: List[Callable], original_function: Optional[Callable] = None
    ) -> Callable:
        def new_prepare_input_frame(df: DataFrame):
            for func in functions:
                df = func(df)
            if original_function is not None:
                df = original_function(df)
            return df

        return new_prepare_input_frame

    def add_prepare_input_functions(self, functions: List[Callable]):
        if not hasattr(self, "prepare_input_frame"):
            self.prepare_input_frame = self.wrap_prepare_input(functions)
        else:
            self.prepare_input_frame = self.wrap_prepare_input(functions, self.prepare_input_frame)


class VolcanoModule(Module):
    required_inputs = ["df"]
    default_threshold = 1
    default_alpha = 0.05

    def __init__(
        self,
        outfile: Path,
        inputs: Dict[str, Union[Callable, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
        **parameters,
    ):
        name = str(outfile) if name is None else name
        super().__init__(name, inputs, description, **parameters)
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
            "fc_threshold": self.parameters.pop("fc_threshold", self.default_threshold),
            "alpha": self.parameters.pop("alpha", self.default_alpha),
            "logFC": self.parameters.pop("logFC", "logFC"),
            "p": self.parameters.pop("p", "p"),
            "fdr": self.parameters.pop("fdr", "fdr"),
        }
        self._parameters.update(default_parameters)  # add remaining parameters

    def prepare_input_frame(self, df: DataFrame) -> DataFrame:
        return df[self.columns]

    def call(self):
        parameters = self.parameters.copy()
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
        description: Optional[str] = None,
        **parameters,
    ):
        name = str(outfile) if name is None else name
        super().__init__(name, inputs, description, **parameters)
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
        if len(self.df) == 0:
            return [plot_empty(), pd.DataFrame()]
        n_components = self.parameters["n_components"]
        title = self.parameters.pop("title", f"PCA (n_components={n_components})")
        show_names = self.parameters.pop("show_names", True)
        pca = PCA(n_components=n_components)
        df_pca = pca(self.df)
        class_label_column = None
        if hasattr(self, "df_samples"):
            df_pca.index.rename("Sample", inplace=True)
            class_label_column = self.df_samples.columns[0]
            df_pca = df_pca.join(self.df_samples, how="left")
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


class HeatmapModule(Module):
    required_inputs = ["df"]

    def __init__(
        self,
        outfile: Path,
        inputs: Dict[str, Union[Callable, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
        **parameters,
    ):
        name = str(outfile) if name is None else name
        super().__init__(name, inputs, description, **parameters)
        self._outputs = [
            outfile,
            outfile.with_suffix(".tsv"),
            outfile.with_suffix(".pdf"),
            outfile.with_suffix(".svg"),
        ]
        self.verify_inputs()

    def prepare_input_frame(self, df: DataFrame) -> DataFrame:
        df = df.astype(float)
        df.index.rename("Sample", inplace=True)
        df = df.transform(sklearn.preprocessing.scale, axis="columns")
        # sort = self.parameters.pop("sort", False) # this should all be in a separate module!
        # if sort:
        #     add = self.parameters.pop("add", False)
        #     cluster_params = self.parameters.pop("cluster_params", {})
        #     kmeans = KMeans(**cluster_params)
        #     df = kmeans(
        #         df, add=add, sort=sort
        #     )  # cluster rows ... this should be a separate module!
        return df

    def prepare_input(self):
        self.df = self.prepare_input_frame(self.df)

    def load(self):
        "ensure inputs are there"
        for input_name in self._inputs:
            self.load_input(input_name)
        self.check_inputs()
        if hasattr(self, "prepare_input_frame"):
            self.prepare_input()

    def call(self):
        if 0 in self.df.shape:
            return [plot_empty(), pd.DataFrame()]
        else:
            parameters = self.parameters.copy()
            title = parameters.pop("title", "Heatmap (Z-scaled, KMeans)")
            f = generate_heatmap_simple_figure(
                self.df,
                title,
                **parameters,
            )
        return [f, self.df]

    def create_outputs(self, f, df_plot):
        folder = self.outputs[0].parent
        filename_stem = self.outputs[0].stem
        save_figure(f, folder, filename_stem)
        df_plot.to_csv(self.outputs[1], index=True, sep="\t")

    def check_inputs(self):
        all_foat = all([dtype == float for dtype in self.df.dtypes])
        if not all_foat:
            raise ValueError("PCA Dataframe contains non-float types.")
