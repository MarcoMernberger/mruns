#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mbf_compliance.py: Contains methods needed to work woth the Pypipgraph."""

import pypipegraph2 as ppg2
from pandas import DataFrame
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union
from pypipegraph2 import Job
from mbf.genomics.annotator import Annotator
from mdataframe import _Transformer
from copy import deepcopy
from mbf.genomics.genes import Genes
from collections import UserDict
from .modules import Module


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class GenesWrapper:
    def __init__(
        self,
        genes: Genes,
        tags: List[str] = [],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        GenesWrapper Constructor.

        Wrapper class for a mbf.genomics.Genes object. It encapsulates the genes
        object, takes care of generating the jobs so that the Modules working
        on the genes DataFrame do not have to take care of this.

        Parameters
        ----------
        genes : Genes
            The genes object.
        outpath : Path
            The result folder, where the genes object is stored.
        tags : List[str], optional
            Tags to label the genes object, e.g. for downstream analysis. The
            wrapper instance can be selected via the tags, by default [].
        name : Optional[str], optional
            Name of the GenesWrapper, by default None.
        """
        self.genes = genes
        self.genes_name = genes.name
        self.name = name if name is not None else f"GW_{self.genes_name}"
        self.path = self.genes.result_dir
        self.__modules: Dict[str, Module] = {}
        self.__dependencies: Dict[str, List[Job]] = {}
        self.__tags = set(tags)
        self.__tags.add(self.genes_name)
        self.__tags.add("all")
        self.decription = description

    @property
    def tags(self):
        return self.__tags

    def add_tag(self, tag: str):
        self.__tags.add(tag)

    @property
    def modules(self):
        return self.__modules

    @property
    def dependencies(self):
        return self.__dependencies

    def genes_df(self):
        return self.genes.df

    def jobify(self):
        self._module_jobs = {}
        for module_name in self.__modules:
            job = self.jobify_module(module_name)
            self._module_jobs[module_name] = job

    def get_module_job(self, module_name: str) -> Job:
        if not hasattr(self, "_module_jobs"):
            raise ValueError(f"{self.name} has no jobs. Call jobify first.")
        else:
            return self._module_jobs[module_name]

    def get_module_default_dependencies(self, module: Module) -> List[Job]:
        jobs = [
            ppg2.ParameterInvariant(f"PI_{module.name}, module.parameters", module.parameters),
            ppg2.ParameterInvariant(f"PI_{module.name}, module.description", module.description),
            ppg2.ParameterInvariant(f"PI_{module.name}, module.outputs", module.outputs),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.run", module.run),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.call", module.call),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.__call__", module.__call__),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.check_inputs", module.check_inputs),
            ppg2.FunctionInvariant(
                f"FI_{module.name}, module.create_outputs", module.create_outputs
            ),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.load", module.load),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.load_input", module.load_input),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.get_input", module.get_input),
            ppg2.FunctionInvariant(f"FI_{module.name}, module.verify_inputs", module.verify_inputs),
        ]
        if hasattr(module, "prepare_input_frame"):
            jobs.append(
                ppg2.FunctionInvariant(
                    f"FI_{module.name}, module.prepare_input_frame", module.prepare_input_frame
                )
            )
        return jobs

    def jobify_module(self, module_name: str) -> Job:
        module = self.__modules[module_name]
        dependencies = self.__dependencies[module_name]
        dependencies.extend(self.get_module_default_dependencies(module))
        outputs = module.outputs
        
        def __write(outputs):
            module.run()

        return ppg2.MultiFileGeneratingJob(outputs, __write).depends_on(dependencies)

    def jobs(self):
        jobs = []  # self.genes.load(), self.genes.write()[0]]
        if not hasattr(self, "_module_jobs"):
            self.jobify()
        jobs.extend(list(self._module_jobs.values()))
        return jobs

    def write(self, mangler_function: Optional[Callable] = None):
        output_filename = self.path / f"{self.genes.name}.tsv"
        self.genes.write(output_filename, mangler_function)

    def get_df_caller_func(self, genes_parameters: Dict[str, Any] = {}) -> Callable:
        """
        Returns a caller that loads the genes Dataframe and ensures a certain
        format.

        This function loads the Dataframe and applies all transormation functions
        specific for the genes object.

        Parameters
        ----------
        genes_parameters : Dict[str, Any]
            A dictionary with parameters for the genes Dataframe. These may include:
            columns : Optional[List[str]], optional
                The columns to show, by default None. None will select all columns.
            rename_columns : Optional[Dict[str, str]], optional
                A dictionary with columns names to be renamed, by default None
            sort_by : Optional[str], optional
                A column to sort by, by default None
            ascending : bool, optional
                Sort ascending or descening, by default False. If sort_by is None,
                this parameter is ignored.

        Returns
        -------
        Callable
            Caller for the gene Dataframe.
        """
        columns = genes_parameters.get("columns", None)
        rename_columns = genes_parameters.get("rename_columns", None)
        sort_by = genes_parameters.get("sort_by", None)
        ascending = genes_parameters.get("ascending", False)

        def get_ddf(*args):
            df = self.genes.df
            df = df.set_index("name")
            if sort_by is not None:
                df = df.sort_values(sort_by, ascending=ascending)
            if columns is not None:
                df = df[columns]
            if rename_columns is not None:
                df = df.rename(columns=rename_columns)
            return df

        return get_ddf

    def register_default_module(
        self,
        module_template: type[Module],
        module_args: List[Any],
        module_kwargs: Dict[str, Any],
        annotators: List[Annotator] = [],
        dependencies: List[Job] = [],
        genes_parameter: Dict[str, Any] = {},
    ):
        module = module_template(*module_args, **module_kwargs)
        module = self.adapt_module(module, genes_parameter)
        load_job = self.genes.load()
        if isinstance(load_job, Job):
            dependencies = dependencies + [load_job]
        anno_dependencies = [self.genes.add_annotator(annotator) for annotator in annotators]
        dependencies.extend(anno_dependencies)
        self.register_module(module, dependencies)

    def register_module(
        self,
        module: Module,
        dependencies: List[Job] = [],
    ):
        self.__modules[module.name] = module
        self.__dependencies[module.name] = dependencies

    def adapt_module(self, module: Module, genes_parameter: Dict[str, Any] = {}) -> Module:
        # set outputs
        module.outputs = self.fix_outputs(module.outputs)
        # set name
        module.name = f"{module.outputs[0].stem}"
        # set inputs
        # module.old_inputs = module.sources.copy()
        module.sources = self.fix_inputs(module.sources, genes_parameter)
        return module

    def fix_outputs(self, outputs: List[Path]) -> List[Path]:
        fixed_outputs = [self.fix_module_outpath(filename) for filename in outputs]
        return fixed_outputs

    def fix_module_outpath(self, filename: Path) -> Path:
        fixed_path = self.path / filename.parent / f"{self.genes.name}.{filename.name}"
        return fixed_path

    def fix_inputs(
        self, sources: Dict[str, Any], genes_parameter: Dict[str, Any]
    ) -> Dict[str, Union[Callable, Any, str, Path]]:
        if "df" in sources and sources["df"] is None:
            sources["df"] = self.get_df_caller_func(genes_parameter)
        return sources

    def column_rename_function(self, rename_columns: Optional[Dict[str, str]]) -> Callable:
        def __prepare_input_frame(df: DataFrame) -> DataFrame:
            return df.rename(columns=rename_columns)

        return __prepare_input_frame


class GenesCollection(UserDict):
    def __init__(self, *args, **kwargs):
        self.__tag_to_names = {}
        super().__init__(*args, **kwargs)
        for key in self.data:
            gw = self.data[key]
            self.__update_tags(key, gw)

    def __update_tags(self, key: str, genewrapper: GenesWrapper):
        for tag in genewrapper.tags:
            self.__update_tag(key, tag)

    def __update_tag(self, key: str, tag: str):
        if tag not in self.__tag_to_names:
            self.__tag_to_names[tag] = set()
        self.__tag_to_names[tag].add(key)

    def __setitem__(self, key, value):
        self.data[key] = value
        self.__update_tags(key, value)

    def genes_by_tag(self, tag):
        all_genes_with_tag = [self.data[key] for key in self.__tag_to_names[tag]]
        return all_genes_with_tag

    def register_module(self, module: Module, genewrapper_name: str, dependencies: List[Job] = []):
        self.data[genewrapper_name].register_module(module, dependencies)

    def add(self, genewrapper: GenesWrapper):
        name = genewrapper.name
        self.__setitem__(name, genewrapper)

    @property
    def tags(self):
        return list(set(self.__tag_to_names.keys()))

    @property
    def tags_to_names(self):
        return self.__tag_to_names

    def register_module_for_tag(self, tag: str, module: Module, dependencies: List[Job] = []):
        for genes_wrapper in self.genes_by_tag(tag):
            genes_wrapper.register_module(
                module,
                dependencies=dependencies,
            )

    def register_default_module_for_tag(
        self,
        tag: str,
        module_template: type[Module],
        module_args: List[Any],
        module_kwargs: Dict[str, Any],
        genes_parameter: Dict[str, Any],
        annotators: List[Annotator] = [],
        dependencies: List[Job] = [],
    ):
        for genes_wrapper in self.genes_by_tag(tag):
            genes_wrapper.register_default_module(
                module_template,
                deepcopy(module_args),
                deepcopy(module_kwargs),
                annotators=annotators,
                dependencies=dependencies,
                genes_parameter=genes_parameter,
            )


class FromTransformerWrapper(Annotator):
    def __init__(self, transformer_wrapper):
        """
        Adds a transformer output to genes.

        An annotator that annotates a gene object with an arbitrary transformer
        output.

        Parameters
        ----------
        transformer : _Transformer
            Transformer to use.
        input_columns : List[str]
        """
        self.wrapper = transformer_wrapper
        self.transformer = self.wrapper.transformer
        self.columns = self.transformer.columns
        self.input_columns = self.wrapper.input_columns

    def calc_ddf(self, ddf):
        """Calculates the ddf to append."""
        df_copy = ddf.df.copy()
        df_copy = df_copy.set_index("gene_stable_id")
        df_in = df_copy[self.input_columns]
        df_transformed = self.transformer(df_in)
        df_transformed = df_transformed.reset_index()
        df_transformed = df_transformed[self.columns]
        return df_transformed

    def deps(self, ddf):
        """Return ppg.jobs"""
        deps = self.wrapper.deps() + [
            ppg2.ParameterInvariant(
                f"{self.transformer.name}_{self.transformer.hash}", self.transformer.hash
            )
        ]
        return deps

    def dep_annos(self):
        return self.annotators

    def __freeze__(self):
        return "FromTransformerWrapper(%s)" % self.columns[0]


class DifferentialWrapper(Annotator):
    def __init__(
        self,
        name: str,
        comparison_group: str,
        transformer: _Transformer,
        counter: str,
        samples: List[str],
        input_columns: List[str],
        dependencies: List[Job],
        annotators: List[Annotator],
    ):
        """
        A convenience wrapper class to store the input columns, annotator group,
        samples, dependencies and output columns of a differential analysis.

        Parameters
        ----------
        name : str
            _description_
        comparison_group : str
            _description_
        transformer : _Transformer
            _description_
        counter : str
            _description_

        samples: List[str]
            sample names used.
        input_columns : List[str]
            _description_
        dependencies : List[Job]
            _description_
        """
        self.name = name
        self.comparison_group = comparison_group
        self.transformer = transformer
        self.counter = counter
        self.samples = samples
        self.input_columns = input_columns
        self.__dependencies = dependencies
        self.columns = self.transformer.columns
        self.annotators = annotators

    @property
    def dependencies(self):
        return self.__dependencies

    def __freeze__(self):
        return self.transformer.__freeze__()

    def __call__(self, *args):
        self.transformer.__call__(*args)

    def __getattr__(self, name):
        if hasattr(self.transformer, name):
            return object.__getattribute__(self.transformer, name)
        else:
            return object.__getattribute__(self, name)

    def calc_ddf(self, ddf):
        """Calculates the ddf to append."""
        df_copy = ddf.df.copy()
        df_copy = df_copy.set_index("gene_stable_id")
        df_in = df_copy[self.input_columns]
        df_transformed = self.transformer(df_in)
        df_transformed = df_transformed.reset_index()
        key_mapping = dict(zip(ddf.df["gene_stable_id"], ddf.df.index))
        df_transformed.index = df_transformed["gene_stable_id"].map(key_mapping).values
        df_transformed = df_transformed[self.columns]
        return df_transformed

    def deps(self, ddf) -> List[Job]:
        """Return ppg.jobs"""
        deps = self.dependencies + [
            ppg2.ParameterInvariant(
                f"{self.transformer.name}_{self.transformer.hash}", self.transformer.hash
            )
        ]
        return deps

    def dep_annos(self) -> List[Annotator]:
        return self.annotators
