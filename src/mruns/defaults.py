#!/usr/bin/env python
# -*- coding: utf-8 -*-
4
"""defaults.py: Contains standard plots and analyses."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, ClassVar, Union
from pandas import DataFrame
from mpathways import GSEA, GMTCollection, MSigChipEnsembl, MSigDBCollection, ORAHyper
from mdataframe import MDF, ClassLabel
from mbf.genomics.genes import Genes
from mbf.genomics.annotator import Annotator
from pypipegraph import Job
from mreports import MarkdownItem, PlotItem, HTMLItem, Item
from mplots import volcano
from .base import Analysis
from .util import assert_uniqueness
from .modules import VolcanoModule, PCAModule


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class RunDef:
    def __init__(self, runner):
        self.runner = runner

    def get_annos_deps_from_differential(
        self, comparison_name: str
    ) -> Tuple[List[Annotator], List[Job]]:
        differential = self.runner.differential[comparison_name]
        annotators = self.runner.differential_annotators[comparison_name]
        dependencies = differential.dependencies
        return annotators, dependencies

    def create_volcano_module(self, comparison_name: str, **parameters) -> VolcanoModule:
        differential = self.runner.differential[comparison_name]
        module_parameters = {
            "comparison_name": comparison_name,
            "fc_threshold": parameters.pop("fc_threshold", 1),
            "alpha": parameters.pop("alpha", 0.05),
            "logFC": differential.transformer.logFC,
            "p": differential.transformer.P,
            "fdr": differential.transformer.FDR,
        }
        module_parameters.update(parameters)
        columns = differential.input_columns
        outfile = Path(f"{comparison_name}.volcano.png")
        inputs = {"df": [columns]}
        return VolcanoModule(outfile, inputs, **module_parameters)

    def register_volcano(
        self, tag: str, comparison_names: Optional[List[str]] = None, **parameters
    ) -> None:
        if comparison_names is None:
            comparison_names = list(self.runner.differential.keys())
        for comparison_name in comparison_names:
            module = self.create_volcano_module(comparison_name)
            annotators, dependencies = self.get_annos_deps_from_differential(comparison_name)
            self.runner.genes.register_default_module_for_tag(
                tag,
                module,
                annotators=annotators,
                dependencies=dependencies,
            )

    def column_rename_function(self, rename_columns: Optional[Dict[str, str]]) -> Callable:
        def __prepare_input_frame(self, df: DataFrame) -> DataFrame:
            return df.rename(columns=rename_columns)

        return __prepare_input_frame

    def __get_counters(self, counter: Optional[Union[str, List[str]]] = None) -> List[str]:
        if counter is None:
            counters = list(self.runner.norm.keys())  # if no name specified, use all
        elif isinstance(counter, list):
            counters = counter
        elif isinstance(counter, str):
            counters = [counter]
        return counters

    def __get_samples_for_pca(
        self, comparisons: Optional[Union[bool, str, List[str]]] = False
    ) -> Dict[str, List[str]]:
        samples_to_plot = {"all": list(self.runner.raw_samples.keys())}
        if comparisons:
            if isinstance(comparisons, list):
                comparison_names = comparisons
            elif isinstance(comparisons, str):
                comparison_names = [comparisons]
            else:
                comparison_names = self.runner.differential.keys()
            for comparison_name in comparison_names:
                differential = self.runner.differential[comparison_name]
                samples = differential.samples
                samples_to_plot[comparison_name] = samples
        return samples_to_plot

    def create_pca_module(self, prefix: str, samples: List[str], counter_name: str) -> PCAModule:
        columns = [self.runner.columns_lookup[counter_name][sample] for sample in samples]
        rename = dict(zip(columns, samples))
        inputs = {
            "df": columns,
            "df_samples": self.runner.samples,
        }
        outfile = Path(f"{prefix}.{counter_name}.pca.png")
        module = PCAModule(outfile, inputs)
        module.prepare_input_frame = self.column_rename_function(rename_columns=rename)
        return module

    def register_pca(
        self,
        tag: str,
        counter: Optional[Union[str, List[str]]],
        comparisons: Optional[Union[bool, str, List[str]]] = False,
    ) -> None:
        """PCA can be done for any Normgroup ... register PCA for all columns and for differential columns"""
        counters = self.__get_counters(counter)
        samples_to_plot = self.__get_samples_for_pca(comparisons)
        for counter_name in counters:
            annotators = [annotator for annotator in self.runner.norm[counter_name]]
            for prefix in samples_to_plot:
                module = self.create_pca_module(prefix, samples_to_plot[prefix], counter_name)
                self.runner.genes.register_default_module_for_tag(
                    tag,
                    module,
                    annotators=annotators,
                    dependencies=[],
                )


class Defaults:
    def __init__(
        self,
        genes_used: Genes,
        analysis: Analysis,
        annotator_column_names_nice: Dict[str, str],
        normalized: Dict[str, Annotator],
        additional_annotators: List[Annotator] = None,
    ):
        """
        Collects some gene lists and relevant infos for standard plots and
        downstream analysis.

        Parameters
        ----------
        genes_used : Genes
            The main genes object used in the analysis.
        analysis : Analysis
            Tne Analysis instance for which these defaults are to be done.
        annotator_column_names_nice : Dict[str, str]
            A dictionary used to rename full column names to human friendly names.
        normalized : Dict[str, Annotator]
            A dictionary with all normalized expression values.
        additional_annotators : List[Annotator]
            A list of additional annotators the genes depend on.
        """
        self.column_names_nice = annotator_column_names_nice
        self.annotators = list(normalized.values())
        self.normalized = normalized
        if additional_annotators is not None:
            self.annotators.extend(additional_annotators)
        self.genes_parameters: Dict[str, Dict] = {}
        self.genes_used = genes_used
        self.analysis = analysis
        self.genes_to_analyze: Dict[str, Genes] = {}
        self.__prepare_ora()

    def register_genes(
        self,
        header: str,
        genes: Genes,
        expression_columns: List[str],
        condition_group: str,
        class_labels: ClassLabel,
        section: str = None,
        dependencies: List[Job] = None,
        comparison_annotators: Dict[str, Annotator] = None,
        volcano_plot_columns: Tuple[str] = None,
    ) -> None:
        """
        Register gene names and relevant infomration for the plots to be done.

        Parameters
        ----------
        genes : Genes
            The genes to be analyzed.
        expression_columns : List[columns]
            Expression columns relevant for this particular gene object.
        class_labels : ClassLabel
            A class label object for grouping the samples.
        condition_group : str
            The condition group this genes belongs to.
        dependencies : List[Job]
            List of job dependencies the genes depend on.
        comparison_annotators : Dict[str, Annotator]
            A dictionary of comparison annotators for volcano plots.
        volcano_plot_colums : str
            Tuple of volcanoplot columns.
        """
        self.genes_parameters[genes.name] = {}
        self.genes_parameters[genes.name]["expression_columns"] = expression_columns
        assert_uniqueness(expression_columns)
        self.genes_parameters[genes.name]["header"] = header
        if dependencies is not None:
            dependencies = []
        self.genes_parameters[genes.name]["dependencies"] = dependencies
        self.genes_parameters[genes.name]["class_label"] = class_labels
        self.genes_parameters[genes.name]["group"] = condition_group
        if comparison_annotators is None:
            comparison_annotators = {}
        self.genes_parameters[genes.name]["comparison_annotators"] = comparison_annotators
        self.genes_parameters[genes.name]["volcano_plot_columns"] = volcano_plot_columns
        self.genes_parameters[genes.name]["genes"] = genes
        self.genes_parameters[genes.name]["section"] = section
        self.genes_to_analyze[genes.name] = genes

    def run_standards(self) -> List[Item]:
        """
        Run standard plots and register them on the report NB.

        Runs standard plots and analyses for each registered gene list and
        adds them to the NB result, then returns the nb instance.

        Returns
        -------
        List[Item]
            List of items added to the notebook.
        """
        items = []
        for genes_name in self.genes_to_analyze:
            genes = self.genes_to_analyze[genes_name]
            md = MarkdownItem(
                self.genes_parameters[genes.name]["section"],
                self.genes_parameters[genes.name]["header"],
            )
            items.append(md)
            ml = MDF(
                f"ML_{genes.name}",
                genes,
                self.genes_parameters[genes.name]["expression_columns"],
                index_column="gene_stable_id",
                dependencies=self.genes_parameters[genes.name]["dependencies"],
                annotators=self.annotators,
            )
            ml = ml.impute().scale().cluster().sort("KNN_1")
            ml_plot = ml.transform(
                ["rename", {"columns": self.column_names_nice}],
                meta_rows=False,
                meta_columns=True,
            )
            plotjobs = ml_plot.plot_simple(
                outfile=Path(genes.result_dir) / f"{genes.name}_hm.png",
            ).plot
            pl = PlotItem(
                self.genes_parameters[genes.name]["section"],
                plotjobs,
                f"#### Heatmap for {genes.name}",
            )
            items.append(pl)
            pca = ml.reduce()
            pcas = [(pca, f"{genes.name}_pca.png", "")]
            all_samples = [ano.columns[0] for ano in self.normalized.values()]
            if (
                len(
                    set(all_samples).difference(
                        self.genes_parameters[genes.name]["expression_columns"]
                    )
                )
                > 0
            ):
                pca_all = (
                    MDF(
                        f"PCA_{genes.name}_all_samples",
                        genes,
                        all_samples,
                        index_column="gene_stable_id",
                        dependencies=[genes.add_annotator(anno) for anno in self.annotators]
                        + self.genes_parameters[genes.name]["dependencies"],
                        annotators=self.annotators,
                    )
                    .impute()
                    .scale()
                    .reduce()
                )
                pcas.append((pca_all, f"{genes.name}_all_samples_pca.png", " using all samples"))
            for p, filename, nbsuffix in pcas:
                p = p.cluster(self.genes_parameters[genes.name]["class_label"], axis=1)
                p = p.transform(
                    ["rename", {"columns": self.column_names_nice}],
                    meta_rows=False,
                    meta_columns=True,
                )
                plotjobs = p.plot_2d(
                    outfile=Path(genes.result_dir) / filename,
                    title=f"PCA {genes.name}",
                    class_label_column=self.genes_parameters[genes.name]["group"],
                    show_names=False,
                    model_name="PCA",
                )[0].plot
                pl = PlotItem(
                    self.genes_parameters[genes.name]["section"],
                    plotjobs,
                    f"#### Principle component analysis (PCA) on DE genes in {genes.name}"
                    + nbsuffix,
                )
                items.append(pl)
            for comp_name in self.genes_parameters[genes.name]["comparison_annotators"]:
                comparison_ab = self.genes_parameters[genes.name]["comparison_annotators"][
                    comp_name
                ]
                volcano_plot_columns = None
                if self.genes_parameters[genes.name]["volcano_plot_columns"] is not None:
                    volcano_plot_columns = self.genes_parameters[genes.name]["volcano_plot_columns"]
                elif (
                    "FDR" in comparison_ab.column_lookup and "log2FC" in comparison_ab.column_lookup
                ):
                    volcano_plot_columns = [
                        comparison_ab.column_lookup["FDR"],
                        comparison_ab.column_lookup["log2FC"],
                    ]
                if volcano_plot_columns is not None:
                    volcanos = volcano(
                        self.genes_used,
                        volcano_plot_columns[0],
                        volcano_plot_columns[1],
                        significance_threshold=0.05,
                        fc_threhold=1,
                        outfile=genes.result_dir / f"{genes.name}_volcano.png",
                        dependencies=[
                            self.genes_used.load(),
                            genes.add_annotator(comparison_ab),
                        ],
                        title=genes.name,
                    ).plot
                    pl = PlotItem(
                        self.genes_parameters[genes.name]["section"],
                        volcanos,
                        f"#### Volcano plot for DE genes in {genes.name} using {comp_name}",
                    )
                    items.append(pl)
        return items


class ORA:
    def __init__(self, genes_used, genes_to_analyze, analysis):
        self.genes_used = genes_used
        self.analysis = analysis
        self.genes_to_analyze = genes_to_analyze
        self.__prepare_ora()

    def __prepare_ora(self):
        hes = []
        for pathway_method in self.analysis.pathways:
            if pathway_method == "ora":
                collections = ["h"]
                if "collections" in self.analysis.pathways[pathway_method]:
                    collections = self.analysis.pathways[pathway_method]["collections"]
                # run ora
                for collection in collections:
                    he = ORAHyper(
                        name=f"ORA({collection})",
                        genome=self.genes_used.genome,
                        background_genes=self.genes_used,
                        collection=collection,
                    )
                    hes.append(he)
                if len(collections) > 1:
                    hes.append(
                        ORAHyper(
                            name="ORA(all)",
                            genome=self.genes_used.genome,
                            background_genes=self.genes_used,
                            collection=collections,
                        )
                    )
        self.hes = hes

    def get_oras(self):
        return self.hes

    def run_ora(self) -> Dict[str, Dict[str, Job]]:
        # downstream analyses
        items: Dict[str, Dict[str, Job]] = {}
        for genes_wrapper in self.genes_to_analyze:
            genes = genes_wrapper.genes
            genes_name = genes.name
            items[genes_name] = {}
            for he in self.get_oras():
                collection_name = he.collection.name
                job = he.run(genes)
                plotjob = he.plot_bars(job)
                # plotjobs = he.plot_bars(job).plot
                # pl = PlotItem(
                #     self.genes_parameters[genes.name]["section"],
                #     plotjobs,
                #     f"#### Over-Representation Analysis using hypergeometric test on DE genes from {genes.name} with collection {he.collection.name}",
                # )
                # items.append(pl)
                items[genes.name][collection_name] = (plotjob, job)
        return items
