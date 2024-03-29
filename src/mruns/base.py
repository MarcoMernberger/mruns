#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""base.py: Contains toml parser and basic data class.
Todo: 
-   Annotate
-   Add more checks
-   filetimeinivariants
"""

from mruns import __version__
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Tuple, Any, ClassVar, Iterator
from mreports import NB
from mbf import genomes
from mbf import genomics
from mbf import align
from mbf import comparisons
from mbf.genomes import EnsemblGenome, Homo_sapiens, Mus_musculus
from mbf.externals.aligners.base import Aligner
from mbf.genomics.genes.anno_tag_counts import _NormalizationAnno, _FastTagCounter
from rich.markdown import Markdown
from rich.console import Console
from pandas import DataFrame
from .util import (
    filter_function,
    read_toml,
    df_to_markdown_table,
    fill_incoming,
    assert_uniqueness,
)
from pprint import PrettyPrinter
import mdataframe
import dataclasses
import pandas as pd
import pypipegraph as ppg
import tomlkit
import logging
import sys
import inspect
import json


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


_logger = logging.getLogger(__name__)
console = Console()

_required_fields = {
    "project": ["name", "analysis_type", "run_ids"],
    "samples": ["file", "reverse_reads", "kit", "stranded"],
    "alignment": ["species", "revision", "aligner"],
    "genes": [],
    "comparison": [],
    "reports": [],
}


@dataclass
class Analysis:
    run_toml: Path
    project: Dict[str, Any]
    samples: Dict[str, Any]
    alignment: Dict[str, Any]
    genes: Dict[str, Any]
    comparison: Dict[str, Any]
    pathways: Dict[str, Any]
    reports: Dict[str, Any]
    combination: Dict[str, Any]
    normalization: Dict[str, Any]
    __allowed_types: ClassVar[List[str]] = ["RNAseq"]
    __known_kits: ClassVar[List[str]] = ["QuantSeq", "NextSeq"]
    # __accepted_species: ClassVar[List[str]] = ["Homo_sapiens", "Mus_musculus"]

    @property
    def incoming(self):
        """
        The incoming folder with all external data.
        """
        return self._incoming

    def set_incoming(self):
        incoming = Path("incoming")
        if "incoming" in self.project:
            incoming = Path(self.project["incoming"])
        self._incoming = incoming

    @property
    def main_incoming(self):
        """
        The folder with all runs.
        """
        return self._main_incoming

    def set_main_incoming(self):
        incoming = Path("/rose/ffs/incoming")
        if "main_incoming" in self.project:
            incoming = Path(self.project["main_incoming"]).resolve()
        self._main_incoming = incoming

    def set_outpath(self):
        if "outpath" in self.project:
            self._outpath = Path(self.project["outpath"])
        else:
            self._outpath = Path("results")

    @property
    def analysis_type(self):
        """
        The analysis type to perform.
        """
        return self.project["analysis_type"]

    @property
    def name(self):
        """
        The project name.
        """
        return self.project["name"]

    @property
    def run_ids(self):
        """
        A list of all run_ids from the sequencer.
        """
        return self.project["run_ids"]

    @property
    def path_to_samples_df(self):
        return self.filepath_from_incoming(self.samples["file"])

    @property
    def path_to_combination_df(self):
        filename = self.combination.get("file", None)
        if filename is None:
            return None
        return self.filepath_from_incoming(filename)

    @property
    def factors(self) -> List[str]:
        """
        Returns the factors for the given samples.
        """
        if "factors" in self.samples:
            return list(self.samples["factors"])
        return None

    @property
    def outpath(self):
        return self._outpath

    @classmethod
    def comparison_method_name(cls, comparison_dict_from_toml: Dict) -> str:
        """
        Returns the name of the comparison method class for a DEG comparison.
        If none is given, it sets a default value.

        Parameters
        ----------
        comparison_dict_from_toml : Dict
            Dict-like toml entry for comparison.

        Returns
        -------
        str
            Name of the comparison method.
        """
        return comparison_dict_from_toml.get("method", "DESeq2Unpaired")

    @classmethod
    def pathway_collections(cls, dict_pathway_analysis: Dict) -> List[str]:
        """
        Returns the collections for a pathway analysis.
        If none is given, it sets a default value.

        Parameters
        ----------
        dict_pathway_analysis : Dict
            Dict-like toml entry for comparison.

        Returns
        -------
        List[str]
            List of collections to be used.
        """
        return list(dict_pathway_analysis.get("collections", ["h"]))

    # @classmethod
    # def get_comparison_methods(cls):
    #     module = sys.modules["comparisons.methods"]
    #     ret = {}
    #     for name, obj in inspect.getmembers(module):
    #         if inspect.isclass(obj):
    #             ret[name] = obj
    #     return ret

    def __post_init__(self):
        """
        Cleans up after initialization.
        """
        self.set_main_incoming()
        self.set_outpath()
        self.set_incoming()
        self.set_outpath()
        self.get_fastqs()
        self._verify()
        if ppg.inside_ppg():
            print("inside ppg")
        else:
            print("outside ppg")
        self.set_genome()
        self.comparison_incoming = self.read_comparison_tables()
        self.verify_frames()
        # self.comparisons_to_do = self.parse_comparisons()

    def verify_frames(self):
        self.verify_samples()
        self.verify_groups()
        if "gsea" in self.pathways:
            if "file" in "file" in self.pathways["gsea"]:
                self.verify_gsea()

    def set_genome(self):
        print("setting genome")
        import pypipegraph2 as ppg
        import mbf

        # pypegraph = ppg.new()
        if self.alignment["species"] == "Mus_musculus":
            self._genome = genomes.Mus_musculus(self.alignment["revision"])
        elif self.alignment["species"] == "Homo_sapiens":
            self._genome = genomes.Homo_sapiens(self.alignment["revision"])
        else:
            try:
                self._genome = genomes.EnsemblGenome(
                    self.alignment["species"], self.alignment["revision"]
                )
            except:
                raise
        # ppg.run()
        # self.jobs = pypegraph.jobs

    def filepath_from_incoming(self, filename: str) -> Path:
        """
        Returns a path for a file in incoming.

        Parameters
        ----------
        filename : str
            The filename.

        Returns
        -------
        Path
            Path to filename.
        """
        return self.incoming / filename

    def get_comparison_group_table(self, comparison_name: str) -> DataFrame:
        """
        Returns the DataFrame, in which the comparisons are specified.

        Parameters
        ----------
        comparison_name : str
            Name of the comparison group specified in run.toml.

        Returns
        -------
        DataFrame
            DataFrame, in which the comparisons are specified.
        """
        return self.comparison_incoming[comparison_name]

    def read_comparison_tables(self) -> Dict[str, DataFrame]:
        """
        Reads all comparison tables from incoming and returns a dictionary of them.

        Returns
        -------
        Dict[str, DataFrame]
            Dictionary with comparison tables.

        Raises
        ------
        FileNotFoundError
            If required fields in the toml file sections are missing.
        """
        comparison_tables = {}
        for group in self.comparison:
            group_file = self.comparison[group].get("file", f"{group}.tsv")
            group_path = self.filepath_from_incoming(group_file)
            if not group_path.exists():
                raise FileNotFoundError(
                    f"Comparison group {group} specified, but no file {str(group_path)} found."
                )
            df_in = pd.read_csv(group_path, sep="\t")
            comparison_tables[group] = df_in
        return comparison_tables

    def parse_single_comparisons(
        self, comparison_group: str, method_name: str, comparison_type: str, path: str
    ):
        comparisons_to_do = {}
        seen = set()
        df_in = self.get_comparison_group_table(comparison_group)
        method, options = self.comparison_method(comparison_group, method_name)
        for _, row in df_in.iterrows():
            comparison_name = f"{row['comparison_name']}({method_name})"
            if comparison_name in seen:
                raise ValueError(f"Duplicate comparison name {comparison_name} in {path}.")
            seen.add(comparison_name)
            comparisons_to_do[comparison_name] = {
                "type": comparison_type,
                "cond1": row["a"],
                "cond2": row["b"],
                "method": method,
                "method_name": method_name,
                "options": options,
            }

        return comparisons_to_do

    def check_number_of_samples(self):
        """check the number of samples"""
        raise NotImplementedError

    def parse_multi_comparisons(
        self, comparison_group: str, method_name: str, comparison_type: str, path: Path
    ):
        df_factor_path = self.filepath_from_incoming(self.comparison[comparison_group]["file"])
        multi_comparisons_to_do = {}
        df_factor = self.get_comparison_group_table(comparison_group)
        method_name = self.comparison_method_name(self.comparison[comparison_group])
        method, options = self.comparison_method(comparison_group, method_name)

        for multi_group, df_factor_group in df_factor.groupby("comparison_group"):
            # so like group ... why did i do this?
            # multi_comp_name = comparison_name
            for multi_comp_name, df_comp in df_factor_group.groupby("comparison"):
                main_factor = df_comp["main"].values[0]
                comparison_name = f"{multi_comp_name}({method_name})"
                try:
                    assert len(df_comp["main_factor"].unique()) == 1
                except AssertionError:
                    print(
                        f"Multiple values for main factor given in {df_factor_path}: {df_comp['main_factor'].unique()}"
                    )

                interactions = None
                if "interaction" in df_comp.columns:
                    interactions = df_comp["interaction"].values[0]
                factors = [x for x in df_comp.columns.values if ":" in x]
                rename = {}
                factor_reference = {}
                other_factor = None
                for factor_ref in factors:
                    factor, ref = factor_ref.split(":")
                    factor_reference[factor] = ref
                    rename[factor_ref] = factor
                    if factor != main_factor:
                        other_factor = factor
                if other_factor is None:
                    raise ValueError("Only one factor in {df_factor_path}.")
                df_factors = df_comp.rename(columns=rename)
                #####
                # for main_level in df_comp[main_factor].unique():
                #                 if main_level == factor_reference[main_factor]:
                #                     continue
                #                 print("main", main_level)
                #                 for genotype in df_comp[other_factor].unique():
                #                     if genotype == factor_reference[other_factor]:
                #                         continue
                #                     print("other", genotype)
                #                     column_prefix_effect = f"{main_level}:{factor_reference[main_factor]}({main_factor}) effect for {genotype}:{factor_reference[other_factor]}({other_factor})"
                #                     column_prefixes = [column_prefix_effect]
                #                     if len(interaction) > 0:
                #                         column_prefix_diff = f"{main_level}:{factor_reference[main_factor]}({main_factor}) effect difference for {genotype}:{factor_reference[other_factor]}({other_factor})"
                #                         column_prefixes.append(column_prefix_diff)
                #####

                multi_comparisons_to_do[comparison_name] = {
                    "type": comparison_type,
                    "main_factor": main_factor,
                    "factor_reference": factor_reference,
                    "df_factor": df_factors,
                    "interaction": interactions,
                    "method": method,
                    "options": options,
                    "method_name": method_name,
                    "multi_comp_name": multi_comp_name,
                    "file": path,
                }
        return multi_comparisons_to_do

    def parse_comparisons(self):
        comparisons_to_do = {}
        for group in self.comparison:
            comparisons_to_do[group] = {}
            method = self.comparison_method_name(self.comparison[group])
            comp_type = self.comparison[group].get("type", "ab")
            group_file = self.comparison[group].get("file", f"{group}.tsv")
            group_path = self.filepath_from_incoming(group_file)
            if comp_type == "ab":
                comparisons_to_do[group].update(
                    self.parse_single_comparisons(group, method, comp_type, group_path)
                )
            elif comp_type == "multi":
                print(group, method, comp_type, group_path)
                raise ValueError()
                comparisons_to_do[group].update(
                    self.parse_multi_comparisons(group, method, comp_type, group_path)
                )
            else:
                raise ValueError(f"Don't know what to do with type {comp_type}.")
        return comparisons_to_do

    def _verify(self):
        """
        Verifies the information given in the ru.toml file.

        This checks the presence of certain required fields and performs some
        checks on the values provided within the sections. This is called
        directly after instantiation if Analysis.

        Raises
        ------
        ValueError
            If required fields in the toml file sections are missing.
        ValueError
            If the toml file does not contain a required section.
        NotImplementedError
            If an analysis type is specified, that is currently not supported.
        FileNotFoundError
            If a run folder is not present in incoming.
        FileNotFoundError
            If no samples table exists in incoming.
        ValueError
            If the specified kit is not supported.
        ValueError
            If the species specified is not supported.
        """
        # assert all required fields are present
        for key in _required_fields:
            if hasattr(self, key):
                attr_dict = getattr(self, key)
                for field in _required_fields[key]:
                    if field not in attr_dict:
                        raise ValueError(f"Required field '{field}' in section {key} is missing.")
            else:
                raise ValueError(f"Required section '{key}' missing from {str(self.run_toml)}.")

        # assert the type
        if self.analysis_type not in self.__allowed_types:
            raise NotImplementedError(f"{self.analysis_type} is not yet implemented.")
        # check run ids
        for run_id in self.run_ids:
            if not (self.filepath_from_incoming(run_id)).exists():
                print(self.incoming.absolute())
                raise FileNotFoundError(f"Folder {run_id} not present in '{str(self.incoming)}'.")
                # TODO automatically pulling the data from rose/incoming ...
        # check samples
        if not Path(self.path_to_samples_df).exists():
            raise FileNotFoundError(f"{self.path_to_samples_df} does not exist. Please create it.")
        kit = self.samples["kit"]
        if kit not in self.__known_kits:
            raise ValueError(
                f"Provided kit {kit} is not known, currently supported are {str(self.__known_kits)}."
            )
        # check alignment
        # species = self.alignment["species"]
        # if species not in self.__accepted_species:
        #     # for some donwstream analysis we can only handle mouse and human automatically
        #     raise ValueError(f"Provided species {species} not in {str(self.__accepted_species)}.")

    @property
    def genome(self) -> EnsemblGenome:
        """
        Returns an instance of the specified ensembl genome.

        Species and revision are obtained from the run.toml and the appropriate
        genome is instantiated here.

        Returns
        -------
        EnsemblGenome
            The genome to use.
        """
        return self._genome

    def aligner(self) -> Aligner:
        """
        Returns an instance of the specified aligner and parameters for the run.

        This looks up the aligner classes in externals.aligners and returns
        an instance of the specified aligner, if such a class exists.

        Returns
        -------
        ExternalAlgorithm
            The aligner to be used.
        dict
            the aligner parameters.

        Raises
        ------
        ValueError
            If the aligner name does not match to a Class in the module.
        """
        module = sys.modules["mbf.externals.aligners"]
        aligner_name = self.alignment["aligner"]
        if not hasattr(module, aligner_name):
            raise ValueError(f"No aligner named {aligner_name} found in mbf.externals.aligners.py.")
        aligner_ = getattr(module, aligner_name)
        aligner = aligner_()
        params = {}
        if "parameters" in self.alignment:
            params = self.alignment["parameters"]
        return aligner, dict(params)

    def sample_df(self) -> DataFrame:
        """
        Reads and returns the DataFrame containing the samples to be analyzed.

        Returns
        -------
        DataFrame
            DataFrame with samples to be analyzed.
        """
        df_samples = pd.read_csv(self.path_to_samples_df, sep="\t", comment="#")
        return df_samples

    def samples_file_invariant(self):
        return ppg.FileInvariant(self.path_to_samples_df)

    def combination_df(self):
        if self.path_to_combination_df is not None:
            return pd.read_csv(self.path_to_combination_df, sep="\t")
        return None

    def combination_file_invariant(self):
        if self.path_to_combination_df is not None:
            return ppg.FileInvariant(self.path_to_samples_df)
        return []

    def assert_required_columns(self, df_samples):
        columns = ["number", "sample", "prefix", "comment", "vid"]
        not_present = set(columns).difference(set(df_samples.columns.values))
        if len(not_present) > 0:
            raise ValueError(
                f"The samples table {self.path_to_samples_df} does not contain the following required columns {not_present}.\nColumns were: {df_samples.columns.values}."
            )

    def assert_vid_unique(self, df_samples):
        vids = df_samples["vid"].dropna()
        duplicate_vids = vids[vids.duplicated()].values
        if len(duplicate_vids) > 0:
            raise ValueError(f"The following vids where assigned twice: {list(duplicate_vids)}.")

    def assert_group_in_samples(self, group, df_samples):
        if "factors" in self.comparison[group]:
            missing = set(self.comparison[group]["factors"]).difference(set(df_samples.columns))
            if len(missing) > 0:
                raise ValueError(
                    f"The sample table {str(self.path_to_samples_df)} is missing the following factors: {missing}"
                )
        elif group not in df_samples.columns:
            raise ValueError(
                f"No factors and no grouping column {group} found in {self.path_to_samples_df}. This is needed to define groups for comparisons."
            )

    def verify_samples(self):
        """
        Checks the samples and groups tables that are supposed to be in incoming.

        Checks for the correct columns to be present and the file existence.

        Raises
        ------
        ValueError
            If the samples table does not contain all required columns.
        ValueError
            If vids are assigned twice.
        ValueError
            If no groups are specified for the samples.
        FileNotFoundError
            If the group table file specified by the group column in samples table
            does not exist.
        ValueError
            If the group table is missing required columns.
        """
        df_samples = self.sample_df()
        self.assert_required_columns(df_samples)
        self.assert_vid_unique(df_samples)
        for group in self.comparison:
            self.assert_group_in_samples(group, df_samples)

    def verify_groups(self):
        ab_columns_needed = ["a", "b", "groupby", "comparison_name", "comment"]
        multi_columns_needed = ["factors", "comparison_name", "comment"]
        for group in self.comparison:
            not_present = []
            df_groups = self.get_comparison_group_table(group)
            if self.comparison[group]["type"] == "ab":
                not_present = set(ab_columns_needed).difference(set(df_groups.columns.values))
            elif self.comparison[group]["type"] == "multi":
                not_present = set(multi_columns_needed).difference(set(df_groups.columns.values))
            if len(not_present) > 0:
                raise ValueError(
                    f"The groups table for {group} does not contain the following required columns {not_present}."
                )

    def df_gsea(self):
        if self.pathways["gsea"]["file"] is None:
            return None
        return pd.read_csv(self.incoming / self.pathways["gsea"]["file"], sep="\t")

    def df_gsea_fileinvariant(self):
        if self.pathways["gsea"]["file"] is None:
            return []
        return ppg.FileInvariant(self.incoming / self.pathways["gsea"]["file"])

    def gsea_required_columns(self):
        return ["comparison_name", "phenotypes", "a", "b", "groupby"]

    def verify_gsea(self):
        df_gsea = pd.read_csv(self.incoming / self.pathways["gsea"]["file"], sep="\t")
        required = pd.Index(self.gsea_required_columns())
        missing = required.difference(df_gsea.columns)
        if len(missing) != 0:
            raise ValueError(
                f"The following required columns were missing in the gsea config file: {missing}.\nColumns were: {df_gsea.columns}."
            )

    def fastq_processor(self) -> Any:
        """
        Returns an appropriate fastq processor.

        This is based on the kit provided.

        Returns
        -------
        Any
            align.fastq2. class instance.
        """
        kit = self.samples["kit"]
        if kit == "QuantSeq":
            fastq_processor = align.fastq2.UMIExtractAndTrim(
                umi_length=6, cut_5_prime=4, cut_3_prime=0
            )
            return fastq_processor
        elif kit == "NextSeq":
            return align.fastq2.Straight()
        else:
            raise NotImplementedError  # TODO: read processor from toml for more fine-grained control

    def post_processor(self) -> Any:
        """
        Returns an appropriate post_processor for an aligned lane.

        This is based on the kit provided in the run.toml.

        Returns
        -------
        Any
            align.fastq2. class instance.

        Raises
        ------
        NotImplementedError
            For different kits.
        """
        kit = self.samples["kit"]
        if kit == "QuantSeq":
            post_processor = align.post_process.UmiTools_Dedup()
            return post_processor
        elif kit == "NextSeq":
            return None
        else:
            raise NotImplementedError  # TODO: read processor from toml for more fine-grained control

    # def raw_counter(self) -> _FastTagCounter:
    #     kit = self.samples["kit"]
    #     stranded = self.samples["stranded"]
    #     if kit == "QuantSeq":
    #         if stranded:
    #             return genomics.genes.anno_tag_counts.ExonSmartStrandedRust
    #         else:
    #             raise NotImplementedError
    #     elif kit == "NextSeq":
    #         if stranded:
    #             return genomics.genes.anno_tag_counts.ExonSmartStrandedRust
    #         else:
    #             raise NotImplementedError
    #     else:
    #         raise NotImplementedError  # TODO: to toml for more fine-grained control

    # def norm_counter(self) -> _NormalizationAnno:
    #     kit = self.samples["kit"]
    #     stranded = self.samples["stranded"]
    #     if stranded:
    #         if kit == "QuantSeq":
    #             return genomics.genes.anno_tag_counts.NormalizationCPM
    #         elif kit == "NextSeq":
    #             return genomics.genes.anno_tag_counts.NormalizationTPM
    #         else:
    #             raise NotImplementedError  # TODO: to toml for more fine-grained control
    #     else:
    #         raise NotImplementedError

    def report_name(self) -> str:
        if "name" in self.reports:
            return self.reports["name"]
        else:
            return "run_report"

    def report(self) -> NB:
        """
        Generates a NB to collect plots with a name given in the run.toml.

        Returns
        -------
        NB
            The NB instance to use.
        """
        dependencies = [
            ppg.FunctionInvariant("FI_ana", self.summary_markdown),
            ppg.FunctionInvariant("FI_ana", self.summary_markdown),
        ]
        if "name" in self.reports:
            nb = NB(self.reports["name"], dependencies=dependencies)
        else:
            nb = NB("run_report", dependencies=dependencies)
        return nb

    def has_gene_filter_specified(self) -> bool:
        """
        Wether the genes should be filtered prior to DEG analysis.

        If a genes.filter section is given in the run.toml, this is true.
        Filter conditions specified in this section are used to create a filter
        function to use.

        Returns
        -------
        bool
            True, if gene.filter is supplied in run.toml.
        """
        return "filter" in self.genes

    def genes_filter(self) -> Callable:
        """
        Returns a filter function getter that takes a list of expression columns to filter by.

        The filter function is generated based on the run.toml genes.filter
        subsection.

        Returns
        -------
        Callable
            Callable that returns a DataFrame filter function and takes a list
            of columns to filter by.

        Raises
        ------
        ValueError
            If no filter criteria for genes have been specified but this is called.
        """
        if not self.has_gene_filter_specified():
            raise ValueError("No filters have been specified in run.toml.")
        filter_spec = self.genes["filter"]
        threshold = filter_spec.get("cpm_threshold", None)
        canonical = filter_spec.get("canonical", True)
        canonical_chromosomes = None
        if canonical:
            canonical_chromosomes = self._genome.get_true_chromosomes()
        biotypes = filter_spec.get("biotypes", None)
        at_least = filter_spec.get("at_least", 1)
        return filter_function(threshold, at_least, canonical_chromosomes, biotypes)

    def comparison_method(self, group_name: str, method: str) -> Any:
        """
        Returns an instance of a comparison method.

        This is intended as a parameter for comparisons.Comparison.__init__.
        This looks up the method classes in comparions.methods and returns
        an instance of the specified method, if such a class exists.
        Optional paramaters can be specified in run.toml with 'parameter' key.

        Parameters
        ----------
        method : str
            Name of the method to be used.

        Returns
        -------
        Any
            A class from comparisons.methods.

        Raises
        ------
        ValueError
            If the method is not found in the module.
        """
        module = sys.modules["mbf.comparisons.methods"]
        if hasattr(module, method):
            method_ = getattr(module, method)
        else:
            module = sys.modules["mdataframe.differential"]
            if hasattr(module, method):
                method_ = getattr(module, method)
            else:
                raise ValueError(
                    f"No method named {method} found in mbf.comparisons.methods.py or mdataframe.differential.py."
                )
        options = {
            "laplace_offset": self.comparison[group_name].get("laplace_offset", 0),
            "include_other_samples_for_variance": self.comparison[group_name].get(
                "include_other_samples_for_variance", True
            ),
        }
        if "parameters" in self.comparison[group_name]:
            parameters = self.comparison[group_name]["parameters"]
            print("called with parameters")
            return method_(**parameters), options
        else:
            print("called without parameters")
            return method_(), options

    def deg_filter_expressions(self, comparison_group: str) -> List[Any]:
        """
        Returns the filter expression used to filter the DE genes after runnning
        the comparison.

        This defaults to selecting the logFC >= 1 and FDR <= 0.05.

        Parameters
        ----------
        comparison_group : str
            The comparison for which the expression is used.

        Returns
        -------
        List[Any]
            List of filter expressions.
        """
        default = [[["FDR", "<=", 0.05], ["logFC", "|>", 1]]]
        if "filter_expressions" in self.comparison[comparison_group]:
            expr = toml_arrays_to_lists(self.comparison[comparison_group]["filter_expressions"])
            return expr
        else:
            return default

    @classmethod
    def deg_filter_expression_as_str(self, filter_expr: List[List[Any]]) -> str:
        """
        Turns the filter expression into a str used as suffix for filtered genes
        names.

        Parameters
        ----------
        filter_expr : List[List[Any]]
            The expression to be stringified.

        Returns
        -------
        str
            string representation of filter_expr.
        """
        x = "_".join(["".join([str(x) for x in exp]) for exp in filter_expr])
        return x

    def pretty(self) -> str:
        """
        Returns a pretty string for the class instance.

        Returns
        -------
        str
            String representation with indent.
        """
        pp = PrettyPrinter(indent=4)
        d = vars(self).copy()
        # del d["jobs"]
        return "Analysis(\n" + pp.pformat(d) + "\n)"

    def display_summary(self):
        """
        Displays the summary of analysis on console.
        """
        md = self.summary_markdown()
        console.print(Markdown(md))

    def summary_markdown(self) -> str:
        """
        Generates a run report summary and returns it as markdown string.

        This is intended for double checking the analysis settings and
        should countain all the information inferred from the run.toml.

        Returns
        -------
        str
            Summary of run settings.
        """
        pp = PrettyPrinter(indent=4)
        report_header = f"## Analysis specification from toml file '{self.run_toml}'\n"
        genome_name = f"{self.alignment['species']}, {self.alignment['revision']}"
        report_header += f"Genome used: {genome_name}  \n"
        aligner, aligner_params = self.aligner()
        report_header += f"Aligner used: {aligner.name} with parameter {aligner_params}  \n"
        report_header += f"Run-IDs: {pp.pformat(self.run_ids)}  \n"
        report_header += f"Fastq-Processor: {self.fastq_processor().__class__.__name__}  \n"
        postprocessor_name = "None"
        if self.post_processor is not None:
            postprocessor_name = self.post_processor().__class__.__name__
        report_header += f"Alignment Postprocessor: {postprocessor_name} \n"
        report_header += "\nNorm counters requested:  \n"
        for key in self.normalization:
            report_header += f"{key}={self.normalization[key]}  \n"
        report_header += "\n### Samples  \n  \n"
        df_samples = self.sample_df()
        report_header += df_to_markdown_table(df_samples)
        report_header += "\n### Comparisons requested  \n  \n"
        for group_name in self.comparison:
            report_header += f"Comparison group: '{group_name}' \n"
            report_header += f"DGE method = {self.comparison[group_name]['method']}  \n"
            report_header += f"Comparison type = {self.comparison[group_name]['type']}  \n  \n"
            df_in = self.get_comparison_group_table(group_name)  # pd.read_csv(filepath, sep="\t")
            report_header += df_to_markdown_table(df_in)
        report_header += "\n### Genes  \n"
        genes_used_name = self.genes_used_name()
        report_header += f"Genes used: {genes_used_name}  \n"
        if self.has_gene_filter_specified():
            report_header += f"Genes used for DE analysis are \n{self.genes_used_description()}"
        report_header += "\n### Comparisons  \n"
        report_header += "\n### Pathway Analysis  \n"
        for pathway_method in self.pathways:
            parameter = {}
            if "parameters" in self.pathways[pathway_method]:
                parameter = self.pathways[pathway_method]["parameters"]
            collections = self.pathway_collections(self.pathways[pathway_method])
            if pathway_method == "ora":
                report_header += "\nOver-Representation Analysis (ORA)  \n"
            if pathway_method == "gsea":
                report_header += "\nGene Set Enrichment Analysis (GSEA)  \n"
            report_header += f"Collections used: {collections}  \n"
            report_header += f"Parameters: {parameter}  \n"
        combination_df = self.combination_df()
        if combination_df is not None:
            report_header += "\n### Set operations on comparisons  \n  \n"
            report_header += df_to_markdown_table(combination_df)
        return report_header

    def specification(self) -> str:
        """
        Returns the specification of the toml file as string.

        Returns
        -------
        str
            Content of the toml file.
        """
        report_spec = "\n### Specification  \n" + self.pretty()
        return report_spec

    def combinations(self) -> Iterator:
        """
        Returns an Iterator over all set operations to be performed. This
        just reads from the combinations file.

        Returns
        -------
        Iterable
            Iterator over all combinations.

        Yields
        ------
        Iterator
            a tuple of comparison name, new name prefix of the combined genes
            object, a list of comparisons to include, a generator function
            that performs the set operation on Genes objects and the operation
            to be performed.
        """
        df_combinations = self.combination_df()
        if df_combinations is not None:
            for _, row in df_combinations.iterrows():
                condition_group = row["condition_group"]
                new_name_prefix = row["combined_name"]
                comparisons_to_add = row["comparison_names"].split(",")
                generator = self.get_generator(row["operation"])
                assert_uniqueness(comparisons_to_add)
                yield condition_group, new_name_prefix, comparisons_to_add, generator, row[
                    "operation"
                ]

    def get_generator(self, operation):
        def generator_diff(new_name, genes_to_combine, path):
            return genomics.genes.genes_from.FromDifference(
                new_name, genes_to_combine[0], genes_to_combine[1], sheet_name="Differences"
            )

        def generator_intersect(new_name, genes_to_combine, path):
            return genomics.genes.genes_from.FromIntersection(
                new_name, genes_to_combine, sheet_name="Intersections"
            )

        def generator_union(new_name, genes_to_combine, path):
            return genomics.genes.genes_from.FromAny(
                new_name, genes_to_combine, sheet_name="Unions"
            )

        if operation == "difference":
            generator = generator_diff
        elif operation == "intersection":
            generator = generator_intersect
        elif operation == "union":
            generator = generator_union
        else:
            raise NotImplementedError(f"Unknown set operation specified: {operation}")
        return generator

    def get_fastqs(self):
        fill_incoming(self.run_ids, self.main_incoming, self.incoming)

    def get_collections_for_runner(self, pathway_method):
        collections = ["h"]
        if "collections" in self.pathways[pathway_method]:
            collections = list(self.pathways[pathway_method]["collections"])
        if len(collections) == 1:
            collections_to_run = collections
        else:
            collections_to_run = [collections] + [collection for collection in collections]
        return collections_to_run

    def get_gsea_parameter(self):
        return dict(self.pathways["gsea"].get("parameter", {}))

    def genes_used_description(self):
        """"""
        pp = PrettyPrinter(indent=4)
        if self.has_gene_filter_specified():
            desc = "\nfiltered by:  \n"
            if "chr" in self.genes["filter"]:
                desc += f"chr = {self.genes['filter']}\n"
            if "biotypes" in self.genes["filter"]:
                desc += f"biotype in {pp.pformat(self.genes['filter']['biotypes'])}\n"
            if "thresholds" in self.genes["filter"]:
                for filter_spec in self.genes["filter"]["thresholds"]:
                    filter_str = " ".join([str(x) for x in filter_spec])
                    desc += filter_str + "\n"
        else:
            desc = "unfiltered"
        return desc

    def genes_used_name(self):
        """Get the name for the genes used"""
        genes_used_name = f"Genes_{self.genome.name}"
        if self.has_gene_filter_specified():
            if "chr" in self.genes["filter"]:
                genes_used_name += f"_chr={self.genes['filter']['chr']}"
            if "biotypes" in self.genes["filter"]:
                genes_used_name += "_biotypes"
            if "thresholds" in self.genes["filter"]:
                for filter_spec in self.genes["filter"]["thresholds"]:
                    filter_str = "".join([str(x) for x in filter_spec])
                    genes_used_name += f"_{filter_str}"
            if (
                "drop_empty_names" in self.genes["filter"]
                and self.genes["filter"]["drop_empty_names"]
            ):
                genes_used_name += "_drop_empty_names"
        return genes_used_name


def analysis(req_file: Path = Path("run.toml")) -> Analysis:
    """
    Returns a new ANalysis instance from a given toml file.

    Parameters
    ----------
    req_file : Path, optional
        Path to toml file, by default Path("run.toml").

    Returns
    -------
    Analysis
        A new Analysis instance.
    """
    return Analysis(req_file, **read_toml(req_file))


def toml_arrays_to_lists(toml_array) -> List:
    return [[list(expression) for expression in filter_set] for filter_set in toml_array]
