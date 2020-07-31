#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""base.py: Contains toml parser and basic data class."""

from mruns import __version__
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, Callable, List, Dict, Tuple, Any, ClassVar, Iterator
from mreports import NB
from mbf_genomes import EnsemblGenome
from mbf_externals import ExternalAlgorithm
from pandas import DataFrame
from .util import filter_function, read_toml, df_to_markdown_table
from pprint import PrettyPrinter
from mbf_genomics.genes.anno_tag_counts import _NormalizationAnno, _FastTagCounter
import dataclasses
import pandas as pd
import pypipegraph as ppg
import tomlkit
import logging
import sys
import sys
import inspect
import json
import mbf_genomes
import mbf_externals
import mbf_align
import mbf_genomics

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


_logger = logging.getLogger(__name__)


_required_fields = {
    "project": ["name", "analysis_type", "run_ids"],
    "samples": ["df_samples", "reverse_reads", "kit", "stranded"],
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
    downstream: Dict[str, Any]
    reports: Dict[str, Any]
    combination: Dict[str, Any]
    __allowed_types: ClassVar[List[str]] = ["RNAseq"]
    __known_kits: ClassVar[List[str]] = ["QuantSeq", "NextSeq"]
    __accepted_species: ClassVar[List[str]] = ["Homo_sapiens", "Mus_musculus"]

    @property
    def incoming(self):
        """
        The incoming folder with all external data.
        """
        if "incoming" in self.project:
            return Path(self.project["incoming"])
        return Path("incoming")

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
        return self.incoming / self.samples["df_samples"]

    @property
    def path_to_combination_df(self):
        if "set_operations" in self.combination:
            return self.incoming / self.combination["set_operations"]
        else:
            return None

    @classmethod
    def get_comparison_methods(cls):
        module = sys.modules["mbf_comparisons.methods"]
        ret = {}
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                ret[name] = obj
        return ret

    def __post_init__(self):
        """
        Cleans up after initialization.
        """
        self._verify()
        if ppg.inside_ppg():
            self.set_genome()
        self.verify_samples()

    def set_genome(self):
        self._genome = mbf_genomes.EnsemblGenome(
            self.alignment["species"], self.alignment["revision"]
        )

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
                        raise ValueError(
                            f"Required field '{field}' in section {key} is missing."
                        )
            else:
                raise ValueError(
                    f"Required section '{key}' missing from {str(self.run_toml)}."
                )

        # assert the type
        if self.analysis_type not in self.__allowed_types:
            raise NotImplementedError(f"{self.analysis_type} is not yet implemented.")
        # check run ids
        for run_id in self.run_ids:
            if not (self.incoming / run_id).exists():
                print(self.incoming.absolute())
                raise FileNotFoundError(
                    f"Folder {run_id} not present in '{str(self.incoming)}'."
                )
                # TODO automatically pulliong the data from rose/incoming ...
        # check samples
        if not Path(self.path_to_samples_df).exists():
            raise FileNotFoundError(
                f"No samples.tsv in {self.incoming}. Please create it."
            )
        kit = self.samples["kit"]
        if kit not in self.__known_kits:
            raise ValueError(
                f"Provided kit {kit} is not known, currently supported are {str(self.__known_kits)}."
            )
        # check alignment
        species = self.alignment["species"]
        if species not in self.__accepted_species:
            # for some donwstream analysis we can only handle mouse and human automatically
            raise ValueError(
                f"Provided species {species} not in {str(self.__accepted_species)}."
            )

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

    def aligner(self) -> ExternalAlgorithm:
        """
        Returns an instance of the specified aligner and parameters for the run.

        This looks up the aligner classes in mbf_externals.aligners and returns
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
        module = sys.modules["mbf_externals.aligners"]
        aligner_name = self.alignment["aligner"]
        if not hasattr(module, aligner_name):
            raise ValueError(
                f"No aligner named {aligner_name} found in mbf_externals.aligners.py."
            )
        aligner_ = getattr(module, aligner_name)
        aligner = aligner_()
        params = {}
        if "parameters" in self.alignment:
            params = self.alignment["parameter"]
        return aligner, params

    def sample_df(self) -> DataFrame:
        """
        Reads and returns the DataFrame containing the samples to be analyzed.

        Returns
        -------
        DataFrame
            DataFrame with samples to be analyzed.
        """
        df_samples = pd.read_csv(self.path_to_samples_df, sep="\t")
        return df_samples

    def combination_df(self):
        if self.path_to_combination_df is not None:
            return pd.read_csv(self.path_to_combination_df, sep="\t")
        return None

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
        columns = ["number", "sample", "prefix", "comment", "vid"]
        not_present = set(columns).difference(set(df_samples.columns.values))
        if len(not_present) > 0:
            raise ValueError(
                f"The samples table {self.path_to_samples_df} does not contain the following required columns {not_present}."
            )
        group_columns = []
        vids = df_samples["vid"].dropna()
        duplicate_vids = vids[vids.duplicated()].values
        if len(duplicate_vids) > 0:
            raise ValueError(
                f"The following vids where assigned twice: {list(duplicate_vids)}."
            )
        for c in df_samples.columns:
            if c.startswith("group"):
                group_columns.append(c)
        if len(group_columns) == 0:
            raise ValueError(
                f"No column starting with 'group' in {self.path_to_samples_df}. This is needed to define groups for comparisons."
            )
        for col in group_columns:
            fpath = self.incoming / f"{col}.tsv"
            if not fpath.exists():
                raise FileNotFoundError(
                    "Group column {col} specified, but no file {str(fpath)} found."
                )
            df_groups = pd.read_csv(fpath, sep="\t")
            columns = ["a", "b", "comparison_name", "comment"]
            not_present = set(columns).difference(set(df_groups.columns.values))
            if len(not_present) > 0:
                raise ValueError(
                    f"The groups table {str(fpath)} does not contain the following required columns {not_present}."
                )

    def fastq_processor(self) -> Any:
        """
        Returns an appropriate fastq processor.

        This is based on the kit provided.

        Returns
        -------
        Any
            mbf_align.fastq2. class instance.
        """
        kit = self.samples["kit"]
        if kit == "QuantSeq":
            fastq_processor = mbf_align.fastq2.UMIExtractAndTrim(
                umi_length=6, cut_5_prime=4, cut_3_prime=0
            )
            return fastq_processor
        elif kit == "NextSeq":
            return mbf_align.fastq2.Straight()
        else:
            raise NotImplementedError  # TODO: read processor from toml for more fine-grained control

    def raw_counter(self) -> _FastTagCounter:
        kit = self.samples["kit"]
        stranded = self.samples["stranded"]
        if kit == "QuantSeq":
            if stranded:
                return mbf_genomics.genes.anno_tag_counts.ExonSmartStrandedRust
            else:
                raise NotImplementedError
        elif kit == "NextSeq":
            if stranded:
                return mbf_genomics.genes.anno_tag_counts.ExonSmartStrandedRust
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError  # TODO: to toml for more fine-grained control

    def norm_counter(self) -> _NormalizationAnno:
        kit = self.samples["kit"]
        stranded = self.samples["stranded"]
        if stranded:
            if kit == "QuantSeq":
                return mbf_genomics.genes.anno_tag_counts.NormalizationCPM
            elif kit == "NextSeq":
                return mbf_genomics.genes.anno_tag_counts.NormalizationTPM
            else:
                raise NotImplementedError  # TODO: to toml for more fine-grained control
        else:
            raise NotImplementedError

    def report(self) -> NB:
        """
        Generates a NB to collect plots with a name given in the run.toml.

        Returns
        -------
        NB
            The NB instance to use.
        """

        if "name" in self.reports:
            return NB(self.reports["name"])
        else:
            return NB("run_report")

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

    def comparison_method(self, method: str) -> Any:
        """
        Returns an instance of a comparison method.

        This is intended as a parameter for mbf_comparisons.Comparison.__init__.
        This looks up the method classes in mbf_comparions.methods and returns
        an instance of the specified method, if such a class exists.
        Optional paramaters can be specified in run.toml with 'parameter' key.

        Parameters
        ----------
        method : str
            Name of the method to be used.

        Returns
        -------
        Any
            A class from mbf_comparisons.methods.

        Raises
        ------
        ValueError
            If the method is not found in the module.
        """
        assert method in self.comparison
        module = sys.modules["mbf_comparisons.methods"]
        if not hasattr(module, method):
            raise ValueError(
                f"No method named {method} found in mbf_comparisons.methods.py."
            )
        method_ = getattr(module, method)
        options = {
            "laplace_offset": self.comparison[method].get("laplace_offset", 0),
            "include_other_samples_for_variance": self.comparison[method].get(
                "include_other_samples_for_variance", True
            ),
        }
        if "parameters" in self.comparison[method]:
            parameters = self.comparison[method]["parameters"]
            return method_(**parameters), options
        else:
            return method_(), options

    def deg_filter_expressions(self, method: str) -> List[Any]:
        """
        Returns the filter expression used to filter the DE genes after runnning
        the comparison.

        This defaults to selecting the logFC >= 1 and FDR <= 0.05.

        Parameters
        ----------
        method : str
            The method for which the expression is used.

        Returns
        -------
        List[Any]
            List of filter expressions.
        """
        default = [[["FDR", "<=", 0.05], ["log2FC", "|>", 1]]]
        if "filter_expressions" in self.comparison[method]:
            expr = self.comparison[method]["filter_expressions"]
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
        d["genome"] = d["_genome"]
        del d["_genome"]
        return "Analysis(\n" + pp.pformat(d) + "\n)"

    def summary(self) -> str:
        """
        Generates a run report summary and returns it as string.

        This is intended for double checking the analysis settings and
        should countain all the information inferred from the run.toml.

        Returns
        -------
        str
            Summary of run settings.
        """
        pp = PrettyPrinter(indent=4)
        report_header = f"Analysis from toml file '{self.run_toml}'\n\n"
        report_header += "Specification\n-------------\n" + self.pretty() + "\n\n"
        report_header += f"Genome used: {self.genome.name}\n"
        aligner, aligner_params = self.aligner()
        report_header += (
            f"Aligner used: {aligner.name} with parameter {aligner_params}\n"
        )
        report_header += f"Run-IDs: {pp.pformat(self.run_ids)}\n"
        report_header += (
            f"Fastq-Processor: {self.fastq_processor().__class__.__name__}\n"
        )
        raw_counter = self.raw_counter()
        norm_counter = self.norm_counter()
        report_header += f"Raw counter: {raw_counter.__name__}\n"
        report_header += f"Norm counter: {norm_counter.__name__}\n"
        report_header += "\nSamples\n-------\n"
        df_samples = self.sample_df()
        report_header += pp.pformat(df_samples)
        conditions = [x for x in df_samples.columns if x.startswith("group")]
        report_header += "\n\nComparisons requested\n---------------------\n"
        comparisons_to_do: Dict[str, List] = {}
        for condition in conditions:
            comparisons_to_do[condition] = []
            df_in = pd.read_csv(f"incoming/{condition}.tsv", sep="\t")
            for _, row in df_in.iterrows():
                comparisons_to_do[condition].append(
                    (row["a"], row["b"], row["comparison_name"])
                )
            report_header += f"Comparison group: '{condition}'\n"
            report_header += pp.pformat(df_in) + "\n"
        report_header += f"\nGenes\n-----\n"
        genes_used_name = f"Genes_{self.genome.name}"
        if self.has_gene_filter_specified():
            report_header += "Genes filtered prior to DE analysis by: \n"
            if (
                "canonical" in self.genes["filter"]
                and self.genes["filter"]["canonical"]
            ):
                report_header += "- canonical chromosomes only\n"
                genes_used_name += "_canonical"
            if "biotypes" in self.genes["filter"]:
                at_least = self.genes["filter"].get("at_least", 1)
                report_header += (
                    f"- biotype in {pp.pformat(self.genes['filter']['biotypes'])}\n"
                )
                genes_used_name += "_biotypes"
            if "cpm_threshold" in self.genes["filter"]:
                threshold = self.genes["filter"]["cpm_threshold"]
                report_header += f"- at least {at_least} samples with normalized expression >= {threshold}\n"
                genes_used_name += f"_{at_least}samples>={threshold}"
        report_header += f"Genes used: {genes_used_name}\n"
        report_header += f"\nComparisons\n-----------\n"
        for condition_group in conditions:
            report_header += f"From '{condition_group}':\n"
            for method_name in self.comparison:
                _, options = self.comparison_method(method_name)
                for cond1, cond2, _ in comparisons_to_do[condition_group]:
                    if options["include_other_samples_for_variance"]:
                        x = "fit on all samples"
                    else:
                        x = "fit on conditions"
                    desc = f"compare {cond1} vs {cond2} using {method_name} (offset={options['laplace_offset']}, {x})\n"
                    report_header += desc
        report_header += f"\nDownstream Analysis\n-------------------\n"
        for downstream in self.downstream:
            if downstream == "pathway_analysis":
                for pathway_method in self.downstream[downstream]:
                    if pathway_method == "ora":
                        collections = ["h"]
                        if "collections" in self.downstream[downstream][pathway_method]:
                            collections = self.downstream[downstream][pathway_method][
                                "collections"
                            ]
                        report_header += f"Over-Representation Analysis (ORA)\n"
                        report_header += f"Collections used: {collections}\n"
                    if pathway_method == "gsea":
                        collections = ["h"]
                        if "collections" in self.downstream[downstream][pathway_method]:
                            collections = self.downstream[downstream][pathway_method][
                                "collections"
                            ]
                        parameter = {"permutations": 1000}
                        if "parameter" in self.downstream[downstream][pathway_method]:
                            parameter = self.downstream[downstream][pathway_method][
                                "parameter"
                            ]
                        report_header += "Gene Set Enrichment Analysis (GSEA)\n"
                        report_header += f"Collections: {collections}\n"
                        report_header += f"Parameters: {parameter}\n"
        combination_df = self.combination_df()
        if combination_df is not None:
            report_header += f"\nSet operations on comparisons\n-----------------------------\n"
            report_header += pp.pformat(combination_df)
        
        return report_header

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
        report_header = f"## Analysis from toml file '{self.run_toml}'\n"
        report_header += "### Specification  \n"
        for line in self.pretty().split("\n"):
            report_header += line + "  \n"
        report_header += "\n"
        report_header += f"Genome used: {self.genome.name}  \n"
        aligner, aligner_params = self.aligner()
        report_header += (
            f"Aligner used: {aligner.name} with parameter {aligner_params}  \n"
        )
        report_header += f"Run-IDs: {pp.pformat(self.run_ids)}  \n"
        report_header += (
            f"Fastq-Processor: {self.fastq_processor().__class__.__name__}  \n"
        )
        raw_counter = self.raw_counter()
        norm_counter = self.norm_counter()
        report_header += f"Raw counter: {raw_counter.__name__}  \n"
        report_header += f"Norm counter: {norm_counter.__name__}  \n"
        report_header += "\n### Samples  \n  \n"
        df_samples = self.sample_df()
        report_header += df_to_markdown_table(df_samples)
        conditions = [x for x in df_samples.columns if x.startswith("group")]
        report_header += "\n\n### Comparisons requested  \n"
        comparisons_to_do: Dict[str, List] = {}
        for condition in conditions:
            comparisons_to_do[condition] = []
            df_in = pd.read_csv(f"incoming/{condition}.tsv", sep="\t")
            for _, row in df_in.iterrows():
                comparisons_to_do[condition].append(
                    (row["a"], row["b"], row["comparison_name"])
                )
            report_header += f"Comparison group: '{condition}'  \n\n"
            report_header += df_to_markdown_table(df_in) + "\n"
        report_header += f"\n### Genes  \n"
        genes_used_name = f"Genes_{self.genome.name}"
        if self.has_gene_filter_specified():
            report_header += "Genes filtered prior to DE analysis by:  \n"
            if (
                "canonical" in self.genes["filter"]
                and self.genes["filter"]["canonical"]
            ):
                report_header += "- canonical chromosomes only\n"
                genes_used_name += "_canonical"
            if "biotypes" in self.genes["filter"]:
                at_least = self.genes["filter"].get("at_least", 1)
                report_header += (
                    f"- biotype in {pp.pformat(self.genes['filter']['biotypes'])}\n"
                )
                genes_used_name += "_biotypes"
            if "cpm_threshold" in self.genes["filter"]:
                threshold = self.genes["filter"]["cpm_threshold"]
                report_header += f"- at least {at_least} samples with normalized expression >= {threshold}\n"
                genes_used_name += f"_{at_least}samples>={threshold}"
        report_header += f"Genes used: {genes_used_name}  \n"
        report_header += f"\n### Comparisons  \n"
        for condition_group in conditions:
            report_header += f"From '{condition_group}':  \n"
            for method_name in self.comparison:
                _, options = self.comparison_method(method_name)
                for cond1, cond2, _ in comparisons_to_do[condition_group]:
                    if options["include_other_samples_for_variance"]:
                        x = "fit on all samples"
                    else:
                        x = "fit on conditions"
                    desc = f"- compare {cond1} vs {cond2} using {method_name} (offset={options['laplace_offset']}, {x})  \n"
                    report_header += desc
        report_header += f"\n### Downstream Analysis  \n"
        for downstream in self.downstream:
            if downstream == "pathway_analysis":
                for pathway_method in self.downstream[downstream]:
                    if pathway_method == "ora":
                        collections = ["h"]
                        if "collections" in self.downstream[downstream][pathway_method]:
                            collections = self.downstream[downstream][pathway_method][
                                "collections"
                            ]
                        report_header += f"\nOver-Representation Analysis (ORA)  \n"
                        report_header += f"Collections used: {collections}  \n"
                    if pathway_method == "gsea":
                        collections = ["h"]
                        if "collections" in self.downstream[downstream][pathway_method]:
                            collections = self.downstream[downstream][pathway_method][
                                "collections"
                            ]
                        parameter = {"permutations": 1000}
                        if "parameter" in self.downstream[downstream][pathway_method]:
                            parameter = self.downstream[downstream][pathway_method][
                                "parameter"
                            ]
                        report_header += "\nGene Set Enrichment Analysis (GSEA)  \n"
                        report_header += f"Collections: {collections}  \n"
                        report_header += f"Parameters: {parameter}  \n"
        combination_df = self.combination_df()
        if combination_df is not None:
            report_header += f"\n### Set operations on comparisons  \n"
            report_header += df_to_markdown_table(combination_df)
        return report_header

    def combinations(self) -> Iterator:
        df_combinations = self.combination_df()
        if df_combinations is not None:
            for _, row in df_combinations.iterrows():
                condition_group = row["condition_group"]
                new_name_prefix = row["combined_name"]
                comparisons_to_add = row["comparison_names"].split(",")
                if row["operation"] == "difference":
                    def generator(new_name, genes_to_combine):
                        return mbf_genomics.genes.genes_from.FromDifference(new_name, genes_to_combine[0], genes_to_combine[1], sheet_name="Differences")
                elif row["operation"] == "intersection":
                    def generator(new_name, genes_to_combine):
                        return mbf_genomics.genes.genes_from.FromIntersection(new_name, genes_to_combine, sheet_name="Intersections")
                elif row["operation"] == "union":
                    def generator(new_name, genes_to_combine):
                        return mbf_genomics.genes.genes_from.FromAny(new_name, genes_to_combine, sheet_name="Unions")
                yield condition_group, new_name_prefix, comparisons_to_add, generator


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
