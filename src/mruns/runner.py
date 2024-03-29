#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""runner.py: Contains a Runner class that takes care of creating the
necessary jobs for an analysis run.

what we want is a syntax like that:

runner = Runner(analysis)
runner.create_lanes()
runner.align()
runner.count()
runner.differential()
runner.annotate()
"""

import pandas as pd
import pypipegraph2 as ppg
import mbf
import sys
import logging
from collections import OrderedDict
from mbf.align import Sample
from mpathways import GSEA
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union, Set
from pandas import Series
from mbf import align
from mbf.genomics.genes.anno_tag_counts import _NormalizationAnno, _FastTagCounter
from pypipegraph2 import Job
from mbf.genomics.annotator import Annotator
from mbf.genomics.genes import Genes
from .base import Analysis
from mdataframe import Filter, _Transformer
from mreports import NB, MarkdownItem, PlotItem, GSEAHTMLItem
from .mbf_compliance import (
    # FromTransformerWrapper,
    DifferentialWrapper,
    GenesCollection,
    GenesWrapper,
)
from .defaults import ORA
from .modules import VolcanoModule, PCAModule, HeatmapModule


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class Runner:
    def __init__(
        self,
        analysis: Analysis,
        name: str = "DefaultRunner",
        logfile: str = "logs/runner.log",
        log_level: str = "DEBUG",
    ):
        self.analysis = analysis
        self.name = name
        self._counter_lookup: Dict[
            str, Dict[str, str]
        ] = {}  # column lookup for counter
        self._counters: Dict[
            str, Annotator
        ] = {}  # all count annotators by counter key (e.g. raw)
        self.__ora_results: Dict[str, Dict[str, Job]] = {}
        self.__gsea_results: Dict[str, Dict[str, Tuple[Job, Path]]] = {}
        self.results = self.analysis.outpath
        self.init_logger(Path(logfile), log_level)
        self.log_analysis_params()
        self.init_tools()

    def init_logger(self, logfile: Path, log_level: str = "DEBUG"):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel(log_level)
        # console_handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # console_handler.setLevel(log_level)
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        file_handler = logging.FileHandler(
            logfile, mode="w", encoding=None, delay=False, errors=None
        )
        logger.addHandler(file_handler)
        self.logger = logger

    def init_tools(self):
        "initializes the necessary tools to run the analysis"
        self.set_aligner()
        self._fastq_processor = self.get_fastq_processor_from_analysis()
        self.logger.info(f"Fastq processor: {self._fastq_processor.__class__.__name__}")
        self._postprocessor = self.analysis.post_processor()
        self.logger.info(f"Post processor: {self._postprocessor.__class__.__name__}")
        self._samples = self.analysis.sample_df()
        self.logger.info(f"Sample file: {self.analysis.path_to_samples_df}")
        self._genome = self._get_genome()
        self.logger.info(
            f"Genome: {self._genome.species}, rev. {self._genome.revision}"
        )
        self._genes_all = mbf.genomics.genes.Genes(self._genome)
        self._raw_counter = self.get_raw_counter_from_kit()
        self.logger.info(f"Raw Counter: {self._raw_counter}")
        self._normalizer = self.get_normalizer()
        for norm in self._normalizer:
            self.logger.info(f"Norm Counters: {norm}: {self._normalizer[norm]}")
        self._set_comparison_frames()
        self._set_comparison_name_group_lookup()
        self._set_combinations()
        self._load_gseas()
        genes_all_wrapper = GenesWrapper(
            self._genes_all,
            ["genes_all"],
            description=f"All Genes from {self.genome}",
        )
        self._genes_collection = GenesCollection({"genes_all": genes_all_wrapper})
        self.logger.info(f"Master genes table: {genes_all_wrapper.path}")
        self.gsea = GSEA()
        self._init_report()

    def log_analysis_params(self):
        self.logger.info(f"Analysis: {self.analysis.name}")
        self.logger.info(self.analysis.summary_markdown())
        self.logger.info(f"Output path: {self.results}")

    def _init_report(self):
        """
        Generates a NB to collect plots with a name given in the run.toml.

        Returns
        -------
        NB
            The NB instance to use.
        """
        dependencies = [
            ppg.FunctionInvariant("FI_ana", self.analysis.summary_markdown),
            ppg.FunctionInvariant("FI_ana", self.analysis.summary_markdown),
        ]
        nb = NB(self.analysis.report_name(), dependencies=dependencies)
        self.logger.info(f"Initialized report: {nb.name}")
        self._report = nb

    def _get_genome(self):
        # ppg.new()
        if self.analysis.alignment["species"] == "Mus_musculus":
            genome = mbf.genomes.Mus_musculus(self.analysis.alignment["revision"])
        elif self.analysis.alignment["species"] == "Homo_sapiens":
            genome = mbf.genomes.Homo_sapiens(self.analysis.alignment["revision"])
        else:
            genome = mbf.genomes.EnsemblGenome(
                self.analysis.alignment["species"], self.analysis.alignment["revision"]
            )
        # ppg.run()
        return genome

    @property
    def columns_lookup(self):
        "Lookup for all columns"
        return self._counter_lookup

    @property
    def samples(self):
        "DataFrame with all samples"
        return self._samples

    @property
    def genome(self):
        return self._genome

    @property
    def aligner(self):
        return self._aligner

    @property
    def raw_samples(self):
        return self._raw_samples

    @property
    def aligned_samples(self):
        return self._aligned_samples

    @property
    def genes_all(self):
        return self._genes_all

    @property
    def genes_used(self):
        return self._genes_used

    @property
    def postprocessor(self):
        return self._postprocessor

    @property
    def fastq_processor(self):
        return self._fastq_processor

    @property
    def raw_counter(self):
        return self._raw_counter

    @property
    def normalizer(self):
        return self._normalizer

    @property
    def differential(self):
        return self._differential

    @property
    def counters(self):
        return self._counters

    @property
    def genes(self):
        "Collection of all wrapped genes object"
        return self._genes_collection

    @property
    def report(self):
        return self._report

    @property
    def combinations(self):
        return self._combinations

    @property
    def comparison_frames(self):
        """A dictionary of data frames specifying the comparisons to be made,"""
        return self._comparison_frames

    def _set_comparison_frames(self):
        self._comparison_frames = self.load_comparison_frames()

    def load_comparison_frames(self):
        comparison_dataframes = {}
        for comparison in self.analysis.comparison:
            infile = self.analysis.filepath_from_incoming(
                self.analysis.comparison[comparison]["file"]
            )
            self.logger.info("Loading comparison: %s" % infile)
            comparison_dataframes[comparison] = pd.read_csv(infile, sep="\t")
        return comparison_dataframes

    def get_fastq_processor_from_analysis(self):
        """
        Returns an appropriate fastq processor.

        This is based on the kit provided.

        Returns
        -------
        Any
            align.fastq2. class instance.
        """
        kit = self.analysis.samples["kit"]
        if kit == "QuantSeq":
            return align.fastq2.UMIExtractAndTrim(
                umi_length=6, cut_5_prime=4, cut_3_prime=0
            )
        elif kit == "NextSeq":
            return align.fastq2.Straight()
        else:
            raise NotImplementedError  # TODO: read processor from toml for more fine-grained control

    def get_aligner_from_analysis(self):
        """
        Sets an instance of the specified aligner and parameters for the run.

        This looks up the aligner classes in externals.aligners and sets
        an instance of the specified aligner as instance property, if such a
        class exists.

        Raises
        ------
        ValueError
            If the aligner name does not match to a class in the module.
        """
        aligner_ = get_class_from_module(
            "mbf.externals.aligners", self.analysis.alignment["aligner"]
        )
        aligner = aligner_()
        params = self.analysis.alignment.get("parameters", {})
        return aligner, params

    def set_aligner(self):
        """
        set_aligner sets the aligner to be used as class attribute.

        In addition, sets the parameters for the aligner as class attribute.
        """
        aligner, params = self.get_aligner_from_analysis()
        self._aligner = aligner
        self._aligner_params = params
        self.logger.info(f"Using aligner: {self._aligner.__class__.__name__}")

    def create_samples(self):
        """
        create_samples initializes the samples to be used.
        """
        # define samples
        raw_lanes = {}
        self.logger.info("Initializing raw Samples")
        for _, row in self.samples.iterrows():
            sample_name = str(row["sample"])
            raw_lanes[sample_name] = self.create_raw(sample_name, row)
        self._raw_samples = raw_lanes

    def construct_path_prefix(self, row: Series, run_id: str) -> str:
        ret = f"{self.analysis.project['incoming']}/{run_id}/{row['prefix']}"
        return ret

    def create_raw(self, sample_name: str, row: Series) -> Sample:
        """
        create_raw initializes a single sample from a row in the samples DataFrame.

        Parameters
        ----------
        sample_name : str
            _Name of the sample.
        row : Series
            Row containing sample information.

        Returns
        -------
        Sample
            Raw mbf sample.
        """
        sample = mbf.align.raw.Sample(
            sample_name,
            mbf.align.strategies.FASTQsJoin(
                [
                    mbf.align.strategies.FASTQsFromPrefix(
                        self.construct_path_prefix(row, run_id)
                    )
                    for run_id in self.analysis.run_ids
                ]
            ),
            reverse_reads=self.analysis.samples["reverse_reads"],
            fastq_processor=self.analysis.fastq_processor(),
            pairing="auto",
            vid=row["vid"],
        )
        return sample

    def align(self):
        """Start alignments for all samples"""
        if not hasattr(self, "raw_samples"):
            raise ValueError("No raw samples were defined. Call create_samples first.")
        aligned_lanes = {}
        self.logger.info("Initializing AlignedSamples")
        for sample_name in self.raw_samples:
            aligned_lanes[sample_name] = self.align_lane(self.raw_samples[sample_name])
        self._aligned_samples = aligned_lanes

    def align_lane(self, raw_sample):
        """Create AlignedLane for a single sample"""
        unprocessed = raw_sample.align(self.aligner, self.genome, self._aligner_params)
        if self.postprocessor is not None:
            return unprocessed.post_process(self.postprocessor)
        else:
            return unprocessed

    def get_raw_counter_from_kit(self):
        kit = self.analysis.samples["kit"]
        stranded = self.analysis.samples["stranded"]
        if kit == "QuantSeq":
            if stranded:
                return mbf.genomics.genes.anno_tag_counts.ExonSmartStrandedRust
            else:
                raise NotImplementedError
        elif kit == "NextSeq":
            if stranded:
                return mbf.genomics.genes.anno_tag_counts.ExonSmartStrandedRust
            else:
                return mbf.genomics.genes.anno_tag_counts.ExonSmartUnstrandedRust
        else:
            raise NotImplementedError  # TODO: to toml for more fine-grained control

    def _add_raw_to_counters(self):
        self._counter_lookup["raw"] = {}
        self._counters["raw"] = {}
        for sample_name in self.raw:
            annotator = self.raw[sample_name]
            self._counter_lookup["raw"][sample_name] = annotator.columns[0]
            self._counters["raw"][sample_name] = annotator

    def count(self):
        """count all the lanes, must be done after alignment"""
        if not hasattr(self, "aligned_samples"):
            raise ValueError("Please call count after alignment.")
        self.logger.info("Adding raw counters")
        self.set_raw()
        self._add_raw_to_counters()
        for sample_name in self.raw:
            self.genes_all.add_annotator(self.raw[sample_name])

    def get_raw(self) -> Dict[str, mbf.genomics.annotator.Annotator]:
        """Generates a dictionary of count annotators for raw counts."""
        raw = {}
        for sample_name in self.raw_samples:
            raw[sample_name] = self.raw_counter(self.aligned_samples[sample_name])
        return raw

    def set_raw(self):
        """Sets self.raw as a dictionary of count annotators for raw counts."""
        self.raw = self.get_raw()

    def get_normalize_from_analysis(self) -> Dict[str, _FastTagCounter]:
        counters = {}
        if self.analysis.normalization is not None:
            for key in self.analysis.normalization:
                counter_class = self.analysis.normalization[key]
                counters[key] = get_class_from_module(
                    "mbf.genomics.genes.anno_tag_counts", counter_class
                )
        return counters

    def get_norm_counter_from_kit(self) -> Dict[str, _NormalizationAnno]:
        kit = self.analysis.samples["kit"]
        stranded = self.analysis.samples["stranded"]
        if stranded:
            if kit == "QuantSeq":
                return {"CPM": mbf.genomics.genes.anno_tag_counts.NormalizationCPM}
            elif kit == "NextSeq":
                return {"TPM": mbf.genomics.genes.anno_tag_counts.NormalizationTPM}
            else:
                raise NotImplementedError  # TODO: to toml for more fine-grained control
        else:
            raise NotImplementedError

    def get_normalizer(self) -> Dict[str, _NormalizationAnno]:
        """Returns a Normalized count annotator class based in the input toml."""
        norm = self.get_normalize_from_analysis()
        if len(norm) == 0:
            norm = self.get_norm_counter_from_kit()

        return norm

    def get_norm(self) -> Dict[str, Dict[str, mbf.genomics.annotator.Annotator]]:
        norm: Dict[str, Dict[str, mbf.genomics.annotator.Annotator]] = {}
        for norm_key in self.normalizer:
            norm[norm_key] = {}
            for sample_name in self.raw:
                normalizer = self.normalizer[norm_key]
                norm[norm_key][sample_name] = normalizer(self.raw[sample_name])
        return norm

    def set_norm(self):
        self.norm = self.get_norm()

    def _init_norm_annotator(self):
        for norm_key in self.norm:
            self._counter_lookup[norm_key] = {}
            self._counters[norm_key] = {}
            for sample_name in self.norm[norm_key]:
                annotator = self.norm[norm_key][sample_name]
                self._counter_lookup[norm_key][sample_name] = annotator.columns[0]
                self._counters[norm_key][sample_name] = annotator

    def normalize(self):
        if not hasattr(self, "raw"):
            raise ValueError(
                "Normalization called before counting. Please call count() first."
            )
        self.logger.info("Adding normalization counters")
        self.set_norm()
        self._init_norm_annotator()
        for norm_key in self.norm:
            for sample_name in self.norm[norm_key]:
                self.genes_all.add_annotator(self.norm[norm_key][sample_name])

    def _interpret_threshold_filter(self, filter_arg) -> Union[str, List[str]]:
        columns, operator, value = filter_arg
        if isinstance(columns, list):
            return filter_arg
        elif columns == "raw":
            if not hasattr(self, "raw"):
                raise ValueError("No raw columns found, call runner.count() first.")
            filter_arg = [
                [column for annotator in self.raw for column in annotator],
                operator,
                value,
            ]
        elif columns in self.normalizer:
            if not hasattr(self, "norm"):
                raise ValueError("norm not present!")
            filter_arg = [
                [
                    column
                    for annotator in self.norm[columns].values()
                    for column in annotator.columns
                ],
                operator,
                value,
            ]
        else:
            raise ValueError(f"Could not interpret filter spec: {list(filter_arg)}.")
        return filter_arg

    def __get_chromosome_filter_arg(self, chromosomes: Union[str, List[str]]):
        filter_arg = ["chr", "in"]
        if chromosomes == "canonical":
            filter_argument = [str(i) for i in range(1, 24)] + ["X", "Y", "MT"]
        elif isinstance(chromosomes, str):
            filter_argument = [chromosomes]
        elif isinstance(chromosomes, list):
            filter_argument = chromosomes
        filter_arg.append(filter_argument)
        return filter_arg

    def generate_custom_filter(self) -> Filter:
        filter_args = []
        filter_spec = self.analysis.genes["filter"]
        if "chr" in filter_spec:
            filter_args.append(self.__get_chromosome_filter_arg(filter_spec["chr"]))
        if "biotypes" in filter_spec:
            filter_args.append(["biotype", "in", list(filter_spec["biotypes"])])
        if "thresholds" in filter_spec:
            for filter_arg in filter_spec["thresholds"]:
                filter_args.append(self._interpret_threshold_filter(filter_arg))
        if "drop_empty_names" in filter_spec:
            filter_args.append([["name"], "notin", ["", "NaN"]])
        custom_filter = Filter(filter_args)
        return custom_filter

    def generate_comparison_prefilter(self, prefilter: List, row: pd.Series) -> Filter:
        filter_args_a = []
        filter_args_b = []
        for filter_args in prefilter:
            counter, operator, value = filter_args
            samples_a, samples_b = self.get_comparison_samples_by_row(row)
            columns_a = [self._counter_lookup[counter][sample] for sample in samples_a]
            columns_b = [self._counter_lookup[counter][sample] for sample in samples_b]
            filter_args_a.append(
                [
                    columns_a,
                    operator,
                    value,
                ]
            )
            filter_args_b.append(
                [
                    columns_b,
                    operator,
                    value,
                ]
            )
        custom_filter = Filter(filter_args_a)
        custom_filter |= Filter(filter_args_b)
        return custom_filter

    def get_genes_prefilter_from_analysis(self):
        __filter = None
        if self.analysis.has_gene_filter_specified():
            custom_filter = self.generate_custom_filter()
            __filter = self.get_filter_callable(custom_filter)
        return __filter

    def get_filter_callable(self, custom_filter: Filter) -> Callable:
        def __filter(df):
            return custom_filter(df).index

        return __filter

    def get_count_annotators(self):
        annotators = list(self.raw.values())
        for counter in self.norm:
            annotators.extend(list(self.norm[counter].values()))
        return annotators

    def prefilter(self):
        "filter the genes used in downstream analysis"
        self.logger.info("Filtering genes used for downstream analysis")
        genes_used = self.genes_all
        pre_filter = self.get_genes_prefilter_from_analysis()
        if pre_filter is not None:
            filtered_name = self.analysis.genes_used_name()
            genes_used = self.genes_all.filter(
                filtered_name,
                pre_filter,
                annotators=self.get_count_annotators(),
            )
        decription = f"{self.analysis.genes_used_name()} \n{self.analysis.genes_used_description()}"
        genes_wrapped = GenesWrapper(
            genes=genes_used,
            tags=["genes_used"],
            description=decription,
        )
        self._genes_collection["genes_used"] = genes_wrapped
        self._genes_used = genes_used
        for comparison_group in self.analysis.comparison:
            df = self.comparison_frames[comparison_group]
            if "prefilter" in self.analysis.comparison[comparison_group]:
                prefilter_args = self.analysis.comparison[comparison_group]["prefilter"]
                for _, row in df.iterrows():
                    comparison_name = row["comparison_name"]
                    tag = f"genes_used_{comparison_name}"
                    custom_filter = self.get_filter_callable(
                        self.generate_comparison_prefilter(prefilter_args, row)
                    )
                    genes_used_for_comparison = genes_used.filter(
                        f"{self.genes_used}_{comparison_name}",
                        custom_filter,
                        annotators=self.get_count_annotators(),
                    )
                    decription = (
                        f"{genes_used_for_comparison} \n filtered by {prefilter_args}"
                    )
                    genes_wrapped_comparison = GenesWrapper(
                        genes=genes_used_for_comparison,
                        tags=[tag],
                        description=decription,
                    )
                    self._genes_collection[tag] = genes_wrapped_comparison
            else:
                for _, row in df.iterrows():
                    comparison_name = row["comparison_name"]
                    tag = f"genes_used_{comparison_name}"
                    genes_wrapped.add_tag(tag)
                    genes_wrapped_comparison = genes_wrapped
                    self._genes_collection[tag] = genes_wrapped_comparison

    def _set_comparison_name_group_lookup(self):
        self.comparison_name_group_lookup = {}
        for comparison_group in self.comparison_frames:
            length = self.comparison_frames[comparison_group].shape[0]
            self.comparison_name_group_lookup.update(
                dict(
                    zip(
                        self.comparison_frames[comparison_group][
                            "comparison_name"
                        ].values,
                        [comparison_group] * length,
                    )
                )
            )

    def _set_differential_transformer(self):
        """Sets a dictionary of differential transformer wrapper to be used in comparisons"""
        self._differential = {}
        for comparison_group in self.analysis.comparison:
            self._differential.update(self.get_differential_for_group(comparison_group))

    def get_differential_for_group(self, comparison_group: str):
        """Returns a dictionary of differential transformers for a comparison_group based on the comparison type."""
        comparison_type = self.analysis.comparison[comparison_group]["type"]
        if comparison_type == "ab":
            differential_wrapper_generator = self.differential_wrapper_ab
        elif comparison_type == "timeseries":
            differential_wrapper_generator = self.differential_wrapper_timeseries
        else:
            raise NotImplementedError(
                f"Cannot interpret comparison type {self.analysis.comparison[comparison_group]['type']}."
            )
        return self.get_differential_transformers(
            comparison_group, differential_wrapper_generator
        )

    def get_differential_transformers(
        self, comparison_group: str, differential_wrapper_generator: Callable
    ):
        """Creates a dictionary of sdifferential transformers to be used for a comparison_group."""
        df = self.comparison_frames[comparison_group]
        diff_transformers = {}
        for _, row in df.iterrows():
            deg = differential_wrapper_generator(row, comparison_group)
            diff_transformers[row["comparison_name"]] = deg
        return diff_transformers

    def differential_wrapper_ab(
        self, row: pd.Series, comparison_group: str
    ) -> DifferentialWrapper:
        """Generates a differential transformer for ab type comparisons."""
        self.logger.info(f"request comparison: {comparison_group}")
        counter = self.analysis.comparison[comparison_group].get("counter", "raw")
        samples_a, samples_b = self.get_comparison_samples_by_row(row)
        self.logger.info(
            f"samples_a for comparison {comparison_group}: {samples_a}",
        )
        self.logger.info(
            f"samples_b for comparison {comparison_group}: {samples_b}",
        )
        columns_a = [self._counter_lookup[counter][sample] for sample in samples_a]
        columns_b = [self._counter_lookup[counter][sample] for sample in samples_b]
        columns_other = [
            self._counter_lookup[counter][sample]
            for sample in self.samples["sample"]
            if sample not in samples_a + samples_b
        ]
        condition_to_columns = {
            row["a"]: columns_a,
            row["b"]: columns_b,
            "other": columns_other,
        }
        arguments = [row["a"], row["b"], condition_to_columns, row["comparison_name"]]
        transformer = self.generate_transformer(comparison_group, arguments)
        input_columns = columns_a + columns_b
        if transformer.name == "NOISeq":
            input_columns += ["chr", "start", "stop", "biotype"]
        deg = DifferentialWrapper(
            name=f"{row['comparison_name']}({comparison_group})",
            comparison_group=comparison_group,
            transformer=transformer,
            counter=counter,
            samples=samples_a + samples_b,
            input_columns=input_columns,
            dependencies=[],
            annotators=list(self.counters[counter].values()),
        )
        return deg

    def differential_wrapper_timeseries(
        self, row: pd.Series, comparison_group: str
    ) -> DifferentialWrapper:
        counter = self.analysis.comparison[comparison_group].get("counter", "raw")
        formula = self.analysis.comparison[comparison_group]["formula"]
        comparison_name = row["comparison_name"]
        reduced = self.analysis.comparison[comparison_group]["reduced"]
        factors = row["factors"]
        samples = self.get_samples_by_factors([row["groupby"]], [row["values"]])
        arguments = [samples, factors, formula, reduced, comparison_name]
        transformer = self.generate_transformer(comparison_group, arguments)
        deg = DifferentialWrapper(
            name=f"{comparison_name}({comparison_group})",
            comparison_group=comparison_group,
            transformer=transformer,
            counter=counter,
            samples=samples,
            input_columns=list(self.columns_lookup[counter].values()),
            dependencies=[],
            annotators=list(self.counters[counter].values()),
        )
        return deg

    def generate_transformer(self, comparison_group, arguments) -> _Transformer:
        """
        Initializes a differential Transformer based on the input.toml for a
        certain group asnd returns it.
        """
        method_class = self.analysis.comparison[comparison_group]["method"]
        transformer = get_class_from_module("mdataframe.differential", method_class)
        parameters = self.analysis.comparison[comparison_group].get("parameters", {})
        # laplace_offset = self.comparison[comparison_group].get("laplace_offset", 0)
        # include_other_samples_for_variance = self.comparison[comparison_group].get(include_other_samples_for_variance, True)
        differential_transformer = transformer(*arguments, **parameters)
        return differential_transformer

    def get_comparison_samples_by_row(self, row):
        factors = row["groupby"].split(",")
        values_a = row["a"].split(",")
        values_b = row["b"].split(",")
        samples_a = list(self.get_samples_by_factors(factors, values_a))
        samples_b = list(self.get_samples_by_factors(factors, values_b))
        return samples_a, samples_b

    def get_samples_by_factors(self, factors, values) -> List[str]:
        index = self.samples[
            (self.samples[factors] == values).all(axis="columns")
        ].index
        samples = self.samples.loc[index]["sample"].values
        return samples

    def deg(self):
        """use genes_used and perform all differential expression analysis. This includes statistics, volcano and heatmap plots."""
        self.logger.info("Setting up differential expression analysis")
        self._set_differential_transformer()
        # self.differential_annotators = {}
        for comparison_name in self.differential:
            differential = self.differential[comparison_name]
            # annotator = FromTransformerWrapper(differential)
            tag = f"genes_used_{comparison_name}"
            genes_used_for_comparison = self.genes[tag]
            genes_used_for_comparison.genes.add_annotator(differential)
            # self.genes_used.add_annotator(differential)  # add different genes_used here
            # self.differential_annotators[comparison_name] = annotator

    def __get_annos_deps_from_differential(
        self, comparison_name: str
    ) -> Tuple[List[Annotator], List[Job]]:
        differential = self.differential[comparison_name]
        # annotators = [self.differential[comparison_name]]
        dependencies = differential.dependencies
        # return annotators, dependencies
        return [differential], dependencies

    def __create_volcano_module_arguments(
        self, comparison_name: str, **parameters
    ) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        differential = self.differential[comparison_name]
        module_parameters = {
            "fc_threshold": parameters.pop(
                "fc_threshold", VolcanoModule.default_threshold
            ),
            "alpha": parameters.pop("alpha", VolcanoModule.default_alpha),
            "logFC": differential.transformer.logFC,
            "p": differential.transformer.P,
            "fdr": differential.transformer.FDR,
            "title": f"Volcano {comparison_name}",
        }
        module_parameters.update(parameters)
        genes_parameters = {
            "columns": differential.columns,
        }
        outfile = Path(f"{comparison_name}.volcano.png")
        inputs = {"df": None}
        description = f"Volcano plot for {comparison_name}"
        return [outfile, inputs, description], module_parameters, genes_parameters

    def get_counter_columns_to_sample(
        self, samples: List[str], counter_name: str
    ) -> OrderedDict[str, str]:
        columns = [self.columns_lookup[counter_name][sample] for sample in samples]
        rename = OrderedDict(zip(columns, samples))
        return rename

    def __create_pca_module_arguments(
        self, prefix: str, samples: List[str], counter_name: str
    ) -> Tuple[List[Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """This is so ugly"""
        rename_columns = self.get_counter_columns_to_sample(samples, counter_name)
        genes_parameter = {
            "columns": list(rename_columns.keys()),
            "rename_columns": rename_columns,
        }
        inputs = {"df": None}
        if self.analysis.factors is not None:
            df_groups = self.samples.copy()
            df_groups = df_groups.set_index("sample")
            df_groups["condition"] = df_groups[self.analysis.factors].apply(
                "_".join, axis=1
            )
        else:
            df_groups = self.samples.copy()
            df_groups["condition"] = df_groups["sample"].values
            df_groups = df_groups.set_index("sample")
        inputs["df_samples"] = df_groups[["condition"]]
        outfile = Path(f"{prefix}.{counter_name}.pca.png")
        description = f"PCA on {prefix} samples using {counter_name} values"  # this is for the report
        return (
            [outfile, inputs, description],
            {"title": f"PCA {prefix} ({counter_name})"},
            genes_parameter,
        )

    def __create_heatmap_module_arguments(
        self,
        prefix: str,
        samples: List[str],
        counter_name: str,
        sort_by: Optional[str] = None,
    ) -> Tuple[List[Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        rename_columns = self.get_counter_columns_to_sample(samples, counter_name)
        genes_parameter = {
            "columns": list(rename_columns.keys()),
            "rename_columns": rename_columns,
            "sort_by": sort_by,
        }
        inputs = {
            "df": None,
        }
        outfile = Path(f"{prefix}.{counter_name}.heatmap.png")
        description = f"Heatmap on {prefix} samples using {counter_name} values"
        return (
            [outfile, inputs, description],
            {
                "add": False,
                "sort": True,
                "title": f"Heatmap {prefix} ({counter_name})",
                # "show_column_label": True,
                # "show_row_label": True,
                "cluster_params": {"n_clusters": 2},
                # "sort_by": sort_by,
            },
            genes_parameter,
        )

    def register_defaults(self) -> None:
        """register default jobs with the genes"""
        all_comparisons = list(self.differential.keys())
        self.register_volcano("genes_used")
        for comparison_name in self.differential:
            tag = f"genes_used_{comparison_name}"
            self.register_volcano(tag, comparison_names=[comparison_name])
            self.register_pca(comparison_name, comparisons=comparison_name)
            self.register_pca(tag, comparisons=comparison_name)
        self.register_pca("genes_used")
        for comparison_name in self.differential:
            self.register_heatmap(
                comparison_name,
                comparisons=[comparison_name],
                sort_by_comparison=comparison_name,
            )
        if self.combinations is not None:
            self.register_pca("combined", comparisons=all_comparisons)
            self.register_heatmap("combined", comparisons=all_comparisons)

    def volcano_columns_present(self, transformer: _Transformer) -> bool:
        present = all(hasattr(transformer, col) for col in ["logFC", "P", "FDR"])
        return present

    def register_volcano(
        self, tag: str, comparison_names: Optional[List[str]] = None, **parameters
    ) -> None:
        if comparison_names is None:
            comparison_names = list(self.differential.keys())
        for comparison_name in comparison_names:
            if self.volcano_columns_present(
                self.differential[comparison_name].transformer
            ):
                (
                    module_args,
                    module_kwargs,
                    genes_parameter,
                ) = self.__create_volcano_module_arguments(comparison_name)
                annotators, dependencies = self.__get_annos_deps_from_differential(
                    comparison_name
                )
                self.logger.info(
                    f"Registering volcano plot for tag '{tag}' on {comparison_name}"
                )
                self.genes.register_default_module_for_tag(
                    tag,
                    VolcanoModule,
                    module_args,
                    module_kwargs,
                    genes_parameter=genes_parameter,
                    annotators=annotators,
                    dependencies=dependencies,
                )

    def register_heatmap(
        self,
        tag: str,
        counter: Optional[str] = None,
        comparisons: Optional[Union[bool, str, List[str]]] = False,
        sort_by_comparison: Optional[str] = None,
    ) -> None:
        counters = self.__get_counters(counter)
        samples_to_plot = self.__get_samples_for_pca(comparisons)
        for counter_name in counters:
            annotators = [annotator for annotator in self.norm[counter_name].values()]
            for prefix in samples_to_plot:
                dependencies: List[Job] = []
                sort_by = None
                if sort_by_comparison is not None:
                    (
                        annotators_deg,
                        dependencies,
                    ) = self.__get_annos_deps_from_differential(sort_by_comparison)
                    sort_by = self.differential[sort_by_comparison].transformer.logFC
                    annotators.extend(annotators_deg)
                    dependencies.extend(dependencies)
                (
                    module_args,
                    module_kwargs,
                    genes_parameter,
                ) = self.__create_heatmap_module_arguments(
                    prefix,
                    samples_to_plot[prefix],
                    counter_name,
                    sort_by=sort_by,
                )
                self.logger.info(f"Registering heatmap for tag '{tag}' on {prefix}")
                self.genes.register_default_module_for_tag(
                    tag,
                    HeatmapModule,
                    module_args,
                    module_kwargs,
                    genes_parameter=genes_parameter,
                    annotators=annotators,
                    dependencies=dependencies,
                )

    def register_pca(
        self,
        tag: str,
        counter: Optional[Union[str, List[str]]] = None,
        comparisons: Optional[Union[bool, str, List[str]]] = False,
    ) -> None:
        """PCA can be done for any Normgroup ... register PCA for all columns and for differential columns"""
        counters = self.__get_counters(counter)
        samples_to_plot = self.__get_samples_for_pca(comparisons)
        for counter_name in counters:
            annotators = [annotator for annotator in self.norm[counter_name].values()]
            for prefix in samples_to_plot:
                (
                    module_args,
                    module_kwargs,
                    genes_parameter,
                ) = self.__create_pca_module_arguments(
                    prefix, samples_to_plot[prefix], counter_name
                )
                if len(genes_parameter["columns"]) < 3:
                    continue
                self.logger.info(f"Registering PCA for tag '{tag}' on {prefix} counts")
                self.genes.register_default_module_for_tag(
                    tag,
                    PCAModule,
                    module_args,
                    module_kwargs,
                    genes_parameter=genes_parameter,
                    annotators=annotators,
                    dependencies=[],
                )

    def __get_counters(
        self, counter: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        if counter is None:
            counters = list(self.norm.keys())  # if no name specified, use all
        elif isinstance(counter, list):
            counters = counter
        elif isinstance(counter, str):
            counters = [counter]
        return counters

    def __get_samples_for_pca(
        self, comparisons: Optional[Union[bool, str, List[str]]] = False
    ) -> Dict[str, List[str]]:
        samples_to_plot = {"all": list(self.raw_samples.keys())}
        if comparisons:
            if isinstance(comparisons, list):
                comparison_names = comparisons
            elif isinstance(comparisons, str):
                comparison_names = [comparisons]
            else:
                comparison_names = self.differential.keys()
            for comparison_name in comparison_names:
                differential = self.differential[comparison_name]
                samples = differential.samples
                samples_to_plot[comparison_name] = samples
        return samples_to_plot

    def filter(self):
        self.logger.info("Filtering DE genes")
        for comparison_name in self.differential:
            comparison_group = self.comparison_name_group_lookup[comparison_name]
            filter_expressions = self.analysis.deg_filter_expressions(comparison_group)
            for filter_expr in filter_expressions:
                suffix = self.analysis.deg_filter_expression_as_str(filter_expr)
                new_name = "_".join([comparison_name, suffix])
                path_filtered = (
                    self.filtered_path(
                        self.genes[f"genes_used_{comparison_name}"].genes
                    )
                    / new_name
                )
                genes_filtered = self.__filter_genes(
                    new_name, comparison_name, filter_expr, path_filtered
                )
                description = f"Comparison {comparison_name}, using {self.differential[comparison_name].transformer.name}\n Genes filtered by {suffix}\n"
                genes_filtered_wrapped = GenesWrapper(
                    genes_filtered,
                    tags=["filtered", comparison_group, comparison_name, new_name],
                    description=description,
                )
                self._genes_collection[new_name] = genes_filtered_wrapped

    def __filter_genes(
        self, new_name: str, comparison_name: str, filter_expr: List[str], path: Path
    ):
        differential_filter = self.__get_differential_filter(
            comparison_name, filter_expr
        )
        annotators = list(self.raw.values()) + [self.differential[comparison_name]]
        for counter_name in self.norm:
            annotators.extend(list(self.norm[counter_name].values()))
        genes_used = self.genes[f"genes_used_{comparison_name}"]
        genes_filtered = genes_used.genes.filter(
            new_name, differential_filter, annotators=annotators, result_dir=path
        )
        return genes_filtered

    def __get_differential_filter(self, comparison_name, filter_expressions):
        transformer = self.differential[comparison_name]
        filter_args = [
            self.interpret_filter_expression(expression, transformer)
            for expression in filter_expressions
        ]
        differential_filter = Filter(filter_args)

        def __filter(df):
            return differential_filter(df).index

        return __filter

    def interpret_filter_expression(self, filter_expression, transformer):
        column, operator, value = filter_expression
        column = getattr(transformer, column)

        return [[column], operator, value]

    def combine(self):
        """Process the primary gene sets and create more refined, e.g. set operations."""
        combined_genes = self.generate_combinations()
        self._genes_collection.update(combined_genes)

    def get_tags_from_parents(
        self, genes_wrapped_to_combine: List[GenesWrapper]
    ) -> Set[str]:
        inherited_tags = set()
        for gw in genes_wrapped_to_combine:
            new_tags = gw.tags.copy()
            new_tags.discard(gw.genes.name)
            new_tags.discard("filtered")
            inherited_tags.update(new_tags)
        return inherited_tags

    def combine_genes(
        self, new_name: str, genes_names_to_combine: List[str], operation: str
    ) -> Genes:
        genes_wrapped_to_combine = [
            self.genes[gene_name] for gene_name in genes_names_to_combine
        ]
        generator = self.analysis.get_generator(operation)
        combined = generator(
            new_name, [g.genes for g in genes_wrapped_to_combine], self.combined_path()
        )
        inherited_tags = self.get_tags_from_parents(genes_wrapped_to_combine)
        description = (
            f"{new_name}, combined using {operation} on:\n{genes_names_to_combine}"
        )
        combined_wrapped = GenesWrapper(
            combined,
            tags=["combined", new_name] + list(inherited_tags),
            description=description,
        )
        return combined_wrapped

    def generate_combinations(self):
        combined_genes = {}
        if self.combinations is not None:
            self.logger.info("Performing set combinations of genes")
            for _, row in self.combinations.iterrows():
                new_name = row["combined_name"]
                combined_wrapped = self.combine_genes(
                    new_name, row["gene_names"].split(","), row["operation"]
                )
                combined_genes[new_name] = combined_wrapped
        return combined_genes

    def combined_path(self):
        return self.results / "Genes" / self._genes_used.name / "combined"

    def differential_path(self):
        return self.results / "Genes" / self._genes_used.name / "differential"

    def filtered_path(self, genes_used):
        return self.results / "Genes" / genes_used.name / "filtered"

    def _load_combinations(self):
        df_combinations = None
        if "file" in self.analysis.combination:
            infile = self.analysis.filepath_from_incoming(
                self.analysis.combination["file"]
            )
            self.logger.info(f"Loading combinations from {infile}")
            df_combinations = pd.read_csv(infile, sep="\t")
        return df_combinations

    def _set_combinations(self):
        self._combinations = self._load_combinations()

    def filtered_genes_to_analyze(self, genes_to_select=None) -> Dict:
        if "filtered" not in self.genes.tags:
            raise ValueError(
                "No filtered genes to perform ORA on. call filter() first."
            )
        if "combined" not in self.genes.tags:
            self.logger.warn(
                "No set combinations to perform on. ORA is performed for filtered genes only."
            )
        if genes_to_select is None:
            genes_to_analyze = self.genes.genes_by_tag("filtered")
            if "combined" in self.genes.tags:
                genes_to_analyze += self.genes.genes_by_tag("combined")
        else:
            genes_to_analyze = [
                self.genes[genes_name] for genes_name in genes_to_select
            ]
        return genes_to_analyze

    def run_ora(
        self, genes_to_select: Optional[List] = None
    ) -> Dict[str, Dict[str, Job]]:
        genes_to_analyze = self.filtered_genes_to_analyze(genes_to_select)
        ora = ORA(self.genes_used, genes_to_analyze, self.analysis)
        return ora.run_ora()

    def _load_gseas(self):
        if "file" in self.analysis.pathways["gsea"]:
            infile = self.analysis.filepath_from_incoming(
                self.analysis.pathways["gsea"]["file"]
            )
            self._gseas = pd.read_csv(infile, sep="\t")
        else:
            self._gseas = None

    @property
    def gseas(self):
        return self._gseas

    def get_columns_ab_for_gsea_by_row(self, row):
        counter = self.analysis.pathways["gsea"]["counter"]
        factors = row["groupby"].split(",")
        values_a = row["a"].split(",")
        values_b = row["b"].split(",")
        samples_a = self.get_samples_by_factors(factors, values_a)
        samples_b = self.get_samples_by_factors(factors, values_b)
        return (
            [self.norm[counter][sample].columns[0] for sample in samples_a],
            [self.norm[counter][sample].columns[0] for sample in samples_b],
        )

    def get_gsea_parameters_by_row(self, row):
        return row[
            row.index.difference(self.analysis.gsea_required_columns())
        ].to_dict()

    def get_gsea_parameter(self, row):
        parameter = self.analysis.get_gsea_parameter().copy()
        parameter.update(self.get_gsea_parameters_by_row(row))
        return parameter

    def run_gsea(self) -> Dict[str, Dict[str, Tuple[Job, Path]]]:
        collections = self.analysis.get_collections_for_runner("gsea")
        counter = self.analysis.pathways["gsea"]["counter"]
        annotators = list(self.norm[counter].values())
        jobs_and_index: Dict[str, Dict[str, Tuple[Job, Path]]] = {}
        for _, row in self.gseas.iterrows():
            comparison_name = row["comparison_name"]
            phenotypes = row["phenotypes"].split(",")
            columns_ab = self.get_columns_ab_for_gsea_by_row(row)
            parameter = self.get_gsea_parameter(row)
            jobs_and_index[comparison_name] = {}
            for collection in collections:
                collection_name = (
                    ",".join(collection) if isinstance(collection, list) else collection
                )
                job, index_html = self.gsea.run_on_counts(
                    self.genes_used,
                    comparison_name=comparison_name,
                    phenotypes=phenotypes,
                    columns_a_b=columns_ab,
                    collection=collection,
                    genome=self.genome,
                    annotators=annotators,
                    dependencies=[self.analysis.df_gsea_fileinvariant()],
                    **parameter,
                )
                jobs_and_index[comparison_name][collection_name] = (job, index_html)
        return jobs_and_index

    def pathways(self):
        """perform pathway analysis"""
        for pathway_method in self.analysis.pathways:
            if pathway_method == "ora":
                gene_names_to_jobs = self.run_ora()
                self.__ora_results = gene_names_to_jobs
            if pathway_method == "gsea":
                jobs_and_html = self.run_gsea()
                self.__gsea_results = jobs_and_html

            # for item in items:
            #     nb.register_item(item)
            #     break

    def analyze_genes(self):
        self.register_defaults()
        jobs = []
        for genes in self.genes.values():
            jobs.extend(genes.jobs())
        return jobs

    def everything(self):
        self.create_samples()
        self.align()
        self.count()
        self.normalize()
        self.prefilter()
        self.deg()
        self.filter()
        self.combine()
        self.write_genes()
        self.analyze_genes()
        self.pathways()
        self.generate_report()

    def write_genes(self, mangler_function=None):
        for genes in set(self.genes.values()):
            self.logger.info(f"Writing genes {genes.name}")
            genes.write(mangler_function)

    def write_genes_by_name(
        self, genes_name="genes_all", output_filename=None, mangler_function=None
    ):
        genes = self.genes[genes_name].genes
        genes.write(output_filename, mangler_function)

    def write_filtered(self, mangler_function=None):
        self.write_genes_with_tag("filtered", mangler_function=mangler_function)

    def write_combined(self, mangler_function=None):
        self.write_genes_with_tag("combined", mangler_function=mangler_function)

    def write_genes_with_tag(self, tag, mangler_function=None):
        for gene_wrapper in self.genes.genes_by_tag(tag):
            gene_wrapper.write(mangler_function)

    def generate_report(self):
        self.logger.info("Generating report")
        self.report.register_item(
            MarkdownItem("Top", self.analysis.summary_markdown(), color=False)
        )
        tags = ["genes_used"] + list(self.differential.keys())
        if self.combinations is not None:
            tags.append("combined")
        for tag in tags:
            if tag in self.differential:
                md = MarkdownItem(
                    section=tag,
                    text="### Differential Expression Analysis {tag}",
                )
            elif tag == "combined":
                md = MarkdownItem(
                    section=tag,
                    text="### Set combinations on genes",
                )
            for genes_wrapped in self.genes.genes_by_tag(tag):
                genes_name = genes_wrapped.genes.name
                md = MarkdownItem(
                    section=tag,
                    text=genes_wrapped.decription,
                )
                self.report.register_item(md)

                for module_name in genes_wrapped.modules:
                    module = genes_wrapped.modules[module_name]
                    module_job = genes_wrapped.get_module_job(module_name)
                    if module.is_figure:
                        pl = PlotItem(
                            section=tag,
                            job=module_job,
                            text=module.description,
                        )
                        self.report.register_item(pl)
                    else:
                        raise NotImplementedError
                # add ORAs if present:
                if genes_name in self.__ora_results:
                    for collection in self.__ora_results[genes_name]:
                        pl = PlotItem(
                            section=tag,
                            job=self.__ora_results[genes_name][collection][0],
                            text=f"#### Over-Representation Analysis using hypergeometric test on DE genes from {genes_name} with collection {collection}",
                        )
                        self.report.register_item(pl)
                # add gsea
            if tag in self.__gsea_results:  # if tag is a comparison_name
                for collection in self.__gsea_results[tag]:
                    job, index_html = self.__gsea_results[tag][collection]
                    html = GSEAHTMLItem(
                        section=tag,
                        filename=index_html,
                        job=job,
                    )
                    self.report.register_item(html)
        self.report.register_item(
            MarkdownItem("Bottom", self.analysis.specification(), color=False)
        )
        self.report.write()
        self.report.convert()


def get_class_from_module(module_name: str, class_to_check: str):
    module = sys.modules[module_name]
    if not hasattr(module, class_to_check):
        raise ValueError(f"No class {class_to_check} in module {module_name}.")
    return getattr(module, class_to_check)
