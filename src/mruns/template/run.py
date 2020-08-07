#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py: analysis script.
Created on July 14, 2020
"""
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any
from mpathways import GSEA, GMTCollection, MSigChipEnsembl, MSigDBCollection, ORAHyper
from mdataframe import MDF, ClassLabel
from mplots import volcanoplot
from mreports import NB
from mruns import analysis, Defaults
from mbf_genomics.genes.anno_tag_counts import _NormalizationAnno, _FastTagCounter
import pandas as pd
import pypipegraph as ppg
import mbf_genomes
import mbf_align
import mbf_genomics
import mbf_externals
import mbf_comparisons
import logging
import pandas as pd
import numpy as np
import mpathways
import mbf_publish
import inspect
from pprint import PrettyPrinter


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


ppg.new_pipegraph(log_file='logs/ppg_run.log', log_level=logging.DEBUG)
ana = analysis()
nb = ana.report()

nb.register_markdown(ana.summary_markdown())
print(ana.summary())

# define settings
genome = ana.genome
aligner, aligner_params = ana.aligner()
raw_counter = ana.raw_counter()
norm_counter = ana.norm_counter()

# read samples and comparisons info
df_samples = ana.sample_df()
conditions = [x for x in df_samples.columns if x.startswith("group")]
comparisons_to_do: Dict[str, List] = {}
for condition in conditions:
    comparisons_to_do[condition] = []
    df_in = pd.read_csv(f"incoming/{condition}.tsv", sep="\t")
    for _, row in df_in.iterrows():
        comparisons_to_do[condition].append(
            (row["a"], row["b"], row["comparison_name"])
        )

# define samples
raw_lanes = {}
for _, row in df_samples.iterrows():
    sample_name = row["sample"]
    raw_lanes[sample_name] = mbf_align.raw.Sample(
        str(sample_name),
        mbf_align.strategies.FASTQsJoin(
            [
                mbf_align.strategies.FASTQsFromPrefix(
                    f"incoming/{run_id}/{row['prefix']}_"
                )
                for run_id in ana.run_ids
            ]
        ),
        reverse_reads=ana.samples["reverse_reads"],
        fastq_processor=ana.fastq_processor(),
        vid=row["vid"],
    )

# generate alignments and count
aligned_lanes = {}
raw = {}
normalized = {}
genes = mbf_genomics.genes.Genes(genome)

column_names_nice = {}
for sample_name in raw_lanes:
    aligned_lanes[sample_name] = raw_lanes[sample_name].align(aligner, genome, aligner_params)
    raw[sample_name] = raw_counter(
        aligned_lanes[sample_name]
    )
    normalized[sample_name] = norm_counter(
        raw[sample_name]
    )
    genes.add_annotator(normalized[sample_name])
    genes.add_annotator(raw[sample_name])
    column_names_nice[raw[sample_name].columns[0]] = sample_name
    column_names_nice[normalized[sample_name].columns[0]] = sample_name

# filter gene list
genes_used = genes
if ana.has_gene_filter_specified():
    filtered_name = f"{genes.name}"
    if "canonical" in ana.genes["filter"] and ana.genes["filter"]["canonical"]:
        filtered_name += "_canonical"
    if "biotypes" in ana.genes["filter"]:
        at_least = ana.genes['filter'].get('at_least', 1)
        filtered_name += "_biotypes"
    if "cpm_threshold" in ana.genes["filter"]:
        threshold = ana.genes['filter']['cpm_threshold']
        filtered_name += f"_{at_least}samples>={threshold}"
    genes_used = genes.filter(
        filtered_name,
        ana.genes_filter()([normalized[sample_name].columns[0] for sample_name in normalized]),
        annotators=list(normalized.values()),
    )


# initialize some dicts
raw_groups: Dict[str, Dict] = {}
norm_groups: Dict[str, Dict] = {}

for condition_group in conditions:
    raw_groups[condition_group] = {}
    norm_groups[condition_group] = {}

for _, row in df_samples.iterrows():
    for condition_group in conditions:
        condition_name = row[condition_group]
        if condition_name not in raw_groups[condition_group]:
            raw_groups[condition_group][condition_name] = []
            norm_groups[condition_group][condition_name] = []
        raw_groups[condition_group][condition_name].append(raw[row["sample"]])
        norm_groups[condition_group][condition_name].append(normalized[row["sample"]])

class_labels_by_group: Dict[str, ClassLabel] = {}
for condition_group in conditions:
    class_label_dict = {}
    for group_name in norm_groups[condition_group]:
        for anno in norm_groups[condition_group][group_name]:
            class_label_dict[anno.columns[0]] = group_name
    class_label_group = ClassLabel(condition_group, class_label_dict)
    class_labels_by_group[condition_group] = class_label_group

# do some plots on the genes used
    pca_all = (
        MDF(
            f"PCA_{genes_used.name}_{condition_group}",
            genes_used,
            [ano.columns[0] for ano in normalized.values()],
            index_column="gene_stable_id",
            dependencies=[
                genes_used.add_annotator(anno) for anno in list(normalized.values())
            ],
            annotators=list(normalized.values()),
        )
        .impute()
        .scale()
        .reduce()
        .cluster(class_labels_by_group[condition_group], axis=1)
        .transform(
            ["rename", {"columns": column_names_nice}],
            meta_rows=False,
            meta_columns=True,
        )
    )
    filename = f"{genes_used.name}_{condition_group}.pca.png"
    nb.register_plot(
        pca_all.plot_2d(
            outfile=Path(genes_used.result_dir) / filename,
            title=f"PCA {genes_used.name}",
            class_label_column=condition_group,
            show_names=False,
            model_name="PCA",
        ),
        f"### Principle component analysis (PCA) on {genes_used.name}({condition_group})"
    )

# declare comparisons and plots
comparisons: Dict[str, Dict] = {}
filtered: Dict[str, Dict] = {}
defaults = Defaults(genes_used, ana, column_names_nice, normalized, [])
section_indices = {}
for condition_group in conditions:
    filtered[condition_group] = {}
    comparison = mbf_comparisons.Comparisons(
        genes_used, raw_groups[condition_group], name=condition_group
    )
    section_index = 0
    comparisons[condition_group] = {}
    for method_name in ana.comparison:
        comparisons[condition_group][method_name] = {}
        method, options = ana.comparison_method(method_name)
        filtered[condition_group][method_name] = {}
        for cond1, cond2, comp_name in comparisons_to_do[condition_group]:
            section_indices[condition_group, comp_name] = section_index
            comparison_ab = comparison.a_vs_b(
                cond1,
                cond2,
                method,
                laplace_offset=options["laplace_offset"],
                include_other_samples_for_variance=options["include_other_samples_for_variance"],
            )
            comparisons[condition_group][method_name][comp_name] = comparison_ab
            if options["include_other_samples_for_variance"]:
                x = "fit on all samples"
            else:
                x = "fit on conditions"
            desc = f"comparing {cond1} vs {cond2} using {method_name} (offset={options['laplace_offset']}, {x})\n"
            nb.register_markdown(
                f"### Comparison {comp_name} with {method_name}\n{desc}", section_index=section_index
            )
            filter_expressions = ana.deg_filter_expressions(method_name)
            filtered[condition_group][method_name][comp_name] = {}
            norm_sample_columns = [anno.columns[0] for c in [cond1, cond2] for anno in norm_groups[condition_group][c]]
            for filter_expr in filter_expressions:
                suffix = ana.deg_filter_expression_as_str(filter_expr)
                new_name = "_".join([comp_name, suffix])
                regulated = comparison_ab.filter(
                    filter_expr,
                    new_name=new_name
                    )
                regulated.write()
                regulated.write(output_filename=f"{regulated.name}.xlsx")
                filtered[condition_group][method_name][comp_name][suffix] = regulated
                defaults.register_genes(
                    regulated,
                    norm_sample_columns,
                    condition_group,
                    class_labels_by_group[condition_group],
                    section_index,
                    [
                        regulated.add_annotator(anno) for anno in list(normalized.values())
                    ], 
                    {method_name: comparison_ab}
                )
        section_index += 1

combinations = {}
for condition_group, new_name_prefix, comparisons_to_add, generator in ana.combinations():
    if not condition_group in combinations:
        combinations[condition_group] = {}
    for method_name in filtered[condition_group]:
        combinations[condition_group][method_name] = {}
        first_comp = comparisons_to_add[0]
        filter_expressions = ana.deg_filter_expressions(method_name)
        for filter_expr in filter_expressions:
            suffix = ana.deg_filter_expression_as_str(filter_expr)
            combinations[condition_group][method_name][suffix] = {}
            new_name = "_".join([new_name_prefix, suffix])
            genes_to_combine = [filtered[condition_group][method_name][comp_name][suffix] for comp_name in comparisons_to_add]
            combined_genes = generator(new_name, genes_to_combine)
            combinations[condition_group][method_name][suffix][new_name_prefix] = combined_genes
            norm_sample_columns = []
            for g in genes_to_combine:
                norm_sample_columns.extend(defaults.genes_parameters[g.name]["expression_columns"])
            defaults.register_genes(
                combined_genes,
                norm_sample_columns,
                condition_group,
                class_labels_by_group[condition_group],
                section_index,
                [
                    combined_genes.add_annotator(anno) for anno in list(normalized.values())
                ],
            )
            combined_genes.write()

nb = defaults.run_standards(nb)

for downstream in ana.downstream:
    if downstream == "pathway_analysis":
        for pathway_method in ana.downstream[downstream]:
            if pathway_method == "ora":
                nb = defaults.run_ora(nb)
            if pathway_method == "gsea":
                g = GSEA()
                collections = ["h"]
                if "collections" in ana.downstream[downstream][pathway_method]:
                    collections = ana.downstream[downstream][pathway_method]["collections"]
                parameter = {"permutations": 1000}
                if "parameter" in ana.downstream[downstream][pathway_method]:
                    parameter = ana.downstream[downstream][pathway_method]["parameter"]
                # run gsea
                for condition_group in conditions:
                    for cond1, cond2, _ in comparisons_to_do[condition_group]:
                        columns_a_b = (
                            [x.columns[0] for x in norm_groups[condition_group][cond1]],
                            [x.columns[0] for x in norm_groups[condition_group][cond2]]
                            )
                        for collection in collections:  # collections_ipa + collections_msig:
                            job, index_html = g.run_on_counts(
                                genes_used,
                                phenotypes=[cond1, cond2],
                                columns_a_b=columns_a_b,
                                collection=collection,
                                genome=genome,
                                annotators=normalized.values(),
                                **parameter,
                            )
                            nb.register_html(index_html, job, section_index=section_indices[condition_group, comp_name])
                        if len(collections) > 1:
                            job, index_html = g.run_on_counts(
                                genes_used,
                                phenotypes=[cond1, cond2],
                                columns_a_b=columns_a_b,
                                collection=collections,
                                genome=genome,
                                annotators=normalized.values(),
                                **parameter,
                            )
                            nb.register_html(index_html, job, section_index=section_indices[condition_group, comp_name])
    else:
        raise NotImplementedError


genes_used.write()
genes.write()
nb.write()
nb.convert()
regulated_genes = []
for condition_group in filtered:
    for method_name in filtered[condition_group]:
        for comp_name in filtered[condition_group][method_name]:
            regulated_genes.extend(list(filtered[condition_group][method_name][comp_name].values()))

mbf_publish.scb.prep_scb(
    aligned_lanes,
    regulated_genes
    )
ppg.run_pipegraph()
