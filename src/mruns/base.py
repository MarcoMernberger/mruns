#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""base.py: Contains toml parser and basic data class."""

from mruns import __version__
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Tuple, Any
import pandas as pd
import pypipegraph as ppg
import tomlkit
import logging
import sys


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


_logger = logging.getLogger(__name__)


@dataclass
class Analysis:
    name: str
    anaylsis_type: str
    sample_file: Path
    reverse: bool
    kit: str
    species: str
    revision: int
    aligner: str
    canonical_chromosomes: bool
    biotypes: List["str"]
    cpm_threshold: float
    comparison_method: str
    filter_expressions: List[List["str"]]
    directions: List[str]
    downstream: Dict[str, Any]


def get_analysis_from_toml(req_file: Path = Path("run.toml")):
    ap = read_toml(req_file)
    for key in ap:
        print(key, ap[key], type(ap[key]))
    # project general
    name = ap["project"]["name"]
    # analysis
    analysis_type = ap["analysis"]["type"]
    # sample specific information
    samples_file = ap["samples"]["df_samples_path"]
    reverse_reads = ap["samples"]["reverse_reads"]
    kit = ap["samples"]["kit"]
    # alignment specific
    species = ap["alignment"]["species"]
    revision = ap["alignment"]["revision"]
    aligner = ap["alignment"]["aligner"]
    # gene specific
    canonical = ap["genes"]["filter"]["canonical"]
    biotypes = ap["genes"]["filter"]["biotypes"]
    cpm_threshold = ap["genes"]["filter"]["cpm_threshold"]
    # comparisons
    comparison_method = ap["comparison"]["method"]
    filter_expression = ap["comparison"]["filter_expression"]
    directions = ap["comparison"]["directions"]
    # downstream analysis
    further_analyses = ap["downstream"].keys()
    specs: Dict[str, Any] = {}
    for downstream_type in further_analyses:
        specs[downstream_type] = {}
        for method in ap["downstream"][downstream_type]:
            specs[downstream_type][method] = ap["downstream"][downstream_type][method]

    # finalize
    report_name = ap["reports"]["name"]
    analysis = Analysis(
        name,
        analysis_type,
        samples_file,
        reverse_reads,
        kit,
        species,
        revision,
        aligner,
        canonical,
        biotypes,
        cpm_threshold,
        comparison_method,
        filter_expression,
        directions,
        specs,
        report_name,
    )
    return analysis


def read_toml(req_file: Path):
    with req_file.open("r") as op:
        p = tomlkit.loads(op.read())
    return p


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


# def main(args):
    # """Main entry point allowing external calls
# 
    # Args:
    #   args ([str]): command line parameter list
    # """
    # args = parse_args(args)
    # setup_logging(args.loglevel)
    # _logger.debug("Starting crazy calculations...")
    # print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    # _logger.info("Script ends here")
# 
# 
# def run():
    # """Entry point for console_scripts
    # """
    # main(sys.argv[1:])
# 
# 
# if __name__ == "__main__":
    # run()
# 