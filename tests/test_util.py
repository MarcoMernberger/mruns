# -*- coding: utf-8 -*-

import pytest
import mbf
from pathlib import Path
from mruns import util

__author__ = "Marco Mernberger"
__copyright__ = "Marco Mernberger"
__license__ = "mit"


def test_locate_folder():
    runs_incoming = Path("tests/data/incoming")
    run1 = "221130_NB552003_testrun1"
    run2 = "221202_NB552003_testrun2"
    run3 = "221201_NB552003_testrun3"
    runfail = "not_there"
    print(runs_incoming)
    for x in runs_incoming.iterdir():
        print(x)
    path = util.locate_folder(run1, runs_incoming)
    assert isinstance(path, Path)
    path = util.locate_folder(run2, runs_incoming)
    assert isinstance(path, Path)
    path = util.locate_folder(run3, runs_incoming)
    assert isinstance(path, Path)
    with pytest.raises(ValueError):
        util.locate_folder(runfail, runs_incoming)


def test_is_run_id():
    assert util.is_run_id("141112_C00113_0085_AHATRCADXX")
    assert not util.is_run_id("2016")
    assert not util.is_run_id("NextSeq")


def test_read_toml():
    p = util.read_toml(Path("tests/data/run.toml"))
    print(p)
    print(type(p))
    assert isinstance(p, dict)


def test_dir_is_empty(tmp_path):
    assert util.dir_is_empty(tmp_path)
    t = tmp_path / "test"
    t.touch()
    t.write_text("text")
    assert not util.dir_is_empty(tmp_path)


def test_is_fastq_folder():
    assert util.is_fastq_folder(
        Path(
            "tests/data/incoming/NextSeq/221202_NB552003_testrun2/221202_NB552003_testrun2/Alignment_2/20221201/Fastq"
        )
    )


def test_find_fastq_folder():
    path = util.find_fastq_folder(Path("tests/data/incoming/NextSeq/221202_NB552003_testrun2"))
    assert (
        str(path)
        == "tests/data/incoming/NextSeq/221202_NB552003_testrun2/221202_NB552003_testrun2/Alignment_2/20221201/Fastq"
    )
    assert util.find_fastq_folder(Path("tests/data/incoming/2022/221201_NB552003_testrun3"))
    with pytest.raises(ValueError):
        util.find_fastq_folder(
            Path("/talizorah/projects/mruns/tests/data/incoming/221130_NB552003_testrun1")
        )


def test_fill_incoming(tmp_path):
    util.fill_incoming(
        ["221202_NB552003_testrun2", "221201_NB552003_testrun3"],
        Path("tests/data/incoming/"),
        incoming=tmp_path,
    )
    assert (tmp_path / "221202_NB552003_testrun2" / "1_r1_.fastq.gz").exists()
    assert (tmp_path / "221202_NB552003_testrun2" / "1_r2_.fastq.gz").exists()
    assert (tmp_path / "221201_NB552003_testrun3" / "1_r1_.fastq").exists()
    assert (tmp_path / "221201_NB552003_testrun3" / "1_r2_.fastq").exists()
