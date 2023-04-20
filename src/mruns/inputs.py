import pandas as pd
from pathlib import Path
from typing import Union, Callable, Any
from pandas import DataFrame


class InputHandler:
    def __init__(self):
        self.name = "InputHandler"

    @classmethod
    def read_tsv(self, input_path: Path, **kwargs) -> DataFrame:
        return pd.read_csv(input_path, sep="\t", **kwargs)

    @classmethod
    def read_excel(self, input_path: Path, **kwargs) -> DataFrame:
        return pd.read_excel(input_path, **kwargs)

    @classmethod
    def read(self, path: Path, **kwargs):
        if path.suffix == ".tsv":
            reader_function = self.read_tsv
        elif path.suffix == ".xlsx":
            reader_function = self.read_exel
        else:
            raise NotImplementedError(f"Need a method to load file type {path.suffix}")

        def __read():
            return reader_function(path, **kwargs)

        return __read

    @classmethod
    def get_load_callable(self, source: Any) -> Any:
        if isinstance(source, str) or isinstance(source, Path):
            return self.read(Path(source))
        elif callable(source):
            return source
        else:
            raise NotImplementedError(f"Don't know how to read from type {type(source)}.")
