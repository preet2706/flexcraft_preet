"""This module provides classes for handling common input and output CSV files in protein design scripts."""

import os

class ScoreCSV:
    """Manages an output CSV file with protein design scores and metrics.

    Args:
        path: path to the output CSV file.
        keys: list of column names in the CSV file.
        default: optional string specifying the CSV entry for missing entries in a row.
        append: if True and `path` already exists, keep it as-is (existing rows preserved)
            instead of truncating -- used to resume an interrupted run. The existing header
            must match `keys` exactly, or this raises rather than silently writing rows under
            the wrong columns. Default False preserves the original always-truncate behavior.
    """
    def __init__(self, path, keys, default="none", append=False):
        self.path = path
        self.keys = keys
        self.default = default
        if append and os.path.isfile(path):
            with open(path, "rt") as f:
                existing_header = f.readline().rstrip("\n").split(",")
            if existing_header != list(keys):
                raise ValueError(
                    f"ScoreCSV(append=True): '{path}' already has columns {existing_header}, "
                    f"but this run configured {list(keys)}. Refusing to append with a "
                    f"mismatched schema. Rerun with the same options as the run that created "
                    f"this file, or move it aside to start fresh.")
            return
        with open(path, "wt") as f:
            f.write(",".join(self.keys) + "\n")

    def write_line(self, data: dict):
        """Write a row to the output CSV file given a data dictionary.
        Searches the `data` dictionary for keys in `self.keys` and writes the values
        to the corresponding columns in the CSV file, handling missing keys by writing
        `self.default`.

        Args:
            data: dictionary with one or more keys which are also in `self.keys`.
        """
        with open(self.path, "at") as f:
            result = []
            for key in self.keys:
                if key in data:
                    result.append(str(data[key]))
                else:
                    result.append(self.default)
            f.write(",".join(result) + "\n")
