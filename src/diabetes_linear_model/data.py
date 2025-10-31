"""Data loading utilities for the azdiabetes dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class TabularData:
    columns: List[str]
    rows: List[List[str]]


def _split_row(row: str) -> List[str]:
    return row.strip().split()


def _manual_parse(lines: Iterable[str]) -> TabularData:
    iterator = iter(lines)
    header = next(iterator, None)
    if header is None:
        raise ValueError("The provided file is empty.")
    columns = _split_row(header)
    rows = [_split_row(line) for line in iterator if line.strip()]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Inconsistent number of columns in the data file.")
    return TabularData(columns=columns, rows=rows)


def load_az_diabetes(path: str | Path, categorical: Sequence[str] = ("diabetes",)) -> TabularData:
    """Load the azdiabetes dataset from a whitespace separated file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        data = _manual_parse(handle.readlines())

    # Coerce numeric columns where appropriate.
    cat_set = set(categorical)
    for row in data.rows:
        for idx, column in enumerate(data.columns):
            if column in cat_set:
                continue
            row[idx] = float(row[idx])
    return data


def prepare_regression_matrices(
    data: TabularData,
    response: str,
    exclude: Sequence[str] | None = None,
    add_intercept: bool = True,
) -> tuple[List[str], str, List[List[float]], List[float]]:
    """Construct a design matrix and response vector from tabular data."""

    if response not in data.columns:
        raise KeyError(f"Response column '{response}' is not present in the data.")

    exclude = set(exclude or []) | {response}

    predictor_columns = [col for col in data.columns if col not in exclude]
    if add_intercept:
        predictors = ["intercept"] + predictor_columns
    else:
        predictors = predictor_columns.copy()

    response_index = data.columns.index(response)
    predictor_indices = [data.columns.index(col) for col in predictor_columns]

    X: List[List[float]] = []
    y: List[float] = []
    for row in data.rows:
        y.append(float(row[response_index]))
        features = [float(row[idx]) for idx in predictor_indices]
        if add_intercept:
            X.append([1.0] + features)
        else:
            X.append(features)

    return predictors, response, X, y
