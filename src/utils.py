"""Shared utility functions used across the analysis pipeline modules."""

import os
import re

import pandas as pd
import plotly.graph_objects as go


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        The directory path to create.

    Returns
    -------
    None
    """
    os.makedirs(path, exist_ok=True)


def load_cleaned_data(path: str) -> pd.DataFrame:
    """Load a cleaned CSV file into a Pandas DataFrame.

    Parameters
    ----------
    path : str
        The file path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def snake_case(name: str) -> str:
    """Convert a string to snake_case format.

    Converts to lowercase, replaces spaces and special characters with
    underscores, and strips leading/trailing underscores.

    Parameters
    ----------
    name : str
        The string to convert.

    Returns
    -------
    str
        The converted snake_case string.
    """
    result = name.lower()
    result = re.sub(r"[^a-z0-9]+", "_", result)
    result = result.strip("_")
    return result


def save_plotly_chart(fig: go.Figure, filepath: str) -> None:
    """Save a Plotly figure as a static PNG image.

    Creates the parent directory if it does not exist before saving.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to save.
    filepath : str
        The destination file path for the PNG image.

    Returns
    -------
    None
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        ensure_directory(parent_dir)
    fig.write_image(filepath, format="png")
