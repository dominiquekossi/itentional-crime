"""Data cleaning module for the global homicide analysis pipeline.

This module handles loading raw Excel data, applying cleaning transformations,
and exporting processed datasets for downstream analysis.
"""

import logging

import pandas as pd

from src.utils import ensure_directory, snake_case

logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw Excel dataset into a Pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the Excel file to load.

    Returns
    -------
    pd.DataFrame
        The raw data loaded from the Excel file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    try:
        return pd.read_excel(filepath, engine="openpyxl")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file not found: '{filepath}'. "
            "Please ensure the Excel file exists at the specified path."
        )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to snake_case format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with snake_case column names.
    """
    df = df.copy()
    df.columns = [snake_case(col) for col in df.columns]
    return df


def filter_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where indicator is 'Victims of intentional homicide'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'indicator' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only victim indicator rows.
    """
    return df[df["indicator"] == "Victims of intentional homicide"].copy()


def filter_rate_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where unit_of_measurement is 'Rate per 100,000 population'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'unit_of_measurement' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rate-based rows.
    """
    return df[df["unit_of_measurement"] == "Rate per 100,000 population"].copy()


def remove_global_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where country contains global aggregation terms.

    Removes rows where the country column contains 'World', 'Global',
    'Various', or 'Unknown' (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'country' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with global aggregation rows removed.
    """
    pattern = r"(?i)(?:World|Global|Various|Unknown)"
    mask = df["country"].str.contains(pattern, na=False)
    return df[~mask].copy()


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null values in country, year, or value columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'country', 'year', and 'value' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with null rows removed from key columns.
    """
    return df.dropna(subset=["country", "year", "value"]).copy()


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert year to int and value to float, dropping unconvertible rows.

    Rows that cannot be converted are dropped and a warning is logged
    with the affected row indices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'year' and 'value' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected column types.
    """
    df = df.copy()

    # Convert value to float
    original_len = len(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    failed_value = df["value"].isna() & df.index.isin(df.index)
    if failed_value.any():
        failed_indices = df[failed_value].index.tolist()
        logger.warning(
            "Dropped %d rows due to non-numeric 'value' at indices: %s",
            len(failed_indices),
            failed_indices,
        )
        df = df.dropna(subset=["value"])

    # Convert year to int
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    failed_year = df["year"].isna()
    if failed_year.any():
        failed_indices = df[failed_year].index.tolist()
        logger.warning(
            "Dropped %d rows due to non-numeric 'year' at indices: %s",
            len(failed_indices),
            failed_indices,
        )
        df = df.dropna(subset=["year"])

    df["year"] = df["year"].astype(int)
    df["value"] = df["value"].astype(float)

    if len(df) < original_len:
        logger.warning(
            "Type conversion reduced DataFrame from %d to %d rows.",
            original_len,
            len(df),
        )

    return df


def segment_by_sex(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group the DataFrame by sex column into separate DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'sex' column containing 'Total', 'Female', 'Male'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys 'Total', 'Female', 'Male' mapping to
        their respective filtered DataFrames.
    """
    segments = {}
    for key in ("Total", "Female", "Male"):
        segments[key] = df[df["sex"] == key].copy()
    return segments


def export_datasets(df: pd.DataFrame, output_dir: str) -> None:
    """Export cleaned and regression-ready datasets to CSV files.

    Exports the full cleaned DataFrame to 'dataset_limpo.csv' and a
    regression-ready subset (Total sex only, columns: country, year, value)
    to 'dataset_regressao.csv'.

    Parameters
    ----------
    df : pd.DataFrame
        The full cleaned DataFrame.
    output_dir : str
        Directory path where CSV files will be saved.

    Returns
    -------
    None
    """
    ensure_directory(output_dir)

    # Export full cleaned dataset
    full_path = f"{output_dir}/dataset_limpo.csv"
    df.to_csv(full_path, index=False, encoding="utf-8")

    # Export regression-ready dataset (Total sex only, key columns)
    regression_df = df[df["sex"] == "Total"][["country", "year", "value"]].copy()
    regression_path = f"{output_dir}/dataset_regressao.csv"
    regression_df.to_csv(regression_path, index=False, encoding="utf-8")


def run_cleaning_pipeline(input_path: str, output_dir: str) -> pd.DataFrame:
    """Execute the full data cleaning pipeline.

    Chains all cleaning steps in sequence: load, standardize columns,
    filter indicator, filter rate unit, remove global aggregations,
    remove nulls, convert types, and export results.

    Parameters
    ----------
    input_path : str
        Path to the raw Excel dataset file.
    output_dir : str
        Directory path where cleaned CSV files will be exported.

    Returns
    -------
    pd.DataFrame
        The fully cleaned DataFrame.
    """
    df = load_raw_data(input_path)
    df = standardize_columns(df)
    df = filter_indicator(df)
    df = filter_rate_unit(df)
    df = remove_global_aggregations(df)
    df = remove_nulls(df)
    df = convert_types(df)
    export_datasets(df, output_dir)
    return df
