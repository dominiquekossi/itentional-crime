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


def filter_counts_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where unit_of_measurement is 'Counts'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'unit_of_measurement' column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only count-based rows.
    """
    return df[df["unit_of_measurement"] == "Counts"].copy()


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


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Counts duplicates before removal, drops them, and logs how many
    were removed and how many rows remain.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing duplicate rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    n_duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    n_remaining = len(df)
    logger.info(
        "Removed %d duplicate rows. %d rows remaining.",
        n_duplicates,
        n_remaining,
    )
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns to the DataFrame.

    Adds:
    - 'decade': the decade of the year (e.g., 2010 for year 2015)
    - 'period': categorical period label (1990s, 2000s, 2010s, 2020s)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'year' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added feature columns.
    """
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    df["period"] = pd.cut(
        df["year"],
        bins=[1989, 1999, 2009, 2019, 2030],
        labels=["1990s", "2000s", "2010s", "2020s"],
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


def export_datasets(
    df: pd.DataFrame, output_dir: str, df_counts: pd.DataFrame | None = None
) -> None:
    """Export cleaned and regression-ready datasets to CSV files.

    Exports the full cleaned rate DataFrame to 'dataset_limpo.csv', a
    regression-ready subset (Total sex only, columns: country, year, value)
    to 'dataset_regressao.csv', and optionally the counts DataFrame to
    'dataset_counts.csv'.

    Parameters
    ----------
    df : pd.DataFrame
        The full cleaned rate DataFrame.
    output_dir : str
        Directory path where CSV files will be saved.
    df_counts : pd.DataFrame, optional
        The cleaned counts DataFrame. If provided, exported to
        'dataset_counts.csv'.

    Returns
    -------
    None
    """
    ensure_directory(output_dir)

    # Export full cleaned dataset (rate)
    full_path = f"{output_dir}/dataset_limpo.csv"
    df.to_csv(full_path, index=False, encoding="utf-8")

    # Export regression-ready dataset (Total sex only, aggregate level, key columns)
    regression_df = df[
        (df["sex"] == "Total")
        & (df["dimension"] == "Total")
        & (df["category"] == "Total")
        & (df["age"] == "Total")
    ][["country", "year", "value"]].copy()
    regression_path = f"{output_dir}/dataset_regressao.csv"
    regression_df.to_csv(regression_path, index=False, encoding="utf-8")

    # Export counts dataset if provided
    if df_counts is not None:
        counts_path = f"{output_dir}/dataset_counts.csv"
        df_counts.to_csv(counts_path, index=False, encoding="utf-8")


def run_cleaning_pipeline(input_path: str, output_dir: str) -> pd.DataFrame:
    """Execute the full data cleaning pipeline.

    Chains all cleaning steps in sequence. After standardizing columns and
    filtering by indicator, branches into two paths:
    - Rate path: filter_rate_unit -> remove_global_aggregations -> remove_nulls
      -> convert_types -> remove_duplicates -> add_feature_engineering
    - Counts path: filter_counts_unit -> remove_global_aggregations -> remove_nulls
      -> convert_types -> remove_duplicates -> add_feature_engineering

    Both are exported. The rate DataFrame is returned for backward compatibility.

    Parameters
    ----------
    input_path : str
        Path to the raw Excel dataset file.
    output_dir : str
        Directory path where cleaned CSV files will be exported.

    Returns
    -------
    pd.DataFrame
        The fully cleaned rate DataFrame.
    """
    df = load_raw_data(input_path)
    df = standardize_columns(df)
    df = filter_indicator(df)

    # Rate path
    df_rate = filter_rate_unit(df)
    df_rate = remove_global_aggregations(df_rate)
    df_rate = remove_nulls(df_rate)
    df_rate = convert_types(df_rate)
    df_rate = remove_duplicates(df_rate)
    df_rate = add_feature_engineering(df_rate)

    # Counts path
    df_counts = filter_counts_unit(df)
    df_counts = remove_global_aggregations(df_counts)
    df_counts = remove_nulls(df_counts)
    df_counts = convert_types(df_counts)
    df_counts = remove_duplicates(df_counts)
    df_counts = add_feature_engineering(df_counts)

    export_datasets(df_rate, output_dir, df_counts=df_counts)
    return df_rate
