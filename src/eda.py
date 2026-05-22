"""Exploratory Data Analysis module for the global homicide analysis pipeline.

This module generates interactive Plotly charts and computes statistical
answers to the 10 mandatory analysis questions.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import save_plotly_chart


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _filter_aggregate(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    """Filter DataFrame to aggregate-level rows for a given sex.

    Keeps only rows where dimension == 'Total', category == 'Total',
    age == 'Total', and sex matches the given value. This ensures one
    row per country/year.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.
    sex : str
        The sex filter value ('Total', 'Female', or 'Male').

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with aggregate rows only.
    """
    mask = (
        (df["sex"] == sex)
        & (df["dimension"] == "Total")
        & (df["category"] == "Total")
        & (df["age"] == "Total")
    )
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Chart Functions
# ---------------------------------------------------------------------------


def plot_histogram(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Generate a histogram chart for the specified column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The column name to plot the distribution for.
    title : str
        The chart title.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the histogram.
    """
    fig = px.histogram(df, x=column, title=title, template="plotly_white")
    return fig


def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Generate a boxplot chart grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        The column name for the categorical axis.
    y : str
        The column name for the numerical axis.
    title : str
        The chart title.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the boxplot.
    """
    fig = px.box(df, x=x, y=y, title=title, template="plotly_white")
    return fig


def plot_line_chart(
    df: pd.DataFrame, x: str, y: str, color: str, title: str
) -> go.Figure:
    """Generate a line chart with color grouping for time series data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        The column name for the x-axis (typically year).
    y : str
        The column name for the y-axis (typically value/rate).
    color : str
        The column name used to color-group lines.
    title : str
        The chart title.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the line chart.
    """
    fig = px.line(df, x=x, y=y, color=color, title=title, template="plotly_white")
    return fig


def plot_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Generate a bar chart for ranked data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        The column name for the x-axis (typically country or category).
    y : str
        The column name for the y-axis (typically value/rate).
    title : str
        The chart title.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the bar chart.
    """
    fig = px.bar(df, x=x, y=y, title=title, template="plotly_white")
    return fig


def plot_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    """Generate a heatmap of the correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric columns.
    title : str
        The chart title.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the heatmap.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig


def plot_continent_comparison_seaborn(df: pd.DataFrame, output_dir: str) -> None:
    """Generate continent comparison charts using seaborn and matplotlib.

    Creates two charts:
    1. Bar plot of average homicide rate by continent
    2. Temporal evolution of homicide rate by continent

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.
    output_dir : str
        Directory path where PNG chart files will be saved.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_agg = _filter_aggregate(df, "Total")

    # Chart 1: Barplot by continent
    fig, ax = plt.subplots(figsize=(10, 6))
    continent_avg = (
        df_agg.groupby("region")["value"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    sns.barplot(data=continent_avg, x="region", y="value", ax=ax, palette="Reds_r")
    ax.set_title("Taxa Media de Homicidio por Continente")
    ax.set_xlabel("Continente")
    ax.set_ylabel("Taxa Media (por 100k)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seaborn_continent_bar.png"), dpi=150)
    plt.close()

    # Chart 2: Temporal evolution by continent
    fig, ax = plt.subplots(figsize=(12, 6))
    continent_year = (
        df_agg.groupby(["year", "region"])["value"].mean().reset_index()
    )
    sns.lineplot(
        data=continent_year, x="year", y="value", hue="region", ax=ax, palette="Set2"
    )
    ax.set_title("Evolucao Temporal da Taxa de Homicidio por Continente")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Taxa Media (por 100k)")
    ax.legend(title="Continente")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seaborn_continent_temporal.png"), dpi=150)
    plt.close()


def generate_all_charts(df: pd.DataFrame, output_dir: str) -> None:
    """Generate all EDA chart types and save them as PNG files.

    Creates representative charts: histogram of homicide rates, boxplot
    by region, line chart of trends for top countries, bar chart of top
    countries by average rate, a correlation heatmap, and seaborn
    continent comparison charts.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame with homicide data.
    output_dir : str
        Directory path where PNG chart files will be saved.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Histogram of homicide rate values
    fig_hist = plot_histogram(df, "value", "Distribuição das Taxas de Homicídio")
    save_plotly_chart(fig_hist, os.path.join(output_dir, "histogram_rates.png"))

    # Boxplot by region (aggregate level)
    df_agg = _filter_aggregate(df, "Total")
    fig_box = plot_boxplot(df_agg, "region", "value", "Taxas de Homicídio por Região")
    save_plotly_chart(fig_box, os.path.join(output_dir, "boxplot_region.png"))

    # Line chart — trends for top 5 countries by average rate
    top_countries = (
        df_agg.groupby("country")["value"]
        .mean()
        .nlargest(5)
        .index.tolist()
    )
    df_top = df_agg[df_agg["country"].isin(top_countries)]
    fig_line = plot_line_chart(
        df_top, "year", "value", "country",
        "Tendência de Homicídios — Top 5 Países"
    )
    save_plotly_chart(fig_line, os.path.join(output_dir, "line_trends.png"))

    # Bar chart — top 10 countries by average rate
    top10 = (
        df_agg.groupby("country")["value"]
        .mean()
        .nlargest(10)
        .reset_index()
    )
    top10.columns = ["country", "avg_rate"]
    fig_bar = plot_bar_chart(
        top10, "country", "avg_rate",
        "Top 10 Países por Taxa Média de Homicídio"
    )
    save_plotly_chart(fig_bar, os.path.join(output_dir, "bar_top10.png"))

    # Heatmap of correlations
    fig_heat = plot_heatmap(df, "Matriz de Correlação — Variáveis Numéricas")
    save_plotly_chart(fig_heat, os.path.join(output_dir, "heatmap_correlation.png"))

    # Seaborn continent comparison charts
    plot_continent_comparison_seaborn(df, output_dir)


# ---------------------------------------------------------------------------
# Statistical Question Functions
# ---------------------------------------------------------------------------


def answer_question_1(df: pd.DataFrame) -> pd.DataFrame:
    """Top 10 countries by average homicide rate in the last 5 available years.

    Filters for sex == 'Total' (aggregate level), identifies the last 5
    years in the dataset, then computes the mean rate per country and
    returns the top 10.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['country', 'avg_rate'] sorted descending.
    """
    df_total = _filter_aggregate(df, "Total")
    last_5_years = sorted(df_total["year"].unique())[-5:]
    df_filtered = df_total[df_total["year"].isin(last_5_years)]
    result = (
        df_filtered.groupby("country")["value"]
        .mean()
        .nlargest(10)
        .reset_index()
    )
    result.columns = ["country", "avg_rate"]
    return result


def answer_question_2(df: pd.DataFrame) -> pd.DataFrame:
    """Top 10 countries by female homicide rate in 2022.

    Filters for sex == 'Female' (aggregate level) and year == 2022,
    then returns the top 10 countries by homicide rate.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['country', 'value'] sorted descending.
    """
    df_female = _filter_aggregate(df, "Female")
    df_female_2022 = df_female[df_female["year"] == 2022]
    result = (
        df_female_2022.nlargest(10, "value")[["country", "value"]]
        .reset_index(drop=True)
    )
    return result


def answer_question_3(
    df: pd.DataFrame, df_counts: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Regions with highest total homicide counts.

    If df_counts is provided, uses it for summing totals (actual counts).
    Otherwise, falls back to summing rates from the rate DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame (rate).
    df_counts : pd.DataFrame, optional
        The cleaned counts DataFrame. If provided, used for summing totals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['region', 'total_value'] sorted descending.
    """
    source = df_counts if df_counts is not None else df
    df_total = _filter_aggregate(source, "Total")
    result = (
        df_total.groupby("region")["value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = ["region", "total_value"]
    return result


def answer_question_4(df: pd.DataFrame) -> pd.DataFrame:
    """Countries with lowest homicide rate per sub-region.

    For each sub-region, finds the country with the minimum average
    homicide rate (sex == 'Total', aggregate level).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['subregion', 'country', 'avg_rate'].
    """
    df_total = _filter_aggregate(df, "Total")
    avg_rates = (
        df_total.groupby(["subregion", "country"])["value"]
        .mean()
        .reset_index()
    )
    avg_rates.columns = ["subregion", "country", "avg_rate"]
    result = avg_rates.loc[avg_rates.groupby("subregion")["avg_rate"].idxmin()]
    return result.reset_index(drop=True)


def answer_question_5(df: pd.DataFrame) -> pd.DataFrame:
    """Countries with lowest female homicide rates (bottom 10).

    Filters for sex == 'Female' (aggregate level), computes average rate
    per country, and returns the 10 countries with the lowest rates.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['country', 'avg_rate'] sorted ascending.
    """
    df_female = _filter_aggregate(df, "Female")
    result = (
        df_female.groupby("country")["value"]
        .mean()
        .nsmallest(10)
        .reset_index()
    )
    result.columns = ["country", "avg_rate"]
    return result


def answer_question_6(
    df: pd.DataFrame, df_counts: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Sub-regions with highest total homicide counts.

    If df_counts is provided, uses it for summing totals (actual counts).
    Otherwise, falls back to summing rates from the rate DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame (rate).
    df_counts : pd.DataFrame, optional
        The cleaned counts DataFrame. If provided, used for summing totals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['subregion', 'total_value'] sorted descending.
    """
    source = df_counts if df_counts is not None else df
    df_total = _filter_aggregate(source, "Total")
    result = (
        df_total.groupby("subregion")["value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = ["subregion", "total_value"]
    return result


def answer_question_7(df: pd.DataFrame) -> pd.DataFrame:
    """Country with highest homicide rate per continent in 2020.

    Filters for sex == 'Total' (aggregate level) and year == 2020,
    groups by region (continent), and finds the country with the maximum
    rate in each.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['region', 'country', 'value'].
    """
    df_total = _filter_aggregate(df, "Total")
    df_2020 = df_total[df_total["year"] == 2020]
    result = df_2020.loc[df_2020.groupby("region")["value"].idxmax()]
    return result[["region", "country", "value"]].reset_index(drop=True)


def answer_question_8(df: pd.DataFrame) -> pd.DataFrame:
    """Most violent country for women in 2021.

    Filters for sex == 'Female' (aggregate level) and year == 2021,
    returns the country with the highest homicide rate.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['country', 'value'] for the top country.
    """
    df_female = _filter_aggregate(df, "Female")
    df_female_2021 = df_female[df_female["year"] == 2021]
    result = df_female_2021.nlargest(1, "value")[["country", "value"]]
    return result.reset_index(drop=True)


def answer_question_9(df: pd.DataFrame) -> pd.DataFrame:
    """Country with highest single indicator value across all years.

    Finds the row with the maximum value in the entire dataset and
    returns the country, year, and value.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['country', 'year', 'value'] for the max row.
    """
    idx_max = df["value"].idxmax()
    result = df.loc[[idx_max], ["country", "year", "value"]]
    return result.reset_index(drop=True)


def answer_question_10(df: pd.DataFrame) -> float:
    """Brazil average homicide rate in the last 10 available years.

    Filters for country == 'Brazil' and sex == 'Total' (aggregate level),
    identifies the last 10 available years, and computes the mean rate.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame.

    Returns
    -------
    float
        The average homicide rate for Brazil over the last 10 years.
    """
    df_total = _filter_aggregate(df, "Total")
    df_brazil = df_total[df_total["country"] == "Brazil"]
    last_10_years = sorted(df_brazil["year"].unique())[-10:]
    df_filtered = df_brazil[df_brazil["year"].isin(last_10_years)]
    return float(df_filtered["value"].mean())


def answer_all_questions(
    df: pd.DataFrame, df_counts: pd.DataFrame | None = None
) -> dict:
    """Execute all 10 statistical question functions and return results.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned homicide DataFrame (rate).
    df_counts : pd.DataFrame, optional
        The cleaned counts DataFrame. If provided, passed to questions
        3 and 6 for accurate total counts.

    Returns
    -------
    dict
        Dictionary with keys 'q1' through 'q10' mapping to each
        question's result.
    """
    return {
        "q1": answer_question_1(df),
        "q2": answer_question_2(df),
        "q3": answer_question_3(df, df_counts=df_counts),
        "q4": answer_question_4(df),
        "q5": answer_question_5(df),
        "q6": answer_question_6(df, df_counts=df_counts),
        "q7": answer_question_7(df),
        "q8": answer_question_8(df),
        "q9": answer_question_9(df),
        "q10": answer_question_10(df),
    }
