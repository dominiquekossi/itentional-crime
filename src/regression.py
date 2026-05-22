"""Regression module for training models, evaluating metrics, and generating predictions.

This module provides functions to train Linear Regression and Random Forest
models on historical homicide rate data, evaluate their performance, generate
future predictions, and produce visualization charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils import save_plotly_chart


def prepare_regression_data(
    df: pd.DataFrame, country: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter data by country and prepare train/test splits for regression.

    Filters the DataFrame for the specified country, extracts year as the
    feature (reshaped to 2D) and value as the target, then performs an
    80/20 train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'country', 'year', and 'value'.
    country : str
        The country name to filter by.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple of (X_train, X_test, y_train, y_test).
    """
    df_country = df[df["country"] == country].copy()
    X = df_country["year"].values.reshape(-1, 1)
    y = df_country["value"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train and return a Linear Regression model.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature array (year values reshaped to 2D).
    y_train : np.ndarray
        Training target array (homicide rate values).

    Returns
    -------
    LinearRegression
        The fitted Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Train and return a Random Forest Regressor model.

    Uses 100 estimators and random_state=42 for reproducibility.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature array (year values reshaped to 2D).
    y_train : np.ndarray
        Training target array (homicide rate values).

    Returns
    -------
    RandomForestRegressor
        The fitted Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate a trained model and return performance metrics.

    Computes Mean Absolute Error (MAE), Mean Squared Error (MSE),
    Root Mean Squared Error (RMSE), and R² score.

    Parameters
    ----------
    model : object
        A fitted sklearn model with a predict method.
    X_test : np.ndarray
        Test feature array.
    y_test : np.ndarray
        Test target array.

    Returns
    -------
    dict[str, float]
        Dictionary with keys "MAE", "MSE", "RMSE", "R2".
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def predict_future(
    model, years: list[int] | None = None
) -> pd.DataFrame:
    """Generate predictions for future years.

    Parameters
    ----------
    model : object
        A fitted sklearn model with a predict method.
    years : list[int], optional
        List of years to predict. Defaults to [2023, 2024, 2025, 2026].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['year', 'predicted_rate'].
    """
    if years is None:
        years = [2023, 2024, 2025, 2026]

    X_future = np.array(years).reshape(-1, 1)
    predictions = model.predict(X_future)

    return pd.DataFrame({"year": years, "predicted_rate": predictions})


def plot_historical_and_predictions(
    df_hist: pd.DataFrame, df_pred: pd.DataFrame, title: str
) -> go.Figure:
    """Create a chart showing historical data as a line and predictions as markers.

    Parameters
    ----------
    df_hist : pd.DataFrame
        Historical data with columns 'year' and 'value'.
    df_pred : pd.DataFrame
        Prediction data with columns 'year' and 'predicted_rate'.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with historical line and prediction markers.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_hist["year"],
            y=df_hist["value"],
            mode="lines",
            name="Historical",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_pred["year"],
            y=df_pred["predicted_rate"],
            mode="markers",
            name="Predictions",
            marker=dict(color="red", size=10),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Homicide Rate",
        template="plotly_white",
    )

    return fig


def plot_real_vs_predicted(
    y_real: np.ndarray, y_pred: np.ndarray, title: str
) -> go.Figure:
    """Create a scatter plot comparing real vs predicted values with a diagonal reference.

    Parameters
    ----------
    y_real : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted target values.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with scatter points and diagonal reference line.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_real,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(color="blue", size=8),
        )
    )

    # Diagonal reference line
    min_val = min(float(np.min(y_real)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_real)), float(np.max(y_pred)))

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Real Values",
        yaxis_title="Predicted Values",
        template="plotly_white",
    )

    return fig


def compare_models(df: pd.DataFrame, country: str) -> dict:
    """Train both models, evaluate both, and return a comparison dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'country', 'year', and 'value'.
    country : str
        The country name to filter by.

    Returns
    -------
    dict
        Dictionary with keys "linear_regression" and "random_forest",
        each containing the metrics dict with MAE, MSE, RMSE, R2.
    """
    X_train, X_test, y_train, y_test = prepare_regression_data(df, country)

    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    return {"linear_regression": lr_metrics, "random_forest": rf_metrics}


def run_regression_pipeline(
    df: pd.DataFrame, country: str, output_dir: str
) -> dict:
    """Run the full regression workflow for a given country.

    Prepares data, trains both models, evaluates performance, generates
    future predictions (2023-2026), creates visualization charts, and
    saves charts to the output directory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'country', 'year', and 'value'.
    country : str
        The country name to filter by.
    output_dir : str
        Directory path where charts will be saved.

    Returns
    -------
    dict
        Results dictionary containing:
        - "country": the country name
        - "linear_regression": metrics dict
        - "random_forest": metrics dict
        - "predictions_lr": DataFrame with LR predictions
        - "predictions_rf": DataFrame with RF predictions
    """
    import os

    X_train, X_test, y_train, y_test = prepare_regression_data(df, country)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # Generate future predictions
    future_years = [2023, 2024, 2025, 2026]
    predictions_lr = predict_future(lr_model, future_years)
    predictions_rf = predict_future(rf_model, future_years)

    # Prepare historical data for charts
    df_hist = df[df["country"] == country][["year", "value"]].copy()

    # Generate charts
    fig_hist_lr = plot_historical_and_predictions(
        df_hist, predictions_lr, f"{country} - Linear Regression: Historical vs Predictions"
    )
    fig_hist_rf = plot_historical_and_predictions(
        df_hist, predictions_rf, f"{country} - Random Forest: Historical vs Predictions"
    )

    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    fig_real_lr = plot_real_vs_predicted(
        y_test, y_pred_lr, f"{country} - Linear Regression: Real vs Predicted"
    )
    fig_real_rf = plot_real_vs_predicted(
        y_test, y_pred_rf, f"{country} - Random Forest: Real vs Predicted"
    )

    # Save charts
    save_plotly_chart(fig_hist_lr, os.path.join(output_dir, "regression_lr_historical.png"))
    save_plotly_chart(fig_hist_rf, os.path.join(output_dir, "regression_rf_historical.png"))
    save_plotly_chart(fig_real_lr, os.path.join(output_dir, "regression_lr_real_vs_pred.png"))
    save_plotly_chart(fig_real_rf, os.path.join(output_dir, "regression_rf_real_vs_pred.png"))

    return {
        "country": country,
        "linear_regression": lr_metrics,
        "random_forest": rf_metrics,
        "predictions_lr": predictions_lr,
        "predictions_rf": predictions_rf,
    }
