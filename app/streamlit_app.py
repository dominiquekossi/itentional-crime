"""Streamlit Dashboard for Global Homicide Analysis.

Interactive web application for exploring global intentional homicide data
with filters, charts, regression models, and statistical analysis.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eda import answer_all_questions  # noqa: E402
from src.regression import (  # noqa: E402
    evaluate_model,
    plot_historical_and_predictions,
    plot_real_vs_predicted,
    predict_future,
    prepare_regression_data,
    train_linear_regression,
    train_random_forest,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Análise de Homicídios Globais",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned homicide dataset from outputs directory."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "dataset_limpo.csv",
    )
    if not os.path.exists(path):
        st.error(
            "Arquivo 'outputs/dataset_limpo.csv' não encontrado. "
            "Execute o pipeline de limpeza primeiro."
        )
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_regression_data() -> pd.DataFrame:
    """Load the regression dataset from outputs directory."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "dataset_regressao.csv",
    )
    if not os.path.exists(path):
        st.error(
            "Arquivo 'outputs/dataset_regressao.csv' não encontrado. "
            "Execute o pipeline de limpeza primeiro."
        )
        st.stop()
    return pd.read_csv(path)


# Load data
df = load_data()
df_regression = load_regression_data()

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------

st.sidebar.title("Filtros")

# Country multiselect
countries = sorted(df["country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "País", options=countries, default=[]
)

# Region/Continent multiselect
regions = sorted(df["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Continente/Região", options=regions, default=[]
)

# Sub-region multiselect
subregions = sorted(df["subregion"].dropna().unique().tolist())
selected_subregions = st.sidebar.multiselect(
    "Sub-região", options=subregions, default=[]
)

# Sex radio button
selected_sex = st.sidebar.radio(
    "Sexo", options=["Total", "Female", "Male"], index=0
)

# Year range slider
min_year = int(df["year"].min())
max_year = int(df["year"].max())
selected_years = st.sidebar.slider(
    "Intervalo de Anos",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

# ---------------------------------------------------------------------------
# Filter Application
# ---------------------------------------------------------------------------


def apply_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply all sidebar filter selections to the DataFrame."""
    filtered = dataframe.copy()

    # Filter by sex
    filtered = filtered[filtered["sex"] == selected_sex]

    # Filter by year range
    filtered = filtered[
        (filtered["year"] >= selected_years[0])
        & (filtered["year"] <= selected_years[1])
    ]

    # Filter by country (if any selected)
    if selected_countries:
        filtered = filtered[filtered["country"].isin(selected_countries)]

    # Filter by region (if any selected)
    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    # Filter by sub-region (if any selected)
    if selected_subregions:
        filtered = filtered[filtered["subregion"].isin(selected_subregions)]

    return filtered


df_filtered = apply_filters(df)

# ---------------------------------------------------------------------------
# Main Content — Tabs
# ---------------------------------------------------------------------------

st.title("Analise de Homicidios Globais")

tabs = st.tabs(
    [
        "Visão Geral",
        "Mapa Mundial",
        "Rankings",
        "Violência contra Mulheres",
        "Séries Temporais",
        "Regressão e Predição",
        "Estatísticas Gerais",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Visão Geral (Overview)
# ---------------------------------------------------------------------------

with tabs[0]:
    st.header("Visão Geral")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df_filtered))
    with col2:
        st.metric("Países", df_filtered["country"].nunique())
    with col3:
        if not df_filtered.empty:
            year_range = f"{int(df_filtered['year'].min())} - {int(df_filtered['year'].max())}"
        else:
            year_range = "N/A"
        st.metric("Intervalo de Anos", year_range)

    # Global trend line chart
    st.subheader("Tendência Global ao Longo do Tempo")
    if not df_filtered.empty:
        trend_data = (
            df_filtered.groupby("year")["value"]
            .mean()
            .reset_index()
        )
        trend_data.columns = ["year", "avg_rate"]
        fig_trend = px.line(
            trend_data,
            x="year",
            y="avg_rate",
            title="Taxa Média de Homicídio por Ano",
            template="plotly_white",
            labels={"year": "Ano", "avg_rate": "Taxa Média"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Nenhum dado disponível para os filtros selecionados.")

# ---------------------------------------------------------------------------
# Tab 2: Mapa Mundial
# ---------------------------------------------------------------------------

with tabs[1]:
    st.header("Mapa Mundial de Homicidios")

    # Year selector for map
    map_years = sorted(df["year"].unique())
    map_year = st.selectbox(
        "Selecione o ano",
        options=map_years,
        index=len(map_years) - 1,
        key="map_year",
    )

    # Sex selector for map
    map_sex = st.radio(
        "Sexo", options=["Total", "Female", "Male"], index=0, key="map_sex", horizontal=True
    )

    # Filter data for map
    df_map = df[
        (df["year"] == map_year)
        & (df["sex"] == map_sex)
    ]

    # Aggregate to one row per country (in case of multiple dimension/category/age)
    df_map_agg = (
        df_map.groupby(["iso3_code", "country"])["value"]
        .mean()
        .reset_index()
    )

    if not df_map_agg.empty:
        fig_map = px.choropleth(
            df_map_agg,
            locations="iso3_code",
            color="value",
            hover_name="country",
            color_continuous_scale="YlOrRd",
            title=f"Taxa de Homicidio por Pais ({map_year} - {map_sex})",
            template="plotly_white",
            labels={"value": "Taxa (por 100k)", "iso3_code": "ISO3"},
        )
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            height=600,
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.caption(
            f"Mapa mostrando a taxa de homicidio por 100.000 habitantes "
            f"para o ano {map_year}, sexo: {map_sex}. "
            f"Paises sem dados aparecem em cinza."
        )
    else:
        st.info("Nenhum dado disponivel para o ano e sexo selecionados.")

# ---------------------------------------------------------------------------
# Tab 3: Rankings
# ---------------------------------------------------------------------------

with tabs[2]:
    st.header("Rankings de Países")

    if not df_filtered.empty:
        # Top 10 countries by average homicide rate
        st.subheader("Top 10 — Maiores Taxas Médias de Homicídio")
        top10 = (
            df_filtered.groupby("country")["value"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        top10.columns = ["country", "avg_rate"]
        fig_top10 = px.bar(
            top10,
            x="country",
            y="avg_rate",
            title="Top 10 Países — Maior Taxa Média de Homicídio",
            template="plotly_white",
            labels={"country": "País", "avg_rate": "Taxa Média"},
        )
        st.plotly_chart(fig_top10, use_container_width=True)

        # Bottom 10 countries by average homicide rate
        st.subheader("Top 10 — Menores Taxas Médias de Homicídio")
        bottom10 = (
            df_filtered.groupby("country")["value"]
            .mean()
            .nsmallest(10)
            .reset_index()
        )
        bottom10.columns = ["country", "avg_rate"]
        fig_bottom10 = px.bar(
            bottom10,
            x="country",
            y="avg_rate",
            title="Top 10 Países — Menor Taxa Média de Homicídio",
            template="plotly_white",
            labels={"country": "País", "avg_rate": "Taxa Média"},
        )
        st.plotly_chart(fig_bottom10, use_container_width=True)
    else:
        st.info("Nenhum dado disponível para os filtros selecionados.")

# ---------------------------------------------------------------------------
# Tab 3: Violência contra Mulheres
# ---------------------------------------------------------------------------

with tabs[3]:
    st.header("Violência contra Mulheres")

    # Filter for female data regardless of sidebar sex selection
    df_female = df[
        (df["sex"] == "Female")
        & (df["year"] >= selected_years[0])
        & (df["year"] <= selected_years[1])
    ]
    if selected_countries:
        df_female = df_female[df_female["country"].isin(selected_countries)]
    if selected_regions:
        df_female = df_female[df_female["region"].isin(selected_regions)]
    if selected_subregions:
        df_female = df_female[df_female["subregion"].isin(selected_subregions)]

    if not df_female.empty:
        # Top countries for female homicide
        st.subheader("Top 10 Países — Homicídio Feminino (Taxa Média)")
        top_female = (
            df_female.groupby("country")["value"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        top_female.columns = ["country", "avg_rate"]
        fig_female_top = px.bar(
            top_female,
            x="country",
            y="avg_rate",
            title="Top 10 Países — Maior Taxa de Homicídio Feminino",
            template="plotly_white",
            labels={"country": "País", "avg_rate": "Taxa Média"},
            color_discrete_sequence=["#e377c2"],
        )
        st.plotly_chart(fig_female_top, use_container_width=True)

        # Comparison Male vs Female
        st.subheader("Comparação: Homicídio Masculino vs Feminino")
        df_male = df[
            (df["sex"] == "Male")
            & (df["year"] >= selected_years[0])
            & (df["year"] <= selected_years[1])
        ]
        if selected_countries:
            df_male = df_male[df_male["country"].isin(selected_countries)]
        if selected_regions:
            df_male = df_male[df_male["region"].isin(selected_regions)]
        if selected_subregions:
            df_male = df_male[df_male["subregion"].isin(selected_subregions)]

        female_avg = df_female.groupby("year")["value"].mean().reset_index()
        female_avg["sex"] = "Female"
        male_avg = df_male.groupby("year")["value"].mean().reset_index()
        male_avg["sex"] = "Male"
        comparison = pd.concat([female_avg, male_avg], ignore_index=True)

        fig_comparison = px.line(
            comparison,
            x="year",
            y="value",
            color="sex",
            title="Tendência: Taxa Média de Homicídio — Masculino vs Feminino",
            template="plotly_white",
            labels={"year": "Ano", "value": "Taxa Média", "sex": "Sexo"},
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.info("Nenhum dado feminino disponível para os filtros selecionados.")

# ---------------------------------------------------------------------------
# Tab 4: Séries Temporais
# ---------------------------------------------------------------------------

with tabs[4]:
    st.header("Séries Temporais")

    if not df_filtered.empty:
        # Country selector for time series
        available_countries = sorted(df_filtered["country"].unique().tolist())
        ts_countries = st.multiselect(
            "Selecione países para comparar",
            options=available_countries,
            default=available_countries[:3] if len(available_countries) >= 3 else available_countries,
            key="ts_countries",
        )

        if ts_countries:
            df_ts = df_filtered[df_filtered["country"].isin(ts_countries)]
            ts_data = (
                df_ts.groupby(["year", "country"])["value"]
                .mean()
                .reset_index()
            )

            fig_ts = px.line(
                ts_data,
                x="year",
                y="value",
                color="country",
                title="Séries Temporais — Taxa de Homicídio por País",
                template="plotly_white",
                labels={"year": "Ano", "value": "Taxa", "country": "País"},
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Year-over-year analysis
            st.subheader("Variação Ano a Ano")
            yoy_data = ts_data.copy()
            yoy_data["yoy_change"] = yoy_data.groupby("country")["value"].pct_change() * 100
            yoy_data = yoy_data.dropna(subset=["yoy_change"])

            if not yoy_data.empty:
                fig_yoy = px.bar(
                    yoy_data,
                    x="year",
                    y="yoy_change",
                    color="country",
                    title="Variação Percentual Ano a Ano (%)",
                    template="plotly_white",
                    labels={
                        "year": "Ano",
                        "yoy_change": "Variação (%)",
                        "country": "País",
                    },
                    barmode="group",
                )
                st.plotly_chart(fig_yoy, use_container_width=True)
        else:
            st.info("Selecione pelo menos um país para visualizar.")
    else:
        st.info("Nenhum dado disponível para os filtros selecionados.")

# ---------------------------------------------------------------------------
# Tab 5: Regressão e Predição
# ---------------------------------------------------------------------------

with tabs[5]:
    st.header("Regressão e Predição")

    # Country selector for regression
    regression_countries = sorted(df_regression["country"].unique().tolist())
    reg_country = st.selectbox(
        "Selecione um país para regressão",
        options=regression_countries,
        index=regression_countries.index("Brazil") if "Brazil" in regression_countries else 0,
        key="reg_country",
    )

    if reg_country:
        df_country_reg = df_regression[df_regression["country"] == reg_country]

        if len(df_country_reg) >= 5:
            try:
                # Prepare data and train models
                X_train, X_test, y_train, y_test = prepare_regression_data(
                    df_regression, reg_country
                )
                lr_model = train_linear_regression(X_train, y_train)
                rf_model = train_random_forest(X_train, y_train)

                # Evaluate models
                lr_metrics = evaluate_model(lr_model, X_test, y_test)
                rf_metrics = evaluate_model(rf_model, X_test, y_test)

                # Display model comparison metrics
                st.subheader("Comparação de Modelos")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Regressão Linear**")
                    st.metric("MAE", f"{lr_metrics['MAE']:.4f}")
                    st.metric("MSE", f"{lr_metrics['MSE']:.4f}")
                    st.metric("RMSE", f"{lr_metrics['RMSE']:.4f}")
                    st.metric("R²", f"{lr_metrics['R2']:.4f}")

                with col2:
                    st.markdown("**Random Forest**")
                    st.metric("MAE", f"{rf_metrics['MAE']:.4f}")
                    st.metric("MSE", f"{rf_metrics['MSE']:.4f}")
                    st.metric("RMSE", f"{rf_metrics['RMSE']:.4f}")
                    st.metric("R²", f"{rf_metrics['R2']:.4f}")

                # Generate predictions
                future_years = [2023, 2024, 2025, 2026]
                predictions_lr = predict_future(lr_model, future_years)
                predictions_rf = predict_future(rf_model, future_years)

                # Historical data for charts
                df_hist = df_country_reg[["year", "value"]].copy()

                # Historical + Predictions charts
                st.subheader("Histórico e Predições")

                col_lr, col_rf = st.columns(2)
                with col_lr:
                    fig_hist_lr = plot_historical_and_predictions(
                        df_hist,
                        predictions_lr,
                        f"{reg_country} — Regressão Linear",
                    )
                    st.plotly_chart(fig_hist_lr, use_container_width=True)

                with col_rf:
                    fig_hist_rf = plot_historical_and_predictions(
                        df_hist,
                        predictions_rf,
                        f"{reg_country} — Random Forest",
                    )
                    st.plotly_chart(fig_hist_rf, use_container_width=True)

                # Real vs Predicted charts
                st.subheader("Real vs Predito (Conjunto de Teste)")
                y_pred_lr = lr_model.predict(X_test)
                y_pred_rf = rf_model.predict(X_test)

                col_rvp_lr, col_rvp_rf = st.columns(2)
                with col_rvp_lr:
                    fig_rvp_lr = plot_real_vs_predicted(
                        y_test, y_pred_lr, f"{reg_country} — Regressão Linear"
                    )
                    st.plotly_chart(fig_rvp_lr, use_container_width=True)

                with col_rvp_rf:
                    fig_rvp_rf = plot_real_vs_predicted(
                        y_test, y_pred_rf, f"{reg_country} — Random Forest"
                    )
                    st.plotly_chart(fig_rvp_rf, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao executar regressão: {e}")
        else:
            st.warning(
                f"Dados insuficientes para {reg_country}. "
                "São necessários pelo menos 5 registros para treinar o modelo."
            )

# ---------------------------------------------------------------------------
# Tab 6: Estatísticas Gerais
# ---------------------------------------------------------------------------

with tabs[6]:
    st.header("Estatísticas Gerais")

    if not df_filtered.empty:
        # Descriptive statistics table
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df_filtered.describe(), use_container_width=True)

        # Distribution histogram
        st.subheader("Distribuição das Taxas de Homicídio")
        fig_dist = px.histogram(
            df_filtered,
            x="value",
            nbins=50,
            title="Distribuição das Taxas de Homicídio",
            template="plotly_white",
            labels={"value": "Taxa de Homicídio", "count": "Frequência"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Statistical questions answers
        st.subheader("Respostas às Perguntas Estatísticas")

        with st.expander("Ver respostas das 10 perguntas estatísticas"):
            try:
                answers = answer_all_questions(df)

                st.markdown("**1. Top 10 países por taxa média nos últimos 5 anos:**")
                st.dataframe(answers["q1"], use_container_width=True)

                st.markdown("**2. Top 10 países por homicídio feminino em 2022:**")
                st.dataframe(answers["q2"], use_container_width=True)

                st.markdown("**3. Regiões com mais homicídios:**")
                st.dataframe(answers["q3"], use_container_width=True)

                st.markdown("**4. Países com menor taxa por sub-região:**")
                st.dataframe(answers["q4"], use_container_width=True)

                st.markdown("**5. Países com menores taxas de homicídio feminino:**")
                st.dataframe(answers["q5"], use_container_width=True)

                st.markdown("**6. Sub-regiões com mais homicídios:**")
                st.dataframe(answers["q6"], use_container_width=True)

                st.markdown("**7. País com maior taxa por continente em 2020:**")
                st.dataframe(answers["q7"], use_container_width=True)

                st.markdown("**8. País mais violento para mulheres em 2021:**")
                st.dataframe(answers["q8"], use_container_width=True)

                st.markdown("**9. País com maior valor de indicador (todos os anos):**")
                st.dataframe(answers["q9"], use_container_width=True)

                st.markdown("**10. Média do Brasil nos últimos 10 anos:**")
                st.metric("Taxa Média — Brasil", f"{answers['q10']:.2f}")

            except Exception as e:
                st.error(f"Erro ao calcular respostas estatísticas: {e}")
    else:
        st.info("Nenhum dado disponível para os filtros selecionados.")
