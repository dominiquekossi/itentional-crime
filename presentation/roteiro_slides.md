# Roteiro de Slides para Apresentacao

## Slide 1 - Capa

- Titulo: Analise de Homicidios Intencionais Globais
- Disciplina: Topicos Especiais em Computacao I
- Nomes da equipe

## Slide 2 - Objetivo

- Analisar dataset UNODC de homicidios globais
- Extrair informacoes com estatistica descritiva
- Predizer taxas para 2023-2026

## Slide 3 - Dataset

- Fonte: UNODC
- 40.000+ registros
- 195+ paises
- Periodo: 1990-2022
- Variaveis: pais, regiao, sub-regiao, ano, sexo, taxa, contagem

## Slide 4 - Limpeza de Dados

- Padronizacao de colunas
- Filtragem por indicador
- Remocao de agregacoes globais
- Tratamento de nulos e duplicados
- Separacao: dataset de taxa vs contagem

## Slide 5 - EDA: Distribuicao

- Histograma das taxas
- Maioria dos paises com taxas baixas
- Poucos outliers extremos (El Salvador, Jamaica)

## Slide 6 - EDA: Rankings

- Top 10 paises mais violentos
- Jamaica, Honduras, Africa do Sul lideram
- 7 de 10 sao do Caribe/America Central

## Slide 7 - EDA: Regioes

- Americas dominam em numeros absolutos
- Africa e Americas lideram em taxas
- Europa e Asia com taxas baixas

## Slide 8 - EDA: Violencia contra Mulheres

- Caribe lidera em homicidio feminino
- Taxa masculina 4-5x maior que feminina
- Padroes geograficos distintos

## Slide 9 - EDA: Brasil

- Taxa media: 26.54 por 100k (ultimos 10 anos)
- 4x acima da media mundial
- Tendencia de queda recente

## Slide 10 - Regressao: Metodologia

- Modelos: Linear Regression e Random Forest
- Split temporal: treino ate 2018, teste 2019-2022
- Feature: ano | Target: taxa de homicidio
- Metricas: MAE, RMSE, R2

## Slide 11 - Regressao: Resultados

- Linear Regression melhor para extrapolacao
- Random Forest limitado fora do intervalo observado
- Previsoes 2023-2026 indicam queda no Brasil

## Slide 12 - Data App (Streamlit)

- Dashboard interativo com filtros
- 6 abas: Visao Geral, Rankings, Mulheres, Series, Regressao, Estatisticas
- Graficos Plotly interativos
- Demonstracao ao vivo

## Slide 13 - Conclusoes

- Violencia concentrada em Americas e Africa
- Tendencia global de reducao
- Modelos preditivos confirmam queda no Brasil
- Random Forest como comparativo, Linear para extrapolacao

## Slide 14 - Limitacoes

- Dados dependem de registros oficiais
- Regressao usa apenas ano como feature
- Random Forest nao extrapola bem
- Alguns paises com dados incompletos

## Slide 15 - Referencias

- UNODC Data Portal
- Scikit-learn documentation
- Plotly documentation
- Streamlit documentation
