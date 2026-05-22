# Análise de Homicídios Intencionais Globais

## Descrição

Este projeto realiza uma análise abrangente dos dados globais de homicídios intencionais, utilizando o dataset do Escritório das Nações Unidas sobre Drogas e Crime (UNODC). O sistema carrega, limpa e analisa os dados para produzir análises exploratórias, responder perguntas estatísticas, construir modelos de regressão para previsões futuras e apresentar os resultados por meio de um dashboard interativo.

## Objetivos

- Realizar a limpeza e padronização dos dados brutos de homicídios globais
- Conduzir análise exploratória de dados (EDA) com visualizações interativas
- Responder 10 perguntas estatísticas sobre padrões de homicídios no mundo
- Treinar modelos de regressão (Linear e Random Forest) para prever taxas futuras
- Desenvolver um dashboard interativo com Streamlit para exploração dos dados
- Documentar toda a análise em um Jupyter Notebook acadêmico

## Contexto Acadêmico

Projeto desenvolvido como entrega acadêmica para a disciplina **Tópicos Especiais em Computação I**. O trabalho aborda técnicas de ciência de dados aplicadas à análise de indicadores de segurança pública global, integrando conceitos de manipulação de dados, visualização, modelagem preditiva e desenvolvimento de aplicações web.

---

## Tecnologias Utilizadas

| Biblioteca       | Descrição                                                  |
| ---------------- | ---------------------------------------------------------- |
| **pandas**       | Manipulação e análise de dados tabulares (DataFrames)      |
| **numpy**        | Operações numéricas e manipulação de arrays                |
| **plotly**       | Criação de gráficos interativos para dashboard e notebook  |
| **matplotlib**   | Suporte para geração de gráficos estáticos                 |
| **seaborn**      | Visualizações estatísticas avançadas                       |
| **scikit-learn** | Modelos de regressão (Linear Regression e Random Forest)   |
| **streamlit**    | Framework para construção do dashboard web interativo      |
| **openpyxl**     | Leitura de arquivos Excel (.xlsx)                          |
| **kaleido**      | Exportação de gráficos Plotly como imagens estáticas (PNG) |
| **nbformat**     | Criação e manipulação programática de Jupyter Notebooks    |

---

## Instalação

### Pré-requisitos

- Python 3.10 ou superior
- Git

### Passo a passo

1. **Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/itentional-crime.git
cd itentional-crime
```

2. **Crie e ative o ambiente virtual:**

```bash
python -m venv venv
```

- No Windows:

```bash
venv\Scripts\activate
```

- No Linux/macOS:

```bash
source venv/bin/activate
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

---

## Execução

### 1. Pipeline de Limpeza de Dados

Execute o pipeline de limpeza para gerar os datasets processados:

```bash
python -c "from src.cleaning import run_cleaning_pipeline; run_cleaning_pipeline('data/data_cts_intentional_homicide.xlsx', 'outputs')"
```

Isso gerará os arquivos `outputs/dataset_limpo.csv` e `outputs/dataset_regressao.csv`.

### 2. Jupyter Notebook

Para abrir o notebook com a análise completa:

```bash
jupyter notebook notebooks/analise_homicidios.ipynb
```

### 3. Dashboard Streamlit

Para iniciar o dashboard interativo:

```bash
streamlit run app/streamlit_app.py
```

> **Nota:** O dashboard requer que o pipeline de limpeza tenha sido executado previamente, pois carrega os dados de `outputs/dataset_limpo.csv`.

---

## Estrutura do Projeto

```
itentional-crime/
├── app/
│   └── streamlit_app.py          # Dashboard interativo Streamlit
├── data/
│   └── data_cts_intentional_homicide.xlsx  # Dataset bruto UNODC
├── images/
│   ├── bar_top10.png             # Gráfico de barras - Top 10 países
│   ├── boxplot_region.png        # Boxplot por região
│   ├── heatmap_correlation.png   # Mapa de calor de correlações
│   ├── histogram_rates.png       # Histograma de taxas
│   ├── line_trends.png           # Gráfico de tendências temporais
│   ├── regression_lr_historical.png    # Regressão Linear - histórico
│   ├── regression_lr_real_vs_pred.png  # Regressão Linear - real vs previsto
│   ├── regression_rf_historical.png    # Random Forest - histórico
│   └── regression_rf_real_vs_pred.png  # Random Forest - real vs previsto
├── notebooks/
│   └── analise_homicidios.ipynb  # Notebook acadêmico completo
├── outputs/
│   ├── dataset_limpo.csv         # Dataset limpo e processado
│   └── dataset_regressao.csv     # Dataset preparado para regressão
├── scripts/
│   └── generate_notebook.py      # Script de geração do notebook
├── src/
│   ├── __init__.py               # Inicialização do pacote
│   ├── cleaning.py               # Módulo de limpeza de dados
│   ├── eda.py                    # Módulo de análise exploratória
│   ├── regression.py             # Módulo de regressão e previsão
│   └── utils.py                  # Funções utilitárias compartilhadas
├── tests/                        # Testes automatizados
├── .gitignore                    # Arquivos ignorados pelo Git
├── README.md                     # Documentação do projeto
└── requirements.txt              # Dependências Python
```

---

## Membros da Equipe

| Nome     | Função                          |
| -------- | ------------------------------- |
| Membro 1 | Limpeza de dados e pipeline     |
| Membro 2 | Análise exploratória e gráficos |
| Membro 3 | Modelos de regressão            |
| Membro 4 | Dashboard Streamlit             |
| Membro 5 | Documentação e notebook         |

---

## Resultados

### Principais Descobertas

- **Distribuição das taxas:** A distribuição global das taxas de homicídio apresenta forte assimetria positiva, com a maioria dos países concentrados em taxas baixas e poucos outliers com valores extremamente elevados.

- **Diferenças regionais:** Existem disparidades significativas entre regiões. Américas e África apresentam as maiores taxas médias de homicídio, enquanto Europa e Ásia possuem as menores taxas.

- **Análise por gênero:** A taxa de homicídio masculina é consistentemente superior à feminina em todas as regiões e períodos analisados. A violência contra mulheres apresenta padrões geográficos distintos.

- **Tendências temporais:** Observa-se uma tendência geral de redução nas taxas de homicídio ao longo das últimas décadas na maioria dos países, embora alguns apresentem estabilidade ou aumento.

- **Previsões por regressão:** Os modelos de regressão (Linear e Random Forest) foram treinados para prever taxas de homicídio para os anos de 2023 a 2026. O modelo Random Forest apresentou melhor desempenho em termos de métricas de erro (MAE, RMSE) e coeficiente de determinação (R²).

---

## Limitacoes do Estudo

### Sobre os Dados

- Os dados dependem da qualidade dos registros oficiais de cada pais
- Alguns paises podem ter subnotificacao significativa
- Dados incompletos em determinados periodos para alguns paises

### Sobre a Regressao

- **Regressao Linear** e mais adequada para extrapolacao temporal (prever anos futuros) pois assume tendencia linear continua
- **Random Forest** possui limitacao conhecida: nao consegue prever valores fora do intervalo observado no treino. Foi utilizado apenas como modelo comparativo
- A previsao utiliza apenas o ano como variavel preditora, nao considerando fatores socioeconomicos, politicas publicas ou eventos externos
- O split temporal (treino ate 2018, teste 2019-2022) garante que o modelo nunca "ve" dados futuros durante o treinamento

### Sobre as Analises

- Perguntas sobre "totais de homicidios" utilizam o dataset de contagem (Counts)
- Perguntas sobre "taxas" utilizam o dataset de taxa (Rate per 100,000)
- A taxa por 100k permite comparacao justa entre paises de tamanhos diferentes

---

## Licença

Projeto acadêmico — uso educacional.
