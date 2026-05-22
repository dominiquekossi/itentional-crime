# Apresentação do Projeto

## Análise de Homicídios Intencionais Globais

**Disciplina:** Tópicos Especiais em Computação I

---

## 1. Objetivo do Trabalho

Analisar o dataset com índices de homicídios em todo o mundo, para compreender e extrair informações importantes na compreensão desse fenômeno.

O projeto utiliza estatística descritiva, bibliotecas de gráficos e Pandas para obter descobertas importantes e entender o comportamento de taxas de homicídios pelo mundo, além de definir uma estratégia de regressão para predizer a taxa de homicídios nos anos seguintes: **2023, 2024, 2025 e 2026**.

---

## 2. Dataset Utilizado

- **Fonte:** UNODC — United Nations Office on Drugs and Crime
- **Arquivo:** `data_cts_intentional_homicide.xlsx`
- **Conteúdo:** Taxas de homicídio intencional por 100.000 habitantes
- **Abrangência:** Dados globais por país, região, sub-região e sexo
- **Período:** 1990–2022

Após a limpeza, o dataset ficou com **40.483 registros** de **195+ países**.

---

## 3. Tecnologias e Bibliotecas

| Biblioteca             | Uso no Projeto                                                      |
| ---------------------- | ------------------------------------------------------------------- |
| **Pandas**             | Manipulação, limpeza e análise dos dados                            |
| **NumPy**              | Operações numéricas e arrays                                        |
| **Plotly**             | Gráficos interativos (histograma, boxplot, linhas, barras, heatmap) |
| **Scikit-learn**       | Modelos de regressão (Linear e Random Forest)                       |
| **Streamlit**          | Dashboard web interativo                                            |
| **Matplotlib/Seaborn** | Suporte a visualizações estáticas                                   |

---

## 4. Pipeline de Dados

```
Excel Bruto → Limpeza → Filtragem → Conversão de Tipos → Exportação CSV
```

**Etapas da limpeza:**

1. Padronização de colunas para snake_case
2. Filtro por indicador: "Victims of intentional homicide"
3. Filtro por unidade: "Rate per 100,000 population"
4. Remoção de agregações globais (World, Global, Various, Unknown)
5. Remoção de valores nulos
6. Conversão de tipos (ano → inteiro, valor → float)

---

## 5. Análise Exploratória (EDA)

### Estatística Descritiva

- Distribuição das taxas de homicídio (histograma)
- Comparação entre regiões (boxplot)
- Tendências temporais dos países mais violentos (gráfico de linhas)
- Rankings de países (gráfico de barras)
- Correlação entre variáveis numéricas (heatmap)

### 10 Perguntas Estatísticas Respondidas

| #   | Pergunta                                                      |
| --- | ------------------------------------------------------------- |
| 1   | Top 10 países com maior taxa média nos últimos 5 anos         |
| 2   | Top 10 países com maior taxa de homicídio feminino em 2022    |
| 3   | Regiões com maiores totais de homicídios                      |
| 4   | Países com menor taxa de homicídio por sub-região             |
| 5   | Países com menores taxas de homicídio feminino                |
| 6   | Sub-regiões com maiores totais de homicídios                  |
| 7   | País com maior taxa por continente em 2020                    |
| 8   | País mais violento para mulheres em 2021                      |
| 9   | País com maior valor individual de indicador em todos os anos |
| 10  | Taxa média de homicídio do Brasil nos últimos 10 anos         |

---

## 6. Estratégia de Regressão

### Modelos Utilizados

| Modelo               | Descrição                                                 |
| -------------------- | --------------------------------------------------------- |
| **Regressão Linear** | Modelo simples, relação linear entre ano e taxa           |
| **Random Forest**    | Modelo ensemble com 100 árvores, captura não-linearidades |

### Metodologia

- **Feature:** Ano (variável independente)
- **Target:** Taxa de homicídio por 100.000 habitantes
- **Split:** 80% treino / 20% teste (random_state=42)
- **País de referência:** Brasil

### Métricas de Avaliação

| Métrica | Descrição                     |
| ------- | ----------------------------- |
| MAE     | Erro Absoluto Médio           |
| MSE     | Erro Quadrático Médio         |
| RMSE    | Raiz do Erro Quadrático Médio |
| R²      | Coeficiente de Determinação   |

### Previsões Geradas (Brasil)

| Ano  | Regressão Linear | Random Forest |
| ---- | ---------------- | ------------- |
| 2023 | ~19.27           | ~16.34        |
| 2024 | ~19.10           | ~16.34        |
| 2025 | ~18.92           | ~16.34        |
| 2026 | ~18.74           | ~16.34        |

_Valores aproximados por 100.000 habitantes_

---

## 7. Principais Descobertas

1. **Distribuição assimétrica:** A maioria dos países tem taxas baixas, com poucos outliers extremos (principalmente nas Américas e África).

2. **Disparidade regional:** Américas e África possuem taxas significativamente maiores que Europa e Ásia.

3. **Tendência de queda:** A maioria dos países apresenta redução nas taxas ao longo das últimas décadas.

4. **Violência de gênero:** A taxa masculina é consistentemente superior à feminina em todas as regiões. A violência contra mulheres tem padrões geográficos distintos.

5. **Brasil:** Taxa média de ~26.5 por 100.000 nos últimos 10 anos, com tendência de redução. Os modelos preveem continuidade dessa queda até 2026.

6. **Random Forest vs Linear:** A Regressao Linear e mais adequada para extrapolacao temporal. O Random Forest nao extrapola bem fora do intervalo observado e foi usado apenas como modelo comparativo. O split e temporal (treino ate 2018, teste 2019-2022).

---

## 8. Entregáveis do Projeto

| Entregável                           | Descrição                                                   |
| ------------------------------------ | ----------------------------------------------------------- |
| `notebooks/analise_homicidios.ipynb` | Notebook completo com toda a análise                        |
| `app/streamlit_app.py`               | Dashboard interativo com 7 abas e mapa mundial              |
| `src/cleaning.py`                    | Módulo de limpeza (taxa + contagem + duplicados + features) |
| `src/eda.py`                         | Módulo de EDA (Plotly + Seaborn + Matplotlib)               |
| `src/regression.py`                  | Módulo de regressão com split temporal                      |
| `outputs/dataset_limpo.csv`          | Dataset de taxa limpo                                       |
| `outputs/dataset_counts.csv`         | Dataset de contagem limpo                                   |
| `outputs/dataset_regressao.csv`      | Dataset preparado para regressão                            |
| `images/`                            | Gráficos exportados (Plotly + Seaborn)                      |
| `presentation/roteiro_slides.md`     | Estrutura de slides para apresentação                       |

---

## 9. Como Executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar pipeline de limpeza
python -c "from src.cleaning import run_cleaning_pipeline; run_cleaning_pipeline('data/data_cts_intentional_homicide.xlsx', 'outputs')"

# 3. Iniciar dashboard
python -m streamlit run app/streamlit_app.py

# Ou simplesmente:
run.bat
```

---

## 10. Membros da Equipe

| Nome     | Responsabilidade                |
| -------- | ------------------------------- |
| Membro 1 | Limpeza de dados e pipeline     |
| Membro 2 | Análise exploratória e gráficos |
| Membro 3 | Modelos de regressão            |
| Membro 4 | Dashboard Streamlit             |
| Membro 5 | Documentação e notebook         |

---

## 11. Informações de Entrega

- **Data de entrega:** 27/05/2026
- **Upload no SIGAA** com as respostas de todas as perguntas e arquivos do projeto
- **Apresentação em sala:** Uma das perguntas será sorteada para apresentação oral + exposição do Data App (Dashboard Streamlit) com a regressão

### O que entregar no SIGAA:

1. Notebook completo (`analise_homicidios.ipynb`) com respostas das 10 perguntas
2. Código-fonte do projeto (módulos `src/`, dashboard `app/`)
3. Dataset limpo (`outputs/`)
4. README com instruções de execução

### Na apresentação em sala:

1. Responder a pergunta sorteada com base nos resultados do notebook
2. Demonstrar o Data App (Streamlit) ao vivo com os filtros e gráficos
3. Mostrar a aba de Regressão e Predição com as previsões para 2023-2026

---

## 12. Conclusão

O projeto demonstrou que é possível extrair informações valiosas do dataset de homicídios globais utilizando técnicas de ciência de dados. A combinação de estatística descritiva, visualizações interativas e modelos preditivos permite compreender padrões, identificar tendências e gerar previsões fundamentadas para os próximos anos.

A estratégia de regressão com dois modelos (Linear e Random Forest) mostrou que o Brasil apresenta tendência de redução nas taxas de homicídio, com previsões para 2023-2026 indicando continuidade desse padrão.
