# Como Rodar o Projeto

## Pré-requisitos

- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)

---

## Opção 1: Script Automático (Recomendado)

Execute o script `run.bat` que faz tudo automaticamente:

```bash
run.bat
```

O script irá:

1. Criar o ambiente virtual (se não existir)
2. Instalar as dependências
3. Executar o pipeline de limpeza de dados
4. Iniciar o dashboard Streamlit

---

## Opção 2: Passo a Passo Manual

### 1. Criar ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o pipeline de limpeza

```bash
python -c "from src.cleaning import run_cleaning_pipeline; run_cleaning_pipeline('data/data_cts_intentional_homicide.xlsx', 'outputs')"
```

Isso gera os arquivos:

- `outputs/dataset_limpo.csv` — dataset completo limpo
- `outputs/dataset_regressao.csv` — dataset preparado para regressão

### 4. Iniciar o Dashboard Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

Acesse no navegador: **http://localhost:8501**

### 5. Abrir o Jupyter Notebook (opcional)

```bash
jupyter notebook notebooks/analise_homicidios.ipynb
```

---

## Estrutura dos Comandos

| Ação                  | Comando                                                                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Instalar dependências | `pip install -r requirements.txt`                                                                                                                                                                                   |
| Limpar dados          | `python run.bat` ou comando do passo 3                                                                                                                                                                              |
| Dashboard             | `python -m streamlit run app/streamlit_app.py`                                                                                                                                                                      |
| Notebook              | `jupyter notebook notebooks/analise_homicidios.ipynb`                                                                                                                                                               |
| Gerar gráficos        | `python -c "from src.eda import generate_all_charts; from src.utils import load_cleaned_data; df = load_cleaned_data('outputs/dataset_limpo.csv'); generate_all_charts(df, 'images')"`                              |
| Regressão (Brasil)    | `python -c "from src.regression import run_regression_pipeline; from src.utils import load_cleaned_data; df = load_cleaned_data('outputs/dataset_regressao.csv'); run_regression_pipeline(df, 'Brazil', 'images')"` |

---

## Observações

- O **passo 3** (pipeline de limpeza) é obrigatório antes de rodar o dashboard ou o notebook.
- O dashboard carrega os dados de `outputs/dataset_limpo.csv`.
- Os gráficos são salvos na pasta `images/`.
- Na primeira execução do Streamlit, pode aparecer um prompt pedindo email — basta pressionar Enter para pular.
