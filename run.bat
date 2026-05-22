@echo off
echo ============================================
echo   Analise de Homicidios Globais - Setup
echo ============================================
echo.

:: Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale Python 3.10+ e tente novamente.
    pause
    exit /b 1
)

:: Criar ambiente virtual se nao existir
if not exist "venv" (
    echo [1/4] Criando ambiente virtual...
    python -m venv venv
) else (
    echo [1/4] Ambiente virtual ja existe.
)

:: Ativar ambiente virtual
echo [2/4] Ativando ambiente virtual...
call venv\Scripts\activate.bat

:: Instalar dependencias
echo [3/4] Instalando dependencias...
pip install -r requirements.txt --quiet

:: Executar pipeline de limpeza
echo [4/4] Executando pipeline de limpeza de dados...
python -c "from src.cleaning import run_cleaning_pipeline; df = run_cleaning_pipeline('data/data_cts_intentional_homicide.xlsx', 'outputs'); print(f'  Pipeline concluido: {df.shape[0]} linhas processadas')"

if errorlevel 1 (
    echo [ERRO] Falha ao executar o pipeline de limpeza.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup concluido com sucesso!
echo ============================================
echo.
echo Arquivos gerados:
echo   - outputs/dataset_limpo.csv
echo   - outputs/dataset_regressao.csv
echo.
echo Iniciando o Dashboard Streamlit...
echo Acesse: http://localhost:8501
echo.
echo Pressione Ctrl+C para encerrar o dashboard.
echo.

python -m streamlit run app/streamlit_app.py --server.headless true
