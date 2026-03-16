"""
Auto Data Scientist v4 — Pipeline End-to-End com CrewAI + Llama 3.1

Historico de correcoes:
  v2 - BUG 1: Tools sem parametro de caminho (sem alucinacao de path)
  v2 - BUG 2: Tools atomicas por etapa (sem loop infinito)
  v2 - BUG 3: Logging UTF-8 no Windows (sem UnicodeEncodeError)
  v3 - BUG 4: Storyteller e Git fora do Crew (sem alucinacao de acao)
  v4 - BUG 5: gerar_readme.func() em vez de gerar_readme() (Tool nao e callable)

Dependencias:
    pip install crewai kagglehub pandas pyarrow python-dotenv
    pip install scikit-learn matplotlib seaborn tabulate numpy

Variaveis de ambiente (.env):
    KAGGLE_USERNAME=seu_usuario
    KAGGLE_KEY=sua_chave_aqui
"""

# ==========================================
# 0. IMPORTS E SETUP INICIAL
# ==========================================
import os
import sys
import logging
import subprocess
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

load_dotenv()

# ==========================================
# FIX: Logging UTF-8 no Windows
# ==========================================
_utf8_handler = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
)
_utf8_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        _utf8_handler,
    ],
)
logger = logging.getLogger("AutoDS")

# ==========================================
# 1. CONFIGURACOES CENTRALIZADAS
# ==========================================
_BASE_DIR = os.getcwd()

CONFIG = {
    "dataset_slug": "rhythmghai/resume-screening-dataset-200k-candidates",
    "dataset_url":  "https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates",

    # Caminhos absolutos — LLM nunca os manipula
    "silver_path":   os.path.join(_BASE_DIR, "df1_silver.parquet"),
    "gold_path":     os.path.join(_BASE_DIR, "df2_gold.parquet"),
    "ml_ready_path": os.path.join(_BASE_DIR, "df3_ml_ready.parquet"),
    "stats_md":      os.path.join(_BASE_DIR, "Estatistica_Descritiva.md"),
    "corr_png":      os.path.join(_BASE_DIR, "matriz_correlacao.png"),
    "metrics_md":    os.path.join(_BASE_DIR, "Metricas_Modelo.md"),
    "model_pkl":     os.path.join(_BASE_DIR, "modelo_final.pkl"),
    "readme_md":     os.path.join(_BASE_DIR, "README.md"),

    "test_size":         0.2,
    "random_state":      42,
    "max_iter":          5,
    "max_retry_limit":   2,
}

# ==========================================
# 2. MODELOS LLM
# ==========================================
llm_analitico = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# ==========================================
# 3. FERRAMENTAS (TOOLS)
#
# REGRAS DE DESIGN:
#   - Nenhuma tool recebe caminho como parametro (elimina alucinacao de path)
#   - Agentes analiticos: uma tool de acao + uma tool de confirmacao
#   - README e Git: executados via .func() fora do Crew (elimina alucinacao de acao)
# ==========================================

# ── ETAPA 1: Ingestao ────────────────────────────────────────────────────────

@tool("baixar_e_salvar_silver")
def baixar_e_salvar_silver(_: str = "") -> str:
    """
    Baixa o dataset de curriculos do Kaggle e salva como df1_silver.parquet.
    Nao requer parametros. Usa as credenciais do arquivo .env automaticamente.
    """
    import kagglehub

    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key  = os.getenv("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        return (
            "ERRO: KAGGLE_USERNAME e/ou KAGGLE_KEY nao encontrados no .env. "
            "Crie o arquivo .env com essas credenciais."
        )

    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"]      = kaggle_key

    try:
        slug = CONFIG["dataset_slug"]
        logger.info(f"[Ingestor] Iniciando download: {slug}")
        path = kagglehub.dataset_download(slug)
        logger.info(f"[Ingestor] Pasta temporaria: {path}")

        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csvs:
            return f"ERRO: Nenhum .csv encontrado. Conteudo: {os.listdir(path)}"

        csv_path = os.path.join(path, csvs[0])
        logger.info(f"[Ingestor] Lendo: {csvs[0]}")

        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )

        destino = CONFIG["silver_path"]
        df.to_parquet(destino, index=False)
        logger.info(f"[Ingestor] Salvo: {destino} — {df.shape[0]} linhas, {df.shape[1]} colunas.")

        return (
            f"SUCESSO_INGESTAO\n"
            f"Arquivo: {destino}\n"
            f"Shape: {df.shape}\n"
            f"Colunas: {list(df.columns)}"
        )
    except Exception as e:
        logger.error(f"[Ingestor] Falha: {e}")
        return f"ERRO na ingestao Kaggle: {e}"


@tool("confirmar_silver_existe")
def confirmar_silver_existe(_: str = "") -> str:
    """
    Confirma se df1_silver.parquet foi salvo com sucesso.
    Nao requer parametros. Retorna OK ou ERRO.
    """
    path = CONFIG["silver_path"]
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        return f"OK: df1_silver.parquet existe. Tamanho: {size_kb:.1f} KB"
    return "ERRO: df1_silver.parquet NAO encontrado. Execute baixar_e_salvar_silver primeiro."


# ── ETAPA 2: Features ────────────────────────────────────────────────────────

@tool("gerar_features_e_estatistica")
def gerar_features_e_estatistica(_: str = "") -> str:
    """
    Le df1_silver.parquet, gera estatistica descritiva em Markdown,
    cria features automaticas e salva df2_gold.parquet.
    Nao requer parametros.
    """
    try:
        df = pd.read_parquet(CONFIG["silver_path"])
        logger.info(f"[Features] Carregado Silver: {df.shape}")

        df.describe(include="all").to_markdown(CONFIG["stats_md"])

        num_cols = df.select_dtypes(include="number").columns.tolist()
        features_criadas = []
        if len(num_cols) >= 2:
            df["feat_ratio"] = df[num_cols[0]] / (df[num_cols[1]] + 1e-9)
            df["feat_soma"]  = df[num_cols[0]] + df[num_cols[1]]
            features_criadas = ["feat_ratio", "feat_soma"]

        df.to_parquet(CONFIG["gold_path"], index=False)

        return (
            f"SUCESSO_FEATURES\n"
            f"Features criadas: {features_criadas}\n"
            f"Shape Gold: {df.shape}\n"
            f"Arquivos gerados: Estatistica_Descritiva.md e df2_gold.parquet"
        )
    except Exception as e:
        return f"ERRO em gerar_features_e_estatistica: {e}"


@tool("confirmar_gold_existe")
def confirmar_gold_existe(_: str = "") -> str:
    """
    Confirma se df2_gold.parquet foi gerado com sucesso.
    Nao requer parametros.
    """
    path = CONFIG["gold_path"]
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        return f"OK: df2_gold.parquet existe. Tamanho: {size_kb:.1f} KB"
    return "ERRO: df2_gold.parquet NAO encontrado. Execute gerar_features_e_estatistica primeiro."


# ── ETAPA 3: EDA ─────────────────────────────────────────────────────────────

@tool("gerar_eda_e_ml_ready")
def gerar_eda_e_ml_ready(_: str = "") -> str:
    """
    Le df2_gold.parquet, gera matriz de correlacao como PNG,
    remove colunas redundantes/IDs e salva df3_ml_ready.parquet.
    Nao requer parametros.
    """
    try:
        df     = pd.read_parquet(CONFIG["gold_path"])
        num_df = df.select_dtypes(include="number")

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            num_df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5
        )
        plt.title("Matriz de Correlacao", fontsize=14)
        plt.tight_layout()
        plt.savefig(CONFIG["corr_png"], dpi=150)
        plt.close()

        corr_matrix      = num_df.corr().abs()
        upper            = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        cols_redundantes = [c for c in upper.columns if any(upper[c] > 0.95)]
        cols_id          = [c for c in df.columns if "id" in c.lower()]
        cols_remover     = list(set(cols_redundantes + cols_id))

        df_filtrado = df.drop(columns=cols_remover, errors="ignore")
        df_filtrado.to_parquet(CONFIG["ml_ready_path"], index=False)

        return (
            f"SUCESSO_EDA\n"
            f"Colunas removidas: {cols_remover}\n"
            f"Shape ML-Ready: {df_filtrado.shape}\n"
            f"Arquivos gerados: matriz_correlacao.png e df3_ml_ready.parquet"
        )
    except Exception as e:
        return f"ERRO em gerar_eda_e_ml_ready: {e}"


@tool("confirmar_ml_ready_existe")
def confirmar_ml_ready_existe(_: str = "") -> str:
    """
    Confirma se df3_ml_ready.parquet foi gerado com sucesso.
    Nao requer parametros.
    """
    path = CONFIG["ml_ready_path"]
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        return f"OK: df3_ml_ready.parquet existe. Tamanho: {size_kb:.1f} KB"
    return "ERRO: df3_ml_ready.parquet NAO encontrado. Execute gerar_eda_e_ml_ready primeiro."


# ── ETAPA 4: Machine Learning ─────────────────────────────────────────────────

@tool("treinar_e_salvar_modelo")
def treinar_e_salvar_modelo(_: str = "") -> str:
    """
    Le df3_ml_ready.parquet, detecta automaticamente se e classificacao ou regressao,
    treina RandomForest, salva modelo.pkl e Metricas_Modelo.md.
    Nao requer parametros.
    """
    try:
        df       = pd.read_parquet(CONFIG["ml_ready_path"]).dropna()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if len(num_cols) < 2:
            return "ERRO: Colunas numericas insuficientes para treinar modelo."

        target_col   = num_cols[-1]
        feature_cols = [c for c in num_cols if c != target_col]

        X = pd.get_dummies(df[feature_cols])
        y = df[target_col]

        is_classificacao = y.nunique() <= 20 and y.dtype in ["object", "int64", "int32"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
        )

        if is_classificacao:
            modelo  = RandomForestClassifier(n_estimators=100, random_state=CONFIG["random_state"])
            modelo.fit(X_train, y_train)
            y_pred  = modelo.predict(X_test)
            acc     = accuracy_score(y_test, y_pred)
            report  = classification_report(y_test, y_pred)
            metricas_txt = (
                f"# Metricas do Modelo\n\n"
                f"**Tipo:** Classificacao\n"
                f"**Coluna Alvo:** `{target_col}`\n"
                f"**Accuracy:** {acc:.4f}\n\n"
                f"```\n{report}\n```"
            )
        else:
            modelo = RandomForestRegressor(n_estimators=100, random_state=CONFIG["random_state"])
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            rmse   = mean_squared_error(y_test, y_pred) ** 0.5
            metricas_txt = (
                f"# Metricas do Modelo\n\n"
                f"**Tipo:** Regressao\n"
                f"**Coluna Alvo:** `{target_col}`\n"
                f"**RMSE:** {rmse:.4f}\n"
            )

        with open(CONFIG["metrics_md"], "w", encoding="utf-8") as f:
            f.write(metricas_txt)
        with open(CONFIG["model_pkl"], "wb") as f:
            pickle.dump(modelo, f)

        return (
            f"SUCESSO_ML\n"
            f"Coluna alvo: '{target_col}'\n"
            f"Tipo: {'Classificacao' if is_classificacao else 'Regressao'}\n"
            f"Arquivos gerados: Metricas_Modelo.md e modelo_final.pkl"
        )
    except Exception as e:
        return f"ERRO em treinar_e_salvar_modelo: {e}"


@tool("confirmar_modelo_existe")
def confirmar_modelo_existe(_: str = "") -> str:
    """
    Confirma se modelo_final.pkl foi salvo com sucesso.
    Nao requer parametros.
    """
    path = CONFIG["model_pkl"]
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        return f"OK: modelo_final.pkl existe. Tamanho: {size_kb:.1f} KB"
    return "ERRO: modelo_final.pkl NAO encontrado. Execute treinar_e_salvar_modelo primeiro."


# ── ETAPA 5: README (chamada via .func() — fora do Crew) ─────────────────────

@tool("gerar_readme")
def gerar_readme(_: str = "") -> str:
    """
    Gera o README.md consolidando todo o pipeline no padrao CRISP-DM.
    Nao requer parametros.
    """
    try:
        metricas = ""
        if os.path.exists(CONFIG["metrics_md"]):
            with open(CONFIG["metrics_md"], encoding="utf-8") as f:
                metricas = f.read()

        conteudo = f"""# Auto Data Scientist - Pipeline End-to-End

> Pipeline autonomo de Data Science gerado por agentes CrewAI + Llama 3.1

---

## 1. Entendimento do Negocio

Dataset de triagem de curriculos com ~200k candidatos.
Objetivo: extrair insights e construir um modelo preditivo automaticamente.

**Fonte:** [Kaggle - Resume Screening Dataset]({CONFIG['dataset_url']})

---

## 2. Preparacao dos Dados (Arquitetura Medalhao)

| Camada   | Arquivo              | Descricao                          |
|----------|----------------------|------------------------------------|
| Silver   | df1_silver.parquet   | Dados brutos limpos e padronizados |
| Gold     | df2_gold.parquet     | Features de engenharia adicionadas |
| ML-Ready | df3_ml_ready.parquet | Colunas redundantes/IDs removidos  |

---

## 3. Analise Exploratoria (EDA)

Correlacao multivariada para evitar multicolinearidade:

![Matriz de Correlacao](matriz_correlacao.png)

Estatistica descritiva completa: [Estatistica_Descritiva.md](Estatistica_Descritiva.md)

---

## 4. Modelagem (Machine Learning)

{metricas}

---

## 5. Artefatos Gerados

| Artefato                  | Descricao                   |
|---------------------------|-----------------------------|
| modelo_final.pkl          | Modelo treinado (pickle)    |
| Metricas_Modelo.md        | Metricas de performance     |
| matriz_correlacao.png     | Matriz de correlacao        |
| Estatistica_Descritiva.md | Estatistica descritiva      |

---

## 6. Como Reproduzir

```bash
git clone <url-do-repo>
echo "KAGGLE_USERNAME=seu_usuario" >> .env
echo "KAGGLE_KEY=sua_chave"        >> .env
pip install crewai kagglehub pandas pyarrow python-dotenv scikit-learn matplotlib seaborn tabulate numpy
python ollama_ds_v4.py
```

---
*Gerado automaticamente pelo Auto Data Scientist - CrewAI + Llama 3.1*
"""

        with open(CONFIG["readme_md"], "w", encoding="utf-8") as f:
            f.write(conteudo)

        return "SUCESSO_STORY: README.md gerado com sucesso."
    except Exception as e:
        return f"ERRO em gerar_readme: {e}"


# ==========================================
# 4. AGENTES
# Apenas os 4 agentes analiticos ficam no Crew.
# Storyteller e Git rodam diretamente no __main__ via .func()
# ==========================================

ingestor = Agent(
    role="Engenheiro de Dados Principal",
    goal=(
        "Baixar o dataset do Kaggle e confirmar que df1_silver.parquet existe no disco. "
        "Primeiro chame baixar_e_salvar_silver. "
        "Depois chame confirmar_silver_existe. "
        "Encerre apenas quando confirmar_silver_existe retornar OK."
    ),
    backstory=(
        "Especialista implacavel em ingestao de dados. "
        "Sempre verifica se o arquivo foi salvo antes de declarar sucesso."
    ),
    tools=[baixar_e_salvar_silver, confirmar_silver_existe],
    llm=llm_analitico,
    verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

analista_features = Agent(
    role="Engenheiro de Features e Analista Estatistico Senior",
    goal=(
        "Gerar estatistica descritiva e enriquecer o dataset com novas features. "
        "Primeiro chame gerar_features_e_estatistica. "
        "Depois chame confirmar_gold_existe. "
        "Encerre apenas quando confirmar_gold_existe retornar OK."
    ),
    backstory=(
        "Metodico e analitico. Usa ferramentas dedicadas para cada etapa. "
        "Sempre valida o output antes de declarar a tarefa concluida."
    ),
    tools=[gerar_features_e_estatistica, confirmar_gold_existe],
    llm=llm_analitico,
    verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

analista_eda = Agent(
    role="Cientista de Dados Senior - Especialista em EDA",
    goal=(
        "Gerar visualizacoes e preparar o dataset para Machine Learning. "
        "Primeiro chame gerar_eda_e_ml_ready. "
        "Depois chame confirmar_ml_ready_existe. "
        "Encerre apenas quando confirmar_ml_ready_existe retornar OK."
    ),
    backstory=(
        "Mestre em estatistica visual. Identifica multicolinearidade e "
        "remove variaveis inuteis antes da modelagem."
    ),
    tools=[gerar_eda_e_ml_ready, confirmar_ml_ready_existe],
    llm=llm_analitico,
    verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

cientista_ml = Agent(
    role="Engenheiro de Machine Learning Senior",
    goal=(
        "Treinar o melhor modelo possivel e salvar metricas e artefatos. "
        "Primeiro chame treinar_e_salvar_modelo. "
        "Depois chame confirmar_modelo_existe. "
        "Encerre apenas quando confirmar_modelo_existe retornar OK."
    ),
    backstory=(
        "Focado em performance real. Detecta automaticamente classificacao ou regressao "
        "e salva o modelo pronto para producao."
    ),
    tools=[treinar_e_salvar_modelo, confirmar_modelo_existe],
    llm=llm_analitico,
    verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

# ==========================================
# 5. TASKS
# ==========================================

task_ingestao = Task(
    description=(
        "Baixe o dataset de curriculos do Kaggle e confirme que o arquivo Silver foi salvo.\n\n"
        "Passos OBRIGATORIOS nesta ordem:\n"
        "1. Chame a tool 'baixar_e_salvar_silver' (sem parametros).\n"
        "2. Chame a tool 'confirmar_silver_existe' (sem parametros).\n"
        "3. Se retornar OK, encerre. Se retornar ERRO, repita o passo 1."
    ),
    agent=ingestor,
    expected_output="Confirmacao 'OK' de que df1_silver.parquet existe no disco.",
)

task_estatistica_features = Task(
    description=(
        "Gere estatistica descritiva e features de engenharia a partir do Silver.\n\n"
        "Passos OBRIGATORIOS nesta ordem:\n"
        "1. Chame 'gerar_features_e_estatistica' (sem parametros).\n"
        "2. Chame 'confirmar_gold_existe' (sem parametros).\n"
        "3. Se retornar OK, encerre. Se retornar ERRO, repita o passo 1."
    ),
    agent=analista_features,
    context=[task_ingestao],
    expected_output="Confirmacao 'OK' de que df2_gold.parquet e Estatistica_Descritiva.md existem.",
)

task_eda_graficos = Task(
    description=(
        "Gere a matriz de correlacao e prepare o dataset para ML.\n\n"
        "Passos OBRIGATORIOS nesta ordem:\n"
        "1. Chame 'gerar_eda_e_ml_ready' (sem parametros).\n"
        "2. Chame 'confirmar_ml_ready_existe' (sem parametros).\n"
        "3. Se retornar OK, encerre. Se retornar ERRO, repita o passo 1."
    ),
    agent=analista_eda,
    context=[task_estatistica_features],
    expected_output="Confirmacao 'OK' de que matriz_correlacao.png e df3_ml_ready.parquet existem.",
)

task_modelagem = Task(
    description=(
        "Treine o modelo de Machine Learning e salve os artefatos.\n\n"
        "Passos OBRIGATORIOS nesta ordem:\n"
        "1. Chame 'treinar_e_salvar_modelo' (sem parametros).\n"
        "2. Chame 'confirmar_modelo_existe' (sem parametros).\n"
        "3. Se retornar OK, encerre. Se retornar ERRO, repita o passo 1."
    ),
    agent=cientista_ml,
    context=[task_eda_graficos],
    expected_output="Confirmacao 'OK' de que modelo_final.pkl e Metricas_Modelo.md existem.",
)

# ==========================================
# 6. ORQUESTRACAO — apenas agentes analiticos
# Storyteller e Git rodam fora do Crew (Python puro)
# ==========================================

squad_ds = Crew(
    agents=[ingestor, analista_features, analista_eda, cientista_ml],
    tasks=[task_ingestao, task_estatistica_features, task_eda_graficos, task_modelagem],
    process=Process.sequential,
    verbose=True,
)

# ==========================================
# 7. POS-PIPELINE: README e Git
#
# FIX v4: Chamamos gerar_readme.func("") em vez de gerar_readme("")
# O decorador @tool transforma a funcao em objeto Tool — nao e diretamente
# callable. O atributo .func acessa a funcao Python original por baixo do wrapper.
# ==========================================

def executar_pos_pipeline():
    """README e Git executados diretamente em Python — sem LLM, sem alucinacao."""

    # ── README ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GERANDO README.md")
    print("=" * 60)
    resultado_readme = gerar_readme.func("")   # <-- .func() acessa a funcao real
    print(resultado_readme)

    # ── Git ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INICIANDO VERSIONAMENTO GIT")
    print("=" * 60)

    def git(cmd):
        print(f"\n> {cmd}")
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=60
            )
            saida  = (r.stdout or r.stderr).strip()
            status = "[OK]" if r.returncode == 0 else "[FALHA]"
            print(f"{status} {saida[:200]}")
            return r.returncode == 0
        except Exception as e:
            print(f"[ERRO] {e}")
            return False

    # Verifica remote antes de qualquer coisa
    if not git("git remote -v"):
        print("\nREMOTE NAO CONFIGURADO.")
        print("Execute: git remote add origin <url-do-seu-repo>")
        return

    arquivos = [
        "README.md",
        "matriz_correlacao.png",
        "Estatistica_Descritiva.md",
        "Metricas_Modelo.md",
        "modelo_final.pkl",
        "df1_silver.parquet",
        "df2_gold.parquet",
        "df3_ml_ready.parquet",
        "ollama_ds_v4.py",
    ]
    for arq in arquivos:
        git(f"git add {arq}")

    git('git commit -m "feat: pipeline end-to-end com ML, EDA e documentacao"')
    git("git push origin main")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO")
    print("=" * 60)


# ==========================================
# 8. PONTO DE ENTRADA
# ==========================================

if __name__ == "__main__":
    logger.info("Iniciando o Auto Data Scientist v4 (CrewAI + Llama 3.1)...")

    if not os.path.exists(".env"):
        logger.warning(
            "Arquivo .env nao encontrado. "
            "Crie-o com KAGGLE_USERNAME e KAGGLE_KEY antes de executar."
        )

    # Crew: ingestao, features, EDA e ML
    resultado = squad_ds.kickoff()

    print("\n" + "=" * 60)
    print("RESULTADO FINAL DO PIPELINE DE DADOS")
    print("=" * 60)
    print(resultado)

    # README e Git: Python puro, sem LLM
    executar_pos_pipeline()

    logger.info("Pipeline encerrado.")