"""
Auto Data Scientist v7 — True SOTA Multi-Agent Pipeline
========================================================

Architecture:
  - CrewAI Layer: stable, sequential, 1 tool per agent (lesson from v5)
  - Real intelligence: Claude 3.5 Sonnet called INSIDE tools to reason
  - LLM writes analysis code dynamically (not pre-written)
  - LLM interprets results and makes real decisions
  - LLM diagnoses errors and suggests corrections (real self-healing)
  - LLM chooses feature strategy based on the data
  - LLM decides which model to use and why
  - LLM writes a full Streamlit stakeholder app tailored to the dataset

Why this is different:
  - v4/v5/v6: LLM decides which function to call. Python does the work.
  - v7: LLM decides AND ALSO does the intelligent work inside each step.

Pipeline steps:
  1. Ingestor        → download_and_save_silver
  2. Analyst         → analyze_data_with_ai
  3. Feature Eng.    → generate_features_with_ai_strategy
  4. EDA Analyst     → generate_eda_and_ml_ready
  5. ML Scientist    → train_and_save_model
  6. Deployer        → deploy_streamlit_app
                         ├── df4_predictions.parquet (all original cols + prediction)
                         ├── streamlit_app.py        (stakeholder dashboard)
                         └── requirements.txt
  7. Notebook Writer → generate_analysis_notebook
                         └── analysis_notebook.ipynb (full pipeline story, renders on GitHub)

Dependencies:
    pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic
    pip install scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm
    pip install streamlit plotly nbformat

Environment variables (.env):
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_key_here
    ANTHROPIC_API_KEY=sk-ant-...

Optional:
    Create business_context.txt with a description of the business problem.
"""

# ==========================================
# 0. IMPORTS
# ==========================================
import os, sys, json, logging, subprocess, pickle
import traceback, textwrap, io, contextlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
import anthropic
optuna.logging.set_verbosity(optuna.logging.WARNING)

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, KFold,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    classification_report, r2_score,
)

load_dotenv()

# ==========================================
# LOGGING UTF-8
# ==========================================
_utf8_handler = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
)
_utf8_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        _utf8_handler,
    ],
)
logger = logging.getLogger("AutoDS")

# ==========================================
# 1. CONFIGURATION
# ==========================================
_BASE_DIR = os.getcwd()

CONFIG = {
    "dataset_slug": "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
    "dataset_url":  "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",

    "silver_path":   os.path.join(_BASE_DIR, "df1_silver.parquet"),
    "gold_path":     os.path.join(_BASE_DIR, "df2_gold.parquet"),
    "ml_ready_path": os.path.join(_BASE_DIR, "df3_ml_ready.parquet"),
    "quality_md":    os.path.join(_BASE_DIR, "Quality_Report.md"),
    "analysis_md":   os.path.join(_BASE_DIR, "Intelligent_Analysis.md"),
    "stats_md":      os.path.join(_BASE_DIR, "Descriptive_Statistics.md"),
    "target_json":   os.path.join(_BASE_DIR, "target_config.json"),
    "strategy_json": os.path.join(_BASE_DIR, "feature_strategy.json"),
    "corr_png":      os.path.join(_BASE_DIR, "correlation_matrix.png"),
    "metrics_md":    os.path.join(_BASE_DIR, "Model_Metrics.md"),
    "eval_md":       os.path.join(_BASE_DIR, "Model_Evaluation.md"),
    "model_pkl":       os.path.join(_BASE_DIR, "final_model.pkl"),
    "readme_md":       os.path.join(_BASE_DIR, "README.md"),
    "business_ctx":    os.path.join(_BASE_DIR, "business_context.txt"),
    "predictions_path": os.path.join(_BASE_DIR, "df4_predictions.parquet"),
    "streamlit_app":    os.path.join(_BASE_DIR, "streamlit_app.py"),
    "requirements_txt": os.path.join(_BASE_DIR, "requirements.txt"),
    "notebook_path":    os.path.join(_BASE_DIR, "analysis_notebook.ipynb"),

    "test_size":       0.2,
    "random_state":    42,
    "cv_folds":        5,
    "optuna_trials":   20,
    "score_threshold": 0.70,
    "max_iter":        5,
    "max_retry_limit": 2,
}

# ==========================================
# 2. LLMs
# ==========================================

# LLM for CrewAI (orchestrates the agents)
llm_agent = LLM(
    model="anthropic/claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.0,
)

# Direct Anthropic client (used INSIDE tools for real reasoning)
_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def _ask_claude(prompt: str, max_tokens: int = 2000) -> str:
    """Calls Claude directly for intelligent reasoning inside tools."""
    try:
        msg = _claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.error(f"[Claude] Error in direct call: {e}")
        return f"CLAUDE_ERROR: {e}"

# ==========================================
# 3. HELPERS
# ==========================================

def _read_ctx() -> str:
    p = CONFIG["business_ctx"]
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return ""

def _detect_type(y: pd.Series) -> str:
    n, nu = len(y), y.nunique()
    if y.dtype == "object":                return "classification"
    if nu <= 15:                           return "classification"
    if nu <= 30 and "int" in str(y.dtype): return "classification"
    if nu / n < 0.05:                      return "classification"
    return "regression"

def _safe_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

def _execute_code(code: str, df: pd.DataFrame) -> tuple[str, bool]:
    """Executes Python code and returns (output, success)."""
    ns = {"pd": pd, "np": np, "df": df.copy(),
          "plt": plt, "sns": sns, "os": os,
          "_BASE_DIR": _BASE_DIR, "json": json}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(textwrap.dedent(code), ns)
        return output.getvalue() or "Executed without output.", True
    except Exception as e:
        return f"{type(e).__name__}: {e}", False

# ==========================================
# 4. TOOLS — ONE PER AGENT, REAL INTELLIGENCE INSIDE
# ==========================================

# ── STEP 1: Ingestion ─────────────────────────────────────────────────────────

@tool("download_and_save_silver")
def download_and_save_silver(_: str = "") -> str:
    """
    Downloads the dataset from Kaggle, standardizes columns, and saves df1_silver.parquet.
    Returns INGESTION_SUCCESS or ERROR. No parameters.
    """
    import kagglehub
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key  = os.getenv("KAGGLE_KEY")
    if not kaggle_user or not kaggle_key:
        return "ERROR: KAGGLE_USERNAME/KAGGLE_KEY not found in .env."
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"]      = kaggle_key
    try:
        path = kagglehub.dataset_download(CONFIG["dataset_slug"])
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csvs:
            return f"ERROR: No CSV found in {path}"
        csv_path = os.path.join(path, csvs[0])
        # Try multiple encodings — many Kaggle CSVs are not UTF-8
        for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                logger.info(f"[Ingestor] CSV read with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        else:
            return "ERROR: Could not decode CSV with any known encoding."
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(" ", "_")
                      .str.replace(r"[^a-z0-9_]", "", regex=True))
        df.to_parquet(CONFIG["silver_path"], index=False)
        logger.info(f"[Ingestor] Silver saved: {df.shape}")
        return (f"INGESTION_SUCCESS\n"
                f"Shape: {df.shape}\n"
                f"Columns: {list(df.columns)}\n"
                f"File: df1_silver.parquet")
    except Exception as e:
        return f"INGESTION_ERROR: {e}"


# ── STEP 2: Quality + Intelligent Analysis ────────────────────────────────────

@tool("analyze_data_with_ai")
def analyze_data_with_ai(_: str = "") -> str:
    """
    Reads df1_silver.parquet. Applies intelligent imputation. Passes a full
    dataset summary to Claude to reason about: data quality, suspicious columns,
    cleaning recommendations, and business insights.
    Claude also writes and executes custom Python analysis code.
    Saves Quality_Report.md and Intelligent_Analysis.md.
    Returns ANALYSIS_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["silver_path"]):
            return "ERROR: df1_silver.parquet does not exist."

        df  = pd.read_parquet(CONFIG["silver_path"])
        n, p = df.shape
        ctx = _read_ctx()

        # ── Imputation ─────────────────────────────────────────────────────────
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        imps = []

        if num_cols and df[num_cols].isnull().any().any():
            try:
                # Only impute columns that have at least one non-null value
                num_cols_valid = [c for c in num_cols
                                  if df[c].notna().any()]
                if num_cols_valid:
                    df[num_cols_valid] = KNNImputer(n_neighbors=5).fit_transform(
                        df[num_cols_valid])
                    imps.append("KNN imputer applied to numeric columns.")
            except Exception:
                num_cols_valid = [c for c in num_cols if df[c].notna().any()]
                if num_cols_valid:
                    df[num_cols_valid] = SimpleImputer(strategy="median").fit_transform(
                        df[num_cols_valid])
                    imps.append("Median imputer (fallback) applied.")
        for c in cat_cols:
            if df[c].isnull().any():
                df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "MISSING",
                             inplace=True)
                imps.append(f"Mode applied to '{c}'.")
        if imps:
            df.to_parquet(CONFIG["silver_path"], index=False)

        # ── Summary for Claude to reason about ────────────────────────────────
        col_summary = {}
        for col in df.columns:
            s = df[col]
            col_summary[col] = {
                "dtype": str(s.dtype),
                "nunique": int(s.nunique()),
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "sample": s.dropna().head(5).tolist(),
            }
            if s.dtype in ["float64", "int64"]:
                col_summary[col].update({
                    "mean": round(float(s.mean()), 4),
                    "std":  round(float(s.std()), 4),
                    "min":  round(float(s.min()), 4),
                    "max":  round(float(s.max()), 4),
                    "skew": round(float(s.skew()), 4),
                })

        # ── Claude analyzes the dataset ────────────────────────────────────────
        prompt_analysis = f"""You are a senior Data Scientist analyzing a dataset.

BUSINESS CONTEXT: {ctx or 'Resume screening dataset with 200k candidates.'}

DATASET:
- Shape: {n} rows x {p} columns
- Columns and statistics:
{json.dumps(col_summary, indent=2, default=_safe_json)}

IMPUTATIONS ALREADY APPLIED: {imps}

Your task:
1. Identify which column is likely the TARGET (response variable) and justify.
2. Identify problematic columns (leakage, high cardinality, constants).
3. List the top-5 most important insights about this dataset.
4. Write Python code (using df, pd, np, plt, os, json, _BASE_DIR) that:
   a) Calculates statistics by group for the most interesting column
   b) Shows the distribution of the likely target
   c) Calculates correlations with the target
   d) Saves a plot to os.path.join(_BASE_DIR, 'intelligent_analysis.png')
5. Recommend a feature engineering strategy for this specific dataset.

Respond in JSON with this exact structure:
{{
  "likely_target": "column_name",
  "target_justification": "...",
  "problematic_columns": ["col1", "col2"],
  "insights": ["insight1", "insight2", "insight3", "insight4", "insight5"],
  "analysis_code": "python code here as string",
  "feature_strategy": "description of recommended strategy"
}}

Respond ONLY with the JSON, no text before or after."""

        raw_response = _ask_claude(prompt_analysis, max_tokens=3000)

        # Parse JSON from response
        try:
            clean_response = raw_response
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0]
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0]
            analysis = json.loads(clean_response.strip())
        except json.JSONDecodeError:
            logger.warning("[Analysis] Claude did not return valid JSON. Using fallback.")
            analysis = {
                "likely_target": "hired",
                "target_justification": "Column name indicates an outcome.",
                "problematic_columns": [],
                "insights": ["Resume dataset with numeric and categorical variables."],
                "analysis_code": "print('Basic analysis executed.')",
                "feature_strategy": "Create ratio and interaction features between numeric variables.",
            }

        # ── Execute the code Claude wrote ──────────────────────────────────────
        code = analysis.get("analysis_code", "print('no code')")
        code_output, success = _execute_code(code, df)

        # Self-healing: if code failed, Claude tries to fix it
        if not success:
            logger.warning(f"[Analysis] Code failed: {code_output}. Claude will fix it.")
            prompt_fix = f"""The following Python code failed with error: {code_output}

Original code:
```python
{code}
```

Fix the code. Available variables: df (DataFrame), pd, np, plt, os, json, _BASE_DIR.
Respond ONLY with the corrected Python code, no explanations, no markdown."""

            fixed_code = _ask_claude(prompt_fix, max_tokens=1500)
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0]
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0]

            code_output, success = _execute_code(fixed_code, df)
            code = fixed_code
            logger.info(f"[Analysis] Self-healing: {'success' if success else 'failed again'}")

        # ── Save quality report ────────────────────────────────────────────────
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0]
        outliers = {}
        for c in num_cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            cnt = ((df[c] < q1-1.5*iqr) | (df[c] > q3+1.5*iqr)).sum()
            if cnt > 0:
                outliers[c] = int(cnt)

        quality_md = f"""# Quality Report — AI-Powered Analysis

**Context:** {ctx or 'Resume screening dataset.'}
**Shape:** {n} x {p}

## Applied Imputation
{chr(10).join(f'- {i}' for i in imps) if imps else '- No imputation required.'}

## Detected Outliers (IQR)
{json.dumps(outliers, indent=2) if outliers else 'No significant outliers.'}

## Intelligent Analysis by Claude

### Identified Target
**Column:** `{analysis['likely_target']}`
**Justification:** {analysis['target_justification']}

### Problematic Columns
{analysis.get('problematic_columns', [])}

### Top Dataset Insights
{chr(10).join(f'{i+1}. {ins}' for i, ins in enumerate(analysis.get('insights', [])))}

### Recommended Feature Engineering Strategy
{analysis.get('feature_strategy', '')}

### Analysis Execution Output
```
{code_output[:2000]}
```

---
*Analysis generated by Claude 3.5 Sonnet*
"""
        with open(CONFIG["quality_md"], "w", encoding="utf-8") as f:
            f.write(quality_md)

        # Save analysis JSON for later use
        with open(CONFIG["analysis_md"], "w", encoding="utf-8") as f:
            f.write(f"# Intelligent Analysis\n\n```json\n{json.dumps(analysis, indent=2, default=_safe_json)}\n```")

        # Save target suggested by AI for the next agent to use
        with open(CONFIG["target_json"], "w", encoding="utf-8") as f:
            json.dump({
                "target_col": analysis["likely_target"],
                "problem_type": _detect_type(df[analysis["likely_target"]])
                    if analysis["likely_target"] in df.columns else "classification",
                "ai_justification": analysis["target_justification"],
                "ai_feature_strategy": analysis.get("feature_strategy", ""),
                "ai_insights": analysis.get("insights", []),
            }, f, indent=2, default=_safe_json)

        return (f"ANALYSIS_SUCCESS\n"
                f"Target identified by AI: '{analysis['likely_target']}'\n"
                f"Insights generated: {len(analysis.get('insights', []))}\n"
                f"Code executed: {'yes' if success else 'with fallback'}\n"
                f"Self-healing activated: {not success}\n"
                f"Files: Quality_Report.md, Intelligent_Analysis.md, target_config.json")
    except Exception as e:
        return f"ANALYSIS_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 3: Intelligent Feature Engineering ───────────────────────────────────

@tool("generate_features_with_ai_strategy")
def generate_features_with_ai_strategy(_: str = "") -> str:
    """
    Reads df1_silver.parquet and the AI-defined feature strategy.
    Claude decides which features to create based on real data.
    Creates standard features + AI-customized features.
    Saves df2_gold.parquet. Returns FEATURES_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["silver_path"]):
            return "ERROR: df1_silver.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["silver_path"])
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col = cfg["target_col"]
        strategy   = cfg.get("ai_feature_strategy", "")

        df.describe(include="all").to_markdown(CONFIG["stats_md"])

        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != target_col]

        # Standard features (always created)
        standard_feats = []
        if len(num_cols) >= 2:
            c0, c1 = num_cols[0], num_cols[1]
            df["feat_ratio"]   = df[c0] / (df[c1] + 1e-9)
            df["feat_sum"]     = df[c0] + df[c1]
            df["feat_product"] = df[c0] * df[c1]
            df["feat_diff"]    = df[c0] - df[c1]
            standard_feats    += ["feat_ratio", "feat_sum", "feat_product", "feat_diff"]

        for c in num_cols[:8]:
            col_d = df[c].dropna()
            if len(col_d) > 0 and col_d.min() > 0 and abs(col_d.skew()) > 1.0:
                df[f"log_{c}"] = np.log1p(df[c])
                standard_feats.append(f"log_{c}")

        if len(num_cols) >= 3:
            df["feat_interact"] = df[num_cols[0]] * df[num_cols[2]]
            standard_feats.append("feat_interact")

        if len(num_cols) >= 2:
            df[f"sq_{num_cols[0]}"] = df[num_cols[0]] ** 2
            df[f"sq_{num_cols[1]}"] = df[num_cols[1]] ** 2
            standard_feats += [f"sq_{num_cols[0]}", f"sq_{num_cols[1]}"]

        # ── Claude decides custom features based on real data ──────────────────
        col_stats = {c: {"mean": round(float(df[c].mean()), 3),
                         "std": round(float(df[c].std()), 3),
                         "corr_target": round(float(df[c].corr(df[target_col]))
                                              if target_col in df.select_dtypes(include="number").columns
                                              else 0.0, 3)}
                     for c in num_cols[:10]}

        prompt_features = f"""You are an expert Feature Engineer.

Dataset: {df.shape[0]} rows, {df.shape[1]} columns
Target: '{target_col}'
Previously suggested strategy: {strategy}

Numeric columns and their correlations with the target:
{json.dumps(col_stats, indent=2)}

Features already created: {standard_feats}

Create 3-5 additional features SPECIFIC to this dataset that may improve
target prediction. Consider:
- Non-linear combinations that make sense in context
- Ratios between correlated columns
- Features of magnitude or relative scale

Respond ONLY with valid Python code. Available variables: df (DataFrame with all columns), np, pd.
Do not use plt. Do not save files. Only modify df by adding new columns.
Example: df['new_feat'] = df['col_a'] / (df['col_b'] + 1)"""

        feature_code = _ask_claude(prompt_features, max_tokens=1000)

        # Clean markdown if present
        if "```python" in feature_code:
            feature_code = feature_code.split("```python")[1].split("```")[0]
        elif "```" in feature_code:
            feature_code = feature_code.split("```")[1].split("```")[0]

        # Execute custom features
        ai_feats = []
        cols_before = set(df.columns)
        feat_output, feat_success = _execute_code(feature_code, df)

        if feat_success:
            # Apply new columns to the real df
            ns = {"pd": pd, "np": np, "df": df}
            try:
                exec(textwrap.dedent(feature_code), ns)
                df = ns["df"]
                ai_feats = [c for c in df.columns if c not in cols_before]
                logger.info(f"[AI Features] Created: {ai_feats}")
            except Exception as ex:
                logger.warning(f"[AI Features] Failed to apply: {ex}")
        else:
            logger.warning(f"[AI Features] Code failed: {feat_output}")

        df.to_parquet(CONFIG["gold_path"], index=False)

        # Save used strategy
        with open(CONFIG["strategy_json"], "w", encoding="utf-8") as f:
            json.dump({
                "standard_features": standard_feats,
                "ai_features": ai_feats,
                "ai_code": feature_code,
                "ai_success": feat_success,
            }, f, indent=2)

        return (f"FEATURES_SUCCESS\n"
                f"Standard features: {len(standard_feats)}\n"
                f"AI-generated features: {ai_feats}\n"
                f"Gold shape: {df.shape}\n"
                f"File: df2_gold.parquet")
    except Exception as e:
        return f"FEATURES_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 4: EDA ───────────────────────────────────────────────────────────────

@tool("generate_eda_and_ml_ready")
def generate_eda_and_ml_ready(_: str = "") -> str:
    """
    Reads df2_gold.parquet. Generates 6 visualizations.
    Removes redundancies. Saves df3_ml_ready.parquet.
    Returns EDA_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["gold_path"]):
            return "ERROR: df2_gold.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["gold_path"])
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col = cfg["target_col"]
        num_df     = df.select_dtypes(include="number")

        # G1: Correlation matrix
        plt.figure(figsize=(13, 10))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.4, annot_kws={"size": 7})
        plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(CONFIG["corr_png"], dpi=150); plt.close()

        base_cols = [c for c in num_df.columns
                     if not any(c.startswith(p)
                                for p in ["feat_", "log_", "sq_"])][:8]

        # G2 + G3: Distributions and Boxplots
        if base_cols:
            n_c = min(len(base_cols), 4)
            n_r = (len(base_cols) + n_c - 1) // n_c
            for fname, plot_fn in [
                ("distributions.png",
                 lambda ax, d: ax.hist(d, bins=40, color="#4C72B0",
                                       edgecolor="white", alpha=0.85)),
                ("boxplots.png",
                 lambda ax, d: ax.boxplot(d, patch_artist=True,
                                          boxprops=dict(facecolor="#4C72B0",
                                                        alpha=0.7))),
            ]:
                fig, axes = plt.subplots(n_r, n_c, figsize=(5*n_c, 4*n_r))
                axes = np.array(axes).flatten()
                for i, col in enumerate(base_cols):
                    plot_fn(axes[i], num_df[col].dropna())
                    axes[i].set_title(col, fontsize=10, fontweight="bold")
                    axes[i].grid(axis="y", alpha=0.3)
                for j in range(len(base_cols), len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                plt.savefig(os.path.join(_BASE_DIR, fname), dpi=150); plt.close()

        # G4: Categorical features
        cat_cols = [c for c in df.select_dtypes(include="object").columns
                    if df[c].nunique() <= 20 and c != target_col][:4]
        if cat_cols:
            fig, axes = plt.subplots(1, len(cat_cols),
                                     figsize=(6*len(cat_cols), 5))
            if len(cat_cols) == 1: axes = [axes]
            for i, col in enumerate(cat_cols):
                vc = df[col].value_counts().head(15)
                axes[i].barh(vc.index.astype(str), vc.values,
                             color="#E05C5C", alpha=0.8)
                axes[i].set_title(col, fontsize=11, fontweight="bold")
                axes[i].invert_yaxis(); axes[i].grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "categoricals.png"), dpi=150)
            plt.close()

        # G5: Target distribution
        if target_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            if cfg["problem_type"] == "classification":
                vc = df[target_col].value_counts().head(20)
                ax.bar(vc.index.astype(str), vc.values,
                       color="#2CA02C", alpha=0.85)
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.hist(df[target_col].dropna(), bins=40,
                        color="#2CA02C", alpha=0.85)
            ax.set_title(f"Target Distribution: {target_col}",
                         fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "target_dist.png"), dpi=150)
            plt.close()

        # G6: Dataset sample
        sample = df.head(10)
        fig, ax = plt.subplots(figsize=(18, 3)); ax.axis("off")
        tb = ax.table(cellText=sample.values, colLabels=sample.columns,
                      cellLoc="center", loc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(7)
        tb.auto_set_column_width(col=list(range(len(sample.columns))))
        for (row, col), cell in tb.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4C72B0")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f0f4ff")
        plt.title("Dataset Sample", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(_BASE_DIR, "dataset_sample.png"),
                    dpi=150, bbox_inches="tight"); plt.close()

        # Remove redundancies
        corr_abs = num_df.corr().abs()
        upper    = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
        redundant_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
        id_cols        = [c for c in df.columns
                          if "id" in c.lower() and c != target_col]
        cols_to_remove = list(set(redundant_cols + id_cols) - {target_col})
        df_filtered    = df.drop(columns=cols_to_remove, errors="ignore")
        df_filtered.to_parquet(CONFIG["ml_ready_path"], index=False)

        return (f"EDA_SUCCESS\n"
                f"Removed columns: {cols_to_remove}\n"
                f"ML-Ready shape: {df_filtered.shape}\n"
                f"Charts: correlation, distributions, boxplots, "
                f"categoricals, target_dist, sample\n"
                f"File: df3_ml_ready.parquet")
    except Exception as e:
        return f"EDA_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 5: ML with Optuna + Stacking + AI Interpretation ─────────────────────

@tool("train_and_save_model")
def train_and_save_model(_: str = "") -> str:
    """
    Reads df3_ml_ready.parquet. Runs model competition with CV.
    Applies Optuna to the top-3. Attempts Stacking. Saves best model.
    Claude interprets results and writes a narrative diagnostic.
    Saves Model_Metrics.md. Returns ML_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["ml_ready_path"]):
            return "ERROR: df3_ml_ready.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["ml_ready_path"]).dropna()
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col   = cfg["target_col"]
        problem_type = cfg["problem_type"]
        feature_cols = [c for c in df.columns if c != target_col]

        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df[target_col].copy()

        le = None
        if y.dtype == "object" or problem_type == "classification":
            le           = LabelEncoder()
            y            = le.fit_transform(y.astype(str))
            problem_type = "classification"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y if problem_type == "classification" else None,
        )
        cv = (StratifiedKFold if problem_type == "classification" else KFold)(
            n_splits=CONFIG["cv_folds"], shuffle=True,
            random_state=CONFIG["random_state"],
        )
        metric = "accuracy" if problem_type == "classification" else "r2"

        # Model catalog
        if problem_type == "classification":
            MODELS = {
                "RandomForest":       RandomForestClassifier(n_estimators=100,
                                          random_state=CONFIG["random_state"]),
                "GradientBoosting":   GradientBoostingClassifier(n_estimators=100,
                                          random_state=CONFIG["random_state"]),
                "ExtraTrees":         ExtraTreesClassifier(n_estimators=100,
                                          random_state=CONFIG["random_state"]),
                "LogisticRegression": LogisticRegression(max_iter=1000,
                                          random_state=CONFIG["random_state"]),
            }
        else:
            MODELS = {
                "RandomForest":     RandomForestRegressor(n_estimators=100,
                                        random_state=CONFIG["random_state"]),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100,
                                        random_state=CONFIG["random_state"]),
                "ExtraTrees":       ExtraTreesRegressor(n_estimators=100,
                                        random_state=CONFIG["random_state"]),
                "Ridge":            Ridge(),
            }

        try:
            from xgboost import XGBClassifier, XGBRegressor
            MODELS["XGBoost"] = (
                XGBClassifier(n_estimators=100, use_label_encoder=False,
                              eval_metric="logloss", verbosity=0,
                              random_state=CONFIG["random_state"])
                if problem_type == "classification" else
                XGBRegressor(n_estimators=100, verbosity=0,
                             random_state=CONFIG["random_state"])
            )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            MODELS["LightGBM"] = (
                LGBMClassifier(n_estimators=100, verbose=-1,
                               random_state=CONFIG["random_state"])
                if problem_type == "classification" else
                LGBMRegressor(n_estimators=100, verbose=-1,
                              random_state=CONFIG["random_state"])
            )
        except ImportError:
            pass

        # CV on all models
        cv_results = {}
        for name, model in MODELS.items():
            try:
                sc = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring=metric, n_jobs=-1)
                cv_results[name] = {"mean": float(sc.mean()), "std": float(sc.std())}
                logger.info(f"[ML] {name}: {sc.mean():.4f} ± {sc.std():.4f}")
            except Exception as ex:
                logger.warning(f"[ML] {name} failed: {ex}")
                cv_results[name] = {"mean": -9999.0, "std": 0.0}

        # Optuna on top-3
        top3 = sorted(cv_results, key=lambda k: cv_results[k]["mean"],
                      reverse=True)[:3]
        optuna_results = {}

        for name in top3:
            def _obj(trial, _n=name):
                if _n in ["RandomForest", "ExtraTrees"]:
                    p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                         "max_depth":    trial.suggest_int("max_depth", 3, 20),
                         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)}
                elif _n == "GradientBoosting":
                    p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                         "max_depth":     trial.suggest_int("max_depth", 2, 8),
                         "subsample":     trial.suggest_float("subsample", 0.5, 1.0)}
                elif _n == "LogisticRegression":
                    p = {"C": trial.suggest_float("C", 0.001, 100, log=True),
                         "max_iter": 1000}
                elif _n == "Ridge":
                    p = {"alpha": trial.suggest_float("alpha", 0.001, 100, log=True)}
                elif _n == "XGBoost":
                    p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                         "max_depth":     trial.suggest_int("max_depth", 2, 8),
                         "subsample":     trial.suggest_float("subsample", 0.5, 1.0)}
                elif _n == "LightGBM":
                    p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                         "num_leaves":    trial.suggest_int("num_leaves", 20, 150)}
                else:
                    p = {}
                try:
                    cls = type(MODELS[_n])
                    kw  = {**p}
                    if "random_state" in cls.__init__.__code__.co_varnames:
                        kw["random_state"] = CONFIG["random_state"]
                    m  = cls(**kw)
                    sc = cross_val_score(m, X_train, y_train, cv=3,
                                         scoring=metric, n_jobs=-1)
                    return float(sc.mean())
                except Exception:
                    return -9999.0

            study = optuna.create_study(direction="maximize")
            study.optimize(_obj, n_trials=CONFIG["optuna_trials"],
                           show_progress_bar=False)

            bp  = study.best_params
            cls = type(MODELS[name])
            kw  = {**bp}
            if "random_state" in cls.__init__.__code__.co_varnames:
                kw["random_state"] = CONFIG["random_state"]
            try:
                m_tuned = cls(**kw)
            except Exception:
                m_tuned = MODELS[name]

            sc_t = cross_val_score(m_tuned, X_train, y_train, cv=cv,
                                   scoring=metric, n_jobs=-1)
            optuna_results[name] = {
                "model": m_tuned,
                "best_params": bp,
                "mean": float(sc_t.mean()),
                "std":  float(sc_t.std()),
            }
            logger.info(f"[Optuna] {name}: {sc_t.mean():.4f} ± {sc_t.std():.4f}")

        # Stacking
        estimators = [(n, optuna_results[n]["model"]) for n in top3]
        try:
            stacker = (StackingClassifier(
                           estimators=estimators,
                           final_estimator=LogisticRegression(max_iter=1000),
                           cv=3, n_jobs=-1)
                       if problem_type == "classification" else
                       StackingRegressor(
                           estimators=estimators,
                           final_estimator=Ridge(),
                           cv=3, n_jobs=-1))
            sc_s = cross_val_score(stacker, X_train, y_train, cv=cv,
                                   scoring=metric, n_jobs=-1)
            optuna_results["Stacking"] = {
                "model": stacker,
                "best_params": {"base": [n for n, _ in estimators]},
                "mean": float(sc_s.mean()),
                "std":  float(sc_s.std()),
            }
            logger.info(f"[Stacking] {sc_s.mean():.4f} ± {sc_s.std():.4f}")
        except Exception as se:
            logger.warning(f"[Stacking] Failed: {se}")

        # Best final model
        best = max(optuna_results, key=lambda k: optuna_results[k]["mean"])
        final_model = optuna_results[best]["model"]
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        test_score = (accuracy_score(y_test, y_pred)
                      if problem_type == "classification"
                      else r2_score(y_test, y_pred))

        # Comparison table
        comp = {**{n: {"mean": v["mean"], "std": v["std"]}
                   for n, v in cv_results.items()},
                **{f"{n}_Optuna": {"mean": v["mean"], "std": v["std"]}
                   for n, v in optuna_results.items()}}
        comp_df = pd.DataFrame(comp).T.sort_values("mean", ascending=False).round(4)

        # ── Claude interprets the results ──────────────────────────────────────
        ai_insights = cfg.get("ai_insights", [])
        prompt_interp = f"""You are a senior Data Scientist interpreting model results.

CONTEXT: {_read_ctx() or 'Resume screening dataset.'}
TARGET: '{target_col}' ({problem_type})
DATASET INSIGHTS: {ai_insights}

MODEL COMPETITION RESULTS:
{comp_df.to_string()}

SELECTED MODEL: {best}
{metric.upper()} ON TEST SET: {test_score:.4f}

Write a narrative interpretation (3-4 paragraphs) explaining:
1. Why the model {best} was the best choice for this problem
2. What the score {test_score:.4f} means in a business context
3. Points of attention or model limitations
4. Practical recommendations for production deployment

Be specific and technical, but also practical."""

        narrative = _ask_claude(prompt_interp, max_tokens=1000)

        # Save pkl
        with open(CONFIG["model_pkl"], "wb") as f:
            pickle.dump({
                "model": final_model, "label_encoder": le,
                "features": list(X.columns), "target": target_col,
                "type": problem_type, "name": best,
                "test_score": test_score,
                "optuna_params": optuna_results[best].get("best_params", {}),
            }, f)

        # Metrics MD with AI narrative
        lines = [f"# Model Metrics\n\n",
                 f"**Type:** {problem_type} | **Target:** `{target_col}`\n\n",
                 f"## Model Comparison\n\n",
                 comp_df.to_markdown() + "\n\n",
                 f"**Selected model:** `{best}`\n\n",
                 f"**{metric.upper()} (test):** {test_score:.4f}\n\n"]

        if problem_type == "classification":
            lines.append(f"```\n{classification_report(y_test, y_pred)}\n```\n\n")
        else:
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            lines.append(f"**RMSE:** {rmse:.4f} | **R²:** {test_score:.4f}\n\n")

        lines.append(f"## AI Interpretation\n\n{narrative}\n")

        with open(CONFIG["metrics_md"], "w", encoding="utf-8") as f:
            f.write("".join(lines))

        # Charts
        comp_plot = comp_df.sort_values("mean")
        fig, ax = plt.subplots(figsize=(12, max(4, len(comp_plot)*0.4)))
        colors = ["#2CA02C" if n == best else
                  ("#FF7F0E" if "Optuna" in n or n == "Stacking" else "#4C72B0")
                  for n in comp_plot.index]
        bars = ax.barh(comp_plot.index, comp_plot["mean"],
                       xerr=comp_plot["std"], color=colors, alpha=0.85, capsize=4)
        ax.bar_label(bars, fmt="%.4f", padding=6, fontsize=9)
        ax.set_xlabel(f"{metric.upper()} CV", fontsize=11)
        ax.set_title("Model Comparison — Baseline vs Optuna vs Stacking",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_BASE_DIR, "model_comparison.png"), dpi=150)
        plt.close()

        if hasattr(final_model, "feature_importances_"):
            imp = (pd.Series(final_model.feature_importances_, index=X.columns)
                   .sort_values(ascending=True).tail(15))
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(imp.index, imp.values, color="#4C72B0", alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
            ax.set_title(f"Top 15 Features — {best}\n(Target: {target_col})",
                         fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "feature_importance.png"), dpi=150)
            plt.close()

        return (f"ML_SUCCESS\n"
                f"Model: '{best}'\n"
                f"{metric.upper()} test: {test_score:.4f}\n"
                f"Optuna trials: {CONFIG['optuna_trials']} per model\n"
                f"AI narrative generated: yes\n"
                f"Files: Model_Metrics.md, final_model.pkl, "
                f"model_comparison.png, feature_importance.png")
    except Exception as e:
        return f"ML_ERROR: {e}\n{traceback.format_exc()}"


# ── README ─────────────────────────────────────────────────────────────────────

@tool("generate_readme")
def generate_readme(_: str = "") -> str:
    """Generates a complete README.md with AI narrative. No parameters."""
    try:
        metrics = ""
        target_info = ""
        ctx = _read_ctx()

        if os.path.exists(CONFIG["metrics_md"]):
            with open(CONFIG["metrics_md"], encoding="utf-8") as f:
                metrics = f.read()
        if os.path.exists(CONFIG["target_json"]):
            with open(CONFIG["target_json"]) as f:
                cfg_t = json.load(f)
            target_info = (
                f"**Target identified by AI:** `{cfg_t['target_col']}`  \n"
                f"**Justification:** {cfg_t.get('ai_justification', '')}  \n"
                f"**Type:** `{cfg_t['problem_type']}`"
            )

        # Claude writes the project executive summary
        prompt_readme = f"""Write a 2-paragraph executive summary for a Data Science project.

Context: {ctx or 'Automated resume screening pipeline with 200k candidates.'}
Target: {cfg_t.get('target_col', 'hired') if os.path.exists(CONFIG['target_json']) else 'response variable'}

The summary should be professional, mention the main results, and be suitable
for a GitHub README. Maximum 150 words."""

        exec_summary = _ask_claude(prompt_readme, max_tokens=300)

        content = f"""# Auto Data Scientist v7 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![BigQuery](https://img.shields.io/badge/BigQuery-SQL-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gurobi](https://img.shields.io/badge/Gurobi-Optimization-007f00?style=flat&logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Google Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?style=flat&logo=google-colab&logoColor=white)](https://colab.research.google.com/)

> {exec_summary}

---
## Architecture

**Orchestration Layer:** CrewAI (stable, sequential, 1 tool per agent)
**Intelligence Layer:** Claude 3.5 Sonnet called inside each tool

| What AI Actually Does |
|-----------------------|
| Analyzes the dataset and automatically identifies the target |
| Writes and executes custom Python analysis code |
| Detects code errors and self-corrects (self-healing) |
| Decides which features to create based on real data |
| Interprets model results in natural language |
| Writes a narrative performance diagnostic |

---
## Target Selection by AI
{target_info}

---
## Data Quality
- KNN imputation (numeric) + Mode (categorical)
- Intelligent analysis by Claude with business insights

[Quality_Report.md](Quality_Report.md)

---
## Medallion Architecture

| Layer | File | Description |
|-------|------|-------------|
| Silver | df1_silver.parquet | Standardized raw data + imputed |
| Gold | df2_gold.parquet | Standard features + AI-generated features |
| ML-Ready | df3_ml_ready.parquet | No redundancies or IDs |

![Sample](dataset_sample.png)

---
## EDA
![Distributions](distributions.png)
![Boxplots](boxplots.png)
![Categoricals](categoricals.png)
![Target](target_dist.png)
![Correlation](correlation_matrix.png)
![AI Analysis](intelligent_analysis.png)

---
## Modeling — CV + Optuna + Stacking + AI Interpretation

{metrics}

![Comparison](model_comparison.png)
![Features](feature_importance.png)

---
## Agent Architecture

| Agent | Tool | AI Intelligence |
|-------|------|----------------|
| Ingestor | download_and_save_silver | Not required |
| Analyst | analyze_data_with_ai | Claude analyzes dataset, identifies target, writes and executes code, self-healing |
| Feature Engineer | generate_features_with_ai_strategy | Claude decides and writes custom feature code |
| EDA Analyst | generate_eda_and_ml_ready | Pure Python (visualizations) |
| ML Scientist | train_and_save_model | Claude interprets results and writes narrative |

---
## How to Reproduce
```bash
git clone <repo>
echo "KAGGLE_USERNAME=x" >> .env
echo "KAGGLE_KEY=y" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
# Optional:
echo "We want to predict whether candidates will be hired." > business_context.txt
pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic \\
            scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm
python auto_data_scientist_v7.py
```

---
*Auto Data Scientist v7 — CrewAI + Claude 3.5 Sonnet + Optuna*
"""
        with open(CONFIG["readme_md"], "w", encoding="utf-8") as f:
            f.write(content)
        return "README_SUCCESS: README.md generated."
    except Exception as e:
        return f"README_ERROR: {e}"


# ── STEP 6: Deploy — Predictions Parquet + Streamlit App ─────────────────────

@tool("deploy_streamlit_app")
def deploy_streamlit_app(_: str = "") -> str:
    """
    Loads final_model.pkl and df1_silver.parquet (all original columns).
    Runs predictions on the full dataset and saves df4_predictions.parquet
    with ALL original columns + 'prediction' + 'prediction_proba' columns.
    Claude writes a full Streamlit app tailored to the dataset for stakeholders.
    Saves streamlit_app.py and requirements.txt.
    Returns DEPLOY_SUCCESS or ERROR. No parameters.
    """
    try:
        # ── Validate prerequisites ─────────────────────────────────────────────
        for required in [CONFIG["model_pkl"], CONFIG["silver_path"],
                         CONFIG["ml_ready_path"], CONFIG["target_json"]]:
            if not os.path.exists(required):
                return f"ERROR: {required} not found. Run the full pipeline first."

        with open(CONFIG["model_pkl"], "rb") as f:
            artifact = pickle.load(f)

        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)

        model        = artifact["model"]
        target_col   = artifact["target"]
        problem_type = artifact["type"]
        model_name   = artifact["name"]
        features     = artifact["features"]   # dummified feature names
        le           = artifact.get("label_encoder")
        test_score   = artifact.get("test_score", 0.0)

        # ── Load original silver (all columns intact) ──────────────────────────
        df_silver = pd.read_parquet(CONFIG["silver_path"])
        df_ml     = pd.read_parquet(CONFIG["ml_ready_path"]).dropna()

        # ── Build feature matrix from ML-ready data ────────────────────────────
        feat_cols = [c for c in df_ml.columns if c != target_col]
        X_full    = pd.get_dummies(df_ml[feat_cols],
                                   drop_first=True).reindex(columns=features,
                                                             fill_value=0)

        # ── Run predictions ────────────────────────────────────────────────────
        raw_preds = model.predict(X_full)

        # Decode labels if classification
        if le is not None:
            pred_labels = le.inverse_transform(raw_preds)
        else:
            pred_labels = raw_preds

        # Probabilities (classification) or None (regression)
        pred_proba = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba_matrix = model.predict_proba(X_full)
            pred_proba   = proba_matrix.max(axis=1)   # confidence of winning class

        # ── Build df4_predictions: silver rows + prediction columns ───────────
        # Align silver to the rows that survived dropna in ml_ready
        min_rows = min(len(df_silver), len(df_ml))
        df_pred  = df_silver.iloc[:min_rows].copy().reset_index(drop=True)

        df_pred["prediction"] = pred_labels[:min_rows]

        if pred_proba is not None:
            df_pred["prediction_proba"] = pred_proba[:min_rows].round(4)

        # Keep original target for actual vs predicted comparison
        if target_col not in df_pred.columns and target_col in df_ml.columns:
            df_pred[target_col] = df_ml[target_col].values[:min_rows]

        df_pred.to_parquet(CONFIG["predictions_path"], index=False)
        logger.info(f"[Deploy] df4_predictions saved: {df_pred.shape}")

        # ── Summarise dataset for Claude to write the app ──────────────────────
        num_cols  = df_pred.select_dtypes(include="number").columns.tolist()
        cat_cols  = [c for c in df_pred.select_dtypes(include="object").columns
                     if df_pred[c].nunique() <= 30]
        ctx       = _read_ctx()

        col_info = {}
        for c in df_pred.columns[:25]:   # cap to avoid huge prompt
            s = df_pred[c]
            col_info[c] = {
                "dtype":   str(s.dtype),
                "nunique": int(s.nunique()),
                "sample":  s.dropna().head(3).tolist(),
            }

        metric_label = "Accuracy" if problem_type == "classification" else "R²"

        prompt_app = f"""You are an expert Streamlit developer building a stakeholder-facing
decision support app for a {problem_type} model.

BUSINESS CONTEXT: {ctx or 'Resume screening — predict hiring outcomes for candidates.'}
TARGET COLUMN: '{target_col}'
PREDICTION COLUMN: 'prediction'
{'CONFIDENCE COLUMN: prediction_proba (0–1 float)' if pred_proba is not None else ''}
PROBLEM TYPE: {problem_type}
MODEL: {model_name}  |  {metric_label}: {test_score:.4f}
ROWS IN df4_predictions.parquet: {len(df_pred)}

AVAILABLE COLUMNS (first 25):
{json.dumps(col_info, indent=2, default=_safe_json)}

NUMERIC COLUMNS: {num_cols[:10]}
CATEGORICAL COLUMNS: {cat_cols[:8]}

Write a COMPLETE, production-ready Streamlit app (streamlit_app.py).

MANDATORY SECTIONS (use st.tabs or sidebar nav):

1. OVERVIEW TAB — KPI cards row:
   - Total records
   - Model score ({metric_label}: {test_score:.4f})
   - {'Most predicted class + % share' if problem_type == 'classification' else 'Mean prediction ± std'}
   - {'Avg confidence (prediction_proba)' if pred_proba is not None else 'RMSE on available actuals'}

2. ACTUAL vs PREDICTED TAB:
   {'- Confusion matrix heatmap (seaborn) with annotation' if problem_type == 'classification' else '- Scatter plot: actual (x) vs prediction (y) with identity line'}
   {'- Bar chart: class distribution — actual vs predicted side by side' if problem_type == 'classification' else '- Residuals histogram'}
   - Only show rows where both target and prediction exist

3. EXPLORE PREDICTIONS TAB:
   - Sidebar multiselect filters for all categorical columns
   - Slider filters for top-3 numeric columns
   - Filtered dataframe with color-coded prediction column
   {'- Download button for filtered CSV' if pred_proba is not None else '- Download button for CSV'}

4. FEATURE INSIGHTS TAB:
   - Load and display feature_importance.png (if exists)
   - Load and display correlation_matrix.png (if exists)
   - st.caption describing what each chart shows

STYLE REQUIREMENTS:
- Dark theme friendly (use st.set_page_config with layout="wide")
- Page title and subtitle that mention the business context
- Use st.metric for KPIs with delta where meaningful
- Use plotly express for interactive charts (import plotly.express as px)
- Add a footer: "Powered by Auto Data Scientist v7 · CrewAI + Claude 3.5 Sonnet"

DATA LOADING:
- Load df4_predictions.parquet with @st.cache_data
- Handle missing columns gracefully with try/except

Write ONLY the complete Python code. No markdown. No explanations."""

        app_code = _ask_claude(prompt_app, max_tokens=4000)

        # Strip markdown fences if Claude wrapped it
        if "```python" in app_code:
            app_code = app_code.split("```python")[1].split("```")[0]
        elif "```" in app_code:
            app_code = app_code.split("```")[1].split("```")[0]

        with open(CONFIG["streamlit_app"], "w", encoding="utf-8") as f:
            f.write(app_code.strip())

        # ── requirements.txt ──────────────────────────────────────────────────
        requirements = """streamlit>=1.35.0
pandas>=2.0.0
pyarrow>=14.0.0
plotly>=5.18.0
seaborn>=0.13.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
numpy>=1.26.0
xgboost>=2.0.0
lightgbm>=4.3.0
"""
        with open(CONFIG["requirements_txt"], "w", encoding="utf-8") as f:
            f.write(requirements)

        # ── Claude writes deployment instructions ──────────────────────────────
        deploy_md = f"""# Streamlit Deployment Guide

## Files Generated
| File | Description |
|------|-------------|
| `streamlit_app.py` | Stakeholder-facing dashboard |
| `df4_predictions.parquet` | Full dataset + predictions |
| `requirements.txt` | Python dependencies |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set **Main file**: `streamlit_app.py`
4. Click **Deploy**

## What the App Shows
- **Overview**: KPI cards — total records, model {metric_label} ({test_score:.4f}), prediction distribution
- **Actual vs Predicted**: {'Confusion matrix + class distribution comparison' if problem_type == 'classification' else 'Scatter plot + residuals histogram'}
- **Explore Predictions**: Filterable table with color-coded predictions + CSV download
- **Feature Insights**: Feature importance and correlation matrix charts

## Prediction Data Schema
- All {len(df_pred.columns)} original columns preserved
- `prediction` — model output ({'class label' if problem_type == 'classification' else 'numeric value'})
{f"- `prediction_proba` — model confidence (0–1)" if pred_proba is not None else ""}
- `{target_col}` — actual value (ground truth for comparison)
"""
        with open(os.path.join(_BASE_DIR, "Deployment_Guide.md"), "w", encoding="utf-8") as f:
            f.write(deploy_md)

        return (f"DEPLOY_SUCCESS\n"
                f"Predictions saved: df4_predictions.parquet ({len(df_pred)} rows, "
                f"{len(df_pred.columns)} columns)\n"
                f"Columns: {list(df_pred.columns[:8])} ... + prediction"
                f"{' + prediction_proba' if pred_proba is not None else ''}\n"
                f"Streamlit app: streamlit_app.py\n"
                f"Requirements: requirements.txt\n"
                f"Guide: Deployment_Guide.md\n"
                f"Run: streamlit run streamlit_app.py")

    except Exception as e:
        return f"DEPLOY_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 7: Generate Analysis Notebook ───────────────────────────────────────

@tool("generate_analysis_notebook")
def generate_analysis_notebook(_: str = "") -> str:
    """
    Compiles all pipeline outputs — markdown reports, charts, metrics,
    Claude narratives, and a live prediction preview — into a single
    analysis_notebook.ipynb that renders beautifully on GitHub.
    Returns NOTEBOOK_SUCCESS or ERROR. No parameters.
    """
    try:
        # ── Load all available context ─────────────────────────────────────────
        def _read_md(path):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return f.read()
            return ""

        def _img_cell(path, caption=""):
            """Code cell that displays a saved image inline."""
            fname = os.path.basename(path)
            meta  = ", metadata={'width': 900}"
            code  = (
                f"from IPython.display import Image, display\n"
                f"display(Image(filename='{fname}'{meta}))"
            )
            if caption:
                code += f"\nprint('{caption}')"
            return new_code_cell(code)

        ctx          = _read_ctx()
        quality_md   = _read_md(CONFIG["quality_md"])
        analysis_md  = _read_md(CONFIG["analysis_md"])
        metrics_md   = _read_md(CONFIG["metrics_md"])
        eval_md      = _read_md(CONFIG["eval_md"])
        deploy_guide = _read_md(os.path.join(_BASE_DIR, "Deployment_Guide.md"))

        target_col   = "unknown"
        problem_type = "unknown"
        model_name   = "unknown"
        test_score   = 0.0
        ai_insights  = []

        if os.path.exists(CONFIG["target_json"]):
            with open(CONFIG["target_json"]) as f:
                cfg_t = json.load(f)
            target_col   = cfg_t.get("target_col", "unknown")
            problem_type = cfg_t.get("problem_type", "unknown")
            ai_insights  = cfg_t.get("ai_insights", [])

        if os.path.exists(CONFIG["model_pkl"]):
            with open(CONFIG["model_pkl"], "rb") as f:
                art = pickle.load(f)
            model_name = art.get("name", "unknown")
            test_score = art.get("test_score", 0.0)

        metric_label = "Accuracy" if problem_type == "classification" else "R²"

        # ── Claude writes executive summary cells ──────────────────────────────
        prompt_nb = f"""You are writing the introduction for a Jupyter notebook
that documents a complete automated Data Science pipeline.

Business context: {ctx or 'Automated ML pipeline on a structured dataset.'}
Target: '{target_col}' ({problem_type})
Best model: {model_name} | {metric_label}: {test_score:.4f}
Top insights discovered: {ai_insights}

Write 3 things:
1. A one-paragraph executive summary (what was done, what was found)
2. A 3-row markdown table: | Step | Tool | Output |
   covering: Ingestion, Analysis, Feature Engineering, EDA, Modeling, Deployment
3. A one-paragraph conclusion with business recommendations

Format your response as JSON:
{{
  "executive_summary": "paragraph here",
  "pipeline_table": "| Step | Tool | Output |\\n|---|---|---|\\n...",
  "conclusion": "paragraph here"
}}
Respond ONLY with the JSON."""

        nb_text = _ask_claude(prompt_nb, max_tokens=1200)
        try:
            if "```json" in nb_text:
                nb_text = nb_text.split("```json")[1].split("```")[0]
            elif "```" in nb_text:
                nb_text = nb_text.split("```")[1].split("```")[0]
            nb_content = json.loads(nb_text.strip())
        except Exception:
            nb_content = {
                "executive_summary": "Automated ML pipeline completed successfully.",
                "pipeline_table": "| Step | Tool | Output |\n|---|---|---|\n| All | CrewAI | See sections below |",
                "conclusion": "Model is ready for production deployment via Streamlit.",
            }

        # ── Build notebook cells ───────────────────────────────────────────────
        cells = []

        # ── SECTION 0: Header ─────────────────────────────────────────────────
        cells.append(new_markdown_cell(
            f"# Auto Data Scientist v7 — Analysis Notebook\n\n"
            f"> **Target:** `{target_col}` | "
            f"**Problem:** {problem_type} | "
            f"**Best Model:** {model_name} | "
            f"**{metric_label}:** {test_score:.4f}\n\n"
            f"*Generated automatically by CrewAI + Claude 3.5 Sonnet*\n\n"
            f"---\n\n"
            f"## Executive Summary\n\n"
            f"{nb_content['executive_summary']}\n\n"
            f"## Pipeline Overview\n\n"
            f"{nb_content['pipeline_table']}"
        ))

        # ── SECTION 1: Setup ──────────────────────────────────────────────────
        cells.append(new_markdown_cell("---\n## 1. Environment Setup"))
        cells.append(new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import json, pickle, os\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "pd.set_option('display.max_columns', 50)\n"
            "pd.set_option('display.float_format', '{:.4f}'.format)\n"
            "print('Libraries loaded.')"
        ))

        # ── SECTION 2: Data Quality ───────────────────────────────────────────
        cells.append(new_markdown_cell("---\n## 2. Data Quality Report\n\n"
                                        + quality_md))

        cells.append(new_markdown_cell("### Silver Dataset — Preview"))
        cells.append(new_code_cell(
            "df_silver = pd.read_parquet('df1_silver.parquet')\n"
            "print(f'Shape: {df_silver.shape}')\n"
            "print(f'Columns: {list(df_silver.columns)}')\n"
            "df_silver.head()"
        ))

        cells.append(new_code_cell(
            "# Null values overview\n"
            "nulls = df_silver.isnull().sum()\n"
            "nulls[nulls > 0].sort_values(ascending=False)"
        ))

        # ── SECTION 3: Intelligent Analysis ──────────────────────────────────
        cells.append(new_markdown_cell("---\n## 3. Intelligent Analysis by Claude\n\n"
                                        + analysis_md))

        if os.path.exists(os.path.join(_BASE_DIR, "intelligent_analysis.png")):
            cells.append(new_markdown_cell("### AI-Generated Analysis Chart"))
            cells.append(_img_cell(
                os.path.join(_BASE_DIR, "intelligent_analysis.png"),
                "Chart generated by Claude's custom analysis code"
            ))

        # ── SECTION 4: EDA ────────────────────────────────────────────────────
        cells.append(new_markdown_cell("---\n## 4. Exploratory Data Analysis"))

        cells.append(new_markdown_cell("### Gold Dataset — After Feature Engineering"))
        cells.append(new_code_cell(
            "df_gold = pd.read_parquet('df2_gold.parquet')\n"
            "print(f'Shape after feature engineering: {df_gold.shape}')\n"
            "df_gold.describe().T.round(3)"
        ))

        for img_file, caption in [
            ("target_dist.png",       f"Target Distribution — `{target_col}`"),
            ("distributions.png",     "Feature Distributions"),
            ("boxplots.png",          "Boxplots — Outlier Detection"),
            ("categoricals.png",      "Categorical Feature Distributions"),
            ("correlation_matrix.png","Correlation Matrix"),
        ]:
            fpath = os.path.join(_BASE_DIR, img_file)
            if os.path.exists(fpath):
                cells.append(new_markdown_cell(f"### {caption}"))
                cells.append(_img_cell(fpath, caption))

        # ── SECTION 5: Feature Engineering ───────────────────────────────────
        cells.append(new_markdown_cell("---\n## 5. Feature Engineering"))

        if os.path.exists(CONFIG["strategy_json"]):
            with open(CONFIG["strategy_json"]) as f:
                strat = json.load(f)
            cells.append(new_code_cell(
                f"# Feature Engineering Summary\n"
                f"strategy = {json.dumps(strat, indent=2, default=_safe_json)}\n"
                f"print('Standard features created:', strategy.get('standard_features', []))\n"
                f"print('AI-generated features:', strategy.get('ai_features', []))\n"
                f"print('AI code executed successfully:', strategy.get('ai_success', False))"
            ))

        # ── SECTION 6: Model Results ──────────────────────────────────────────
        cells.append(new_markdown_cell("---\n## 6. Model Training & Evaluation\n\n"
                                        + metrics_md))

        for img_file, caption in [
            ("model_comparison.png",   "Model Comparison — Baseline vs Optuna vs Stacking"),
            ("feature_importance.png", "Top 15 Feature Importances"),
            ("actual_vs_predicted.png","Actual vs Predicted (Regression)"),
        ]:
            fpath = os.path.join(_BASE_DIR, img_file)
            if os.path.exists(fpath):
                cells.append(new_markdown_cell(f"### {caption}"))
                cells.append(_img_cell(fpath, caption))

        cells.append(new_markdown_cell("### Model Evaluation\n\n" + eval_md))

        # ── SECTION 7: Live Predictions Preview ───────────────────────────────
        cells.append(new_markdown_cell("---\n## 7. Predictions — Full Dataset"))
        cells.append(new_code_cell(
            "df_pred = pd.read_parquet('df4_predictions.parquet')\n"
            "print(f'Shape: {df_pred.shape}')\n"
            f"print(f'Prediction distribution:')\n"
            f"print(df_pred['prediction'].value_counts())\n"
            "df_pred.head(10)"
        ))

        cells.append(new_code_cell(
            "# Actual vs Predicted comparison\n"
            f"if '{target_col}' in df_pred.columns:\n"
            f"    match = (df_pred['{target_col}'].astype(str) == \n"
            f"             df_pred['prediction'].astype(str)).mean()\n"
            f"    print(f'Match rate: {{match:.4f}} ({metric_label})')\n"
            f"    print(df_pred['{target_col}'].value_counts().rename('actual'))\n"
            f"    print(df_pred['prediction'].value_counts().rename('predicted'))"
        ))

        if problem_type == "classification":
            cells.append(new_code_cell(
                "# Confusion matrix\n"
                "from sklearn.metrics import confusion_matrix\n"
                "import seaborn as sns\n\n"
                f"if '{target_col}' in df_pred.columns:\n"
                f"    cm = confusion_matrix(\n"
                f"        df_pred['{target_col}'].astype(str),\n"
                f"        df_pred['prediction'].astype(str)\n"
                f"    )\n"
                f"    fig, ax = plt.subplots(figsize=(7, 5))\n"
                f"    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)\n"
                f"    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')\n"
                f"    ax.set_title('Confusion Matrix — {target_col}')\n"
                f"    plt.tight_layout(); plt.show()"
            ))

        if "prediction_proba" in (pd.read_parquet(CONFIG["predictions_path"]).columns
                                   if os.path.exists(CONFIG["predictions_path"]) else []):
            cells.append(new_code_cell(
                "# Confidence distribution\n"
                "fig, ax = plt.subplots(figsize=(8, 4))\n"
                "df_pred['prediction_proba'].hist(bins=40, ax=ax, color='steelblue', "
                "edgecolor='white', alpha=0.85)\n"
                "ax.set_xlabel('Prediction Confidence'); ax.set_ylabel('Count')\n"
                "ax.set_title('Model Confidence Distribution')\n"
                "ax.grid(axis='y', alpha=0.3)\n"
                "plt.tight_layout(); plt.show()"
            ))

        # ── SECTION 8: Deployment ─────────────────────────────────────────────
        cells.append(new_markdown_cell("---\n## 8. Deployment\n\n" + deploy_guide))
        cells.append(new_code_cell(
            "# Files generated by the full pipeline\n"
            "files = [\n"
            "    'df1_silver.parquet', 'df2_gold.parquet',\n"
            "    'df3_ml_ready.parquet', 'df4_predictions.parquet',\n"
            "    'final_model.pkl', 'streamlit_app.py',\n"
            "    'requirements.txt', 'analysis_notebook.ipynb',\n"
            "]\n"
            "for f in files:\n"
            "    exists = '✅' if os.path.exists(f) else '❌'\n"
            "    size   = f'{os.path.getsize(f)/1024:.1f} KB' if os.path.exists(f) else '-'\n"
            "    print(f'{exists}  {f:<40} {size}')"
        ))

        # ── SECTION 9: Conclusion ─────────────────────────────────────────────
        cells.append(new_markdown_cell(
            f"---\n## 9. Conclusion\n\n"
            f"{nb_content['conclusion']}\n\n"
            f"---\n"
            f"*Auto Data Scientist v7 · CrewAI + Claude 3.5 Sonnet + Optuna*"
        ))

        # ── Write the notebook ────────────────────────────────────────────────
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["language_info"] = {
            "name": "python",
            "version": "3.10.0",
        }

        with open(CONFIG["notebook_path"], "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        n_cells = len(cells)
        logger.info(f"[Notebook] Saved: {CONFIG['notebook_path']} ({n_cells} cells)")

        return (f"NOTEBOOK_SUCCESS\n"
                f"File: analysis_notebook.ipynb\n"
                f"Cells: {n_cells}\n"
                f"Sections: Setup, Data Quality, Intelligent Analysis, EDA, "
                f"Feature Engineering, Modeling, Live Predictions, Deployment, Conclusion\n"
                f"Open with: jupyter notebook analysis_notebook.ipynb\n"
                f"Renders on GitHub automatically.")

    except Exception as e:
        return f"NOTEBOOK_ERROR: {e}\n{traceback.format_exc()}"


# ==========================================
# 5. AGENTS — ONE TOOL, SIMPLE RULE
# ==========================================

ingestor = Agent(
    role="Data Engineer",
    goal=("Download the dataset by calling download_and_save_silver. "
          "If it returns INGESTION_SUCCESS declare done. "
          "If ERROR try again."),
    backstory="Specialist in data ingestion.",
    tools=[download_and_save_silver],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

analyst = Agent(
    role="AI-Powered Data Analyst",
    goal=("Analyze the dataset by calling analyze_data_with_ai. "
          "If it returns ANALYSIS_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Uses Claude internally for intelligent analysis, "
               "target identification, and custom code generation."),
    tools=[analyze_data_with_ai],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

feature_engineer = Agent(
    role="AI-Powered Feature Engineer",
    goal=("Generate features by calling generate_features_with_ai_strategy. "
          "If it returns FEATURES_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Uses Claude to decide and create custom features "
               "specific to the dataset."),
    tools=[generate_features_with_ai_strategy],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

eda_analyst = Agent(
    role="EDA Analyst",
    goal=("Generate visualizations by calling generate_eda_and_ml_ready. "
          "If it returns EDA_SUCCESS declare done. "
          "If ERROR try again."),
    backstory="Generates visualizations and prepares the dataset for ML.",
    tools=[generate_eda_and_ml_ready],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

ml_scientist = Agent(
    role="AI-Powered ML Scientist",
    goal=("Train and save the best model by calling train_and_save_model. "
          "If it returns ML_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("CV + Optuna + Stacking. Claude interprets the results "
               "and writes a performance narrative."),
    tools=[train_and_save_model],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

deployer = Agent(
    role="ML Deployment Engineer",
    goal=("Deploy the model by calling deploy_streamlit_app. "
          "If it returns DEPLOY_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Generates df4_predictions.parquet with all original columns "
               "plus the prediction column, then writes a full Streamlit "
               "stakeholder app using Claude to tailor it to the dataset."),
    tools=[deploy_streamlit_app],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

notebook_writer = Agent(
    role="Technical Notebook Writer",
    goal=("Generate the analysis notebook by calling generate_analysis_notebook. "
          "If it returns NOTEBOOK_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Compiles all pipeline outputs — reports, charts, metrics, "
               "Claude narratives, and live predictions — into a single "
               "analysis_notebook.ipynb that renders on GitHub."),
    tools=[generate_analysis_notebook],
    llm=llm_agent, verbose=True,
    max_iter=CONFIG["max_iter"],
    max_retry_limit=CONFIG["max_retry_limit"],
)

# ==========================================
# 6. TASKS
# ==========================================

task_ingestion = Task(
    description=("Download the dataset from Kaggle.\n"
                 "Call download_and_save_silver (no parameters).\n"
                 "If INGESTION_SUCCESS finish. If ERROR try again."),
    agent=ingestor,
    expected_output="INGESTION_SUCCESS with shape and columns.",
)

task_analysis = Task(
    description=("Analyze the dataset with AI.\n"
                 "Call analyze_data_with_ai (no parameters).\n"
                 "If ANALYSIS_SUCCESS finish. If ERROR try again."),
    agent=analyst,
    context=[task_ingestion],
    expected_output="ANALYSIS_SUCCESS with identified target and insights.",
)

task_features = Task(
    description=("Generate features with AI strategy.\n"
                 "Call generate_features_with_ai_strategy (no parameters).\n"
                 "If FEATURES_SUCCESS finish. If ERROR try again."),
    agent=feature_engineer,
    context=[task_analysis],
    expected_output="FEATURES_SUCCESS with standard and custom features.",
)

task_eda = Task(
    description=("Generate visualizations and ML-Ready dataset.\n"
                 "Call generate_eda_and_ml_ready (no parameters).\n"
                 "If EDA_SUCCESS finish. If ERROR try again."),
    agent=eda_analyst,
    context=[task_features],
    expected_output="EDA_SUCCESS with charts and df3_ml_ready.parquet.",
)

task_ml = Task(
    description=("Train and save the best model.\n"
                 "Call train_and_save_model (no parameters).\n"
                 "If ML_SUCCESS finish. If ERROR try again."),
    agent=ml_scientist,
    context=[task_eda],
    expected_output="ML_SUCCESS with model, score, and narrative.",
)

task_deploy = Task(
    description=("Deploy the model as a Streamlit app.\n"
                 "Call deploy_streamlit_app (no parameters).\n"
                 "If DEPLOY_SUCCESS finish. If ERROR try again."),
    agent=deployer,
    context=[task_ml],
    expected_output=(
        "DEPLOY_SUCCESS with df4_predictions.parquet "
        "(all original columns + prediction), streamlit_app.py, "
        "requirements.txt, and Deployment_Guide.md."
    ),
)

task_notebook = Task(
    description=("Generate the analysis notebook.\n"
                 "Call generate_analysis_notebook (no parameters).\n"
                 "If NOTEBOOK_SUCCESS finish. If ERROR try again."),
    agent=notebook_writer,
    context=[task_deploy],
    expected_output=(
        "NOTEBOOK_SUCCESS with analysis_notebook.ipynb containing "
        "all pipeline sections: setup, data quality, EDA, feature engineering, "
        "modeling, live predictions, deployment, and conclusion."
    ),
)

# ==========================================
# 7. CREW
# ==========================================

ds_squad = Crew(
    agents=[ingestor, analyst, feature_engineer, eda_analyst,
            ml_scientist, deployer, notebook_writer],
    tasks=[task_ingestion, task_analysis, task_features, task_eda,
           task_ml, task_deploy, task_notebook],
    process=Process.sequential,
    memory=False,
    verbose=True,
)

# ==========================================
# 8. POST-PIPELINE
# ==========================================

def evaluate_model():
    """Overfitting/underfitting diagnostic in pure Python."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(CONFIG["model_pkl"]):
        print("final_model.pkl not found."); return

    try:
        with open(CONFIG["model_pkl"], "rb") as f:
            artifact = pickle.load(f)

        model        = artifact["model"]
        target       = artifact["target"]
        problem_type = artifact["type"]
        name         = artifact["name"]
        features     = artifact["features"]
        le           = artifact.get("label_encoder")

        df = pd.read_parquet(CONFIG["ml_ready_path"]).dropna()
        X  = pd.get_dummies(df[[c for c in df.columns if c != target]],
                            drop_first=True).reindex(columns=features, fill_value=0)
        y  = df[target].copy()
        if le:
            y = le.transform(y.astype(str))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y if problem_type == "classification" else None,
        )

        y_ptr = model.predict(X_tr)
        y_pte = model.predict(X_te)

        if problem_type == "classification":
            s_tr = accuracy_score(y_tr, y_ptr)
            s_te = accuracy_score(y_te, y_pte)
            met  = "Accuracy"
        else:
            s_tr = r2_score(y_tr, y_ptr)
            s_te = r2_score(y_te, y_pte)
            met  = "R²"

        gap = s_tr - s_te

        # Claude interprets the diagnostic
        prompt_diag = f"""Diagnose this ML model in 2 short paragraphs:

Model: {name}
Type: {problem_type} | Target: {target}
{met} Train: {s_tr:.4f}
{met} Test: {s_te:.4f}
Gap: {gap:.4f}

State whether there is overfitting, underfitting, or if the model is well-fitted.
Be direct and practical."""

        diagnostic = _ask_claude(prompt_diag, max_tokens=400)

        # Regression chart
        if problem_type == "regression":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(y_te, y_pte, alpha=0.3, s=10, color="#4C72B0")
            mn = min(float(np.array(y_te).min()), float(y_pte.min()))
            mx = max(float(np.array(y_te).max()), float(y_pte.max()))
            ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted — {name}", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "actual_vs_predicted.png"), dpi=150)
            plt.close()

        content = f"""# Model Evaluation

## `{name}`
**Type:** {problem_type} | **Target:** `{target}`

| Dataset   | {met} |
|-----------|-------|
| Train     | {s_tr:.4f} |
| Test      | {s_te:.4f} |
| Gap       | {gap:.4f}  |

## AI Diagnostic

{diagnostic}

## Optimized Parameters (Optuna)
```json
{json.dumps(artifact.get('optuna_params', {}), indent=2)}
```
"""
        with open(CONFIG["eval_md"], "w", encoding="utf-8") as f:
            f.write(content)

        print(f"{met} Train: {s_tr:.4f} | Test: {s_te:.4f} | Gap: {gap:.4f}")
        print(f"Model_Evaluation.md saved.")

    except Exception as e:
        print(f"ERROR in evaluation: {e}\n{traceback.format_exc()}")


def run_post_pipeline():
    evaluate_model()

    print("\n" + "=" * 60)
    print("GENERATING README.md")
    print("=" * 60)
    print(generate_readme.func(""))

    print("\n" + "=" * 60)
    print("GIT VERSIONING")
    print("=" * 60)

    def git(cmd, timeout=300):
        print(f"\n> {cmd}")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=timeout)
            output = (r.stdout or r.stderr).strip()
            status = "[OK]" if r.returncode == 0 else "[FAILED]"
            print(f"{status} {output[:200]}")
            return r.returncode == 0
        except Exception as e:
            print(f"[ERROR] {e}"); return False

    git("git init")
    git("git remote remove origin")
    git("git remote add origin https://github.com/bttisrael/agente_ds2.git")

    gitignore = "\n".join([
        "final_model.pkl", "df1_silver.parquet", "df2_gold.parquet",
        "df3_ml_ready.parquet", "df4_predictions.parquet",
        ".env", "venv/", "__pycache__/",
        "*.pyc", "pipeline.log",
    ]) + "\n"
    with open(os.path.join(_BASE_DIR, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(gitignore)

    for file in ["final_model.pkl", "df1_silver.parquet",
                 "df2_gold.parquet", "df3_ml_ready.parquet"]:
        git(f"git rm --cached {file}")

    git("git config http.postBuffer 524288000")
    git("git branch -M main")

    artifacts = [
        ".gitignore", "README.md", "ollama_ds_v6.py",
        "streamlit_app.py", "requirements.txt", "analysis_notebook.ipynb",
        "correlation_matrix.png", "distributions.png", "boxplots.png",
        "categoricals.png", "target_dist.png", "dataset_sample.png",
        "intelligent_analysis.png", "feature_importance.png",
        "model_comparison.png",
        "Descriptive_Statistics.md", "Model_Metrics.md",
        "Model_Evaluation.md", "Quality_Report.md",
        "Intelligent_Analysis.md", "Deployment_Guide.md",
        "target_config.json", "feature_strategy.json",
    ]
    if os.path.exists(os.path.join(_BASE_DIR, "actual_vs_predicted.png")):
        artifacts.append("actual_vs_predicted.png")
    if os.path.exists(CONFIG["business_ctx"]):
        artifacts.append("business_context.txt")

    for file in artifacts:
        git(f"git add {file}")

    git('git commit -m "feat: pipeline v7 - Claude reasoning inside tools"')
    git("git push origin main")

    print("\n" + "=" * 60)
    print("PIPELINE v7 COMPLETE")
    print("Deploy: streamlit run streamlit_app.py")
    print("=" * 60)


# ==========================================
# 9. MAIN
# ==========================================

if __name__ == "__main__":
    logger.info("Auto Data Scientist v7 starting...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not found in .env. "
                     "Add: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if not os.path.exists(".env"):
        logger.warning("Create .env with KAGGLE_USERNAME, KAGGLE_KEY, and ANTHROPIC_API_KEY.")

    if not os.path.exists(CONFIG["business_ctx"]):
        logger.info("Tip: create business_context.txt to provide business context.")

    result = ds_squad.kickoff()

    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(result)

    run_post_pipeline()
    logger.info("Pipeline v7 finished.")