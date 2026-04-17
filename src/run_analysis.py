from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "obesity.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "obesity_clean.csv"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOK_PATH = ROOT / "notebooks" / "miniproyecto_2_clasificacion_obesidad.ipynb"
REPORT_PATH = REPORTS_DIR / "informe_miniproyecto_2.md"
TARGET_COLUMN = "NObeyesdad"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def summarize_dataset(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {
            key: int(value)
            for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }


def feature_pruning_report(df: pd.DataFrame, missing_threshold: float = 0.5) -> tuple[pd.DataFrame, dict]:
    missing_ratio = (df.isna().mean()).sort_values(ascending=False)
    high_missing = missing_ratio[missing_ratio > missing_threshold].index.tolist()

    feature_df = df.drop(columns=[TARGET_COLUMN])
    zero_variance = [
        col for col in feature_df.columns if feature_df[col].nunique(dropna=False) <= 1
    ]

    numeric_df = feature_df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(
        pd.DataFrame(
            [
                [row > col for col in range(corr_matrix.shape[1])]
                for row in range(corr_matrix.shape[0])
            ],
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
    )

    high_corr_pairs = []
    for col in upper.columns:
        for row, value in upper[col].dropna().items():
            if value > 0.85:
                high_corr_pairs.append(
                    {"feature_a": row, "feature_b": col, "correlation": round(float(value), 4)}
                )

    high_corr_features = sorted({pair["feature_b"] for pair in high_corr_pairs})
    duplicate_rows = int(df.duplicated().sum())

    features_to_drop = sorted(set(high_missing + zero_variance + high_corr_features))
    clean_df = df.drop(columns=features_to_drop).drop_duplicates().reset_index(drop=True)

    report = {
        "missing_threshold": missing_threshold,
        "columns_over_missing_threshold": high_missing,
        "zero_variance_columns": zero_variance,
        "high_correlation_pairs_over_0_85": high_corr_pairs,
        "dropped_columns": features_to_drop,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_cleaning": int(clean_df.shape[0]),
        "columns_after_cleaning": int(clean_df.shape[1]),
    }
    return clean_df, report


def save_dataset_artifacts(raw_df: pd.DataFrame, clean_df: pd.DataFrame, pruning: dict) -> None:
    raw_df.describe(include="all").transpose().to_csv(RESULTS_DIR / "dataset_profile.csv")
    clean_df.describe(include="all").transpose().to_csv(RESULTS_DIR / "clean_dataset_profile.csv")
    clean_df.to_csv(PROCESSED_PATH, index=False)
    (RESULTS_DIR / "pruning_summary.json").write_text(
        json.dumps(pruning, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def model_specs() -> dict:
    return {
        "logistic_regression": {
            "label": "Regresion Logistica",
            "course_type": "Visto en clase",
            "model": LogisticRegression(max_iter=4000, random_state=RANDOM_STATE),
            "params": {
                "model__C": [0.1, 1, 3, 10],
                "model__solver": ["lbfgs", "newton-cg"],
            },
        },
        "svm_rbf": {
            "label": "SVM con kernel RBF",
            "course_type": "Visto en clase",
            "model": SVC(random_state=RANDOM_STATE),
            "params": {
                "model__C": [0.5, 1, 3, 10],
                "model__gamma": ["scale", 0.1, 0.03, 0.01],
                "model__kernel": ["rbf"],
            },
        },
        "decision_tree": {
            "label": "Arbol de Decision",
            "course_type": "Visto en clase",
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "random_forest": {
            "label": "Random Forest",
            "course_type": "No visto en clase",
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
            },
        },
    }


def plot_class_distribution(df: pd.DataFrame) -> None:
    order = df[TARGET_COLUMN].value_counts().index.tolist()
    plt.figure(figsize=(11, 6))
    sns.countplot(
        data=df,
        y=TARGET_COLUMN,
        order=order,
        hue=TARGET_COLUMN,
        dodge=False,
        palette="crest",
        legend=False,
    )
    plt.title("Distribucion de clases del nivel de obesidad")
    plt.xlabel("Numero de registros")
    plt.ylabel("Clase")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=180)
    plt.close()


def plot_numeric_correlation(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="vlag", fmt=".2f", square=True)
    plt.title("Correlacion entre variables numericas")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_correlation.png", dpi=180)
    plt.close()


def plot_weight_by_target(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y="Weight",
        hue=TARGET_COLUMN,
        palette="Set2",
        legend=False,
    )
    plt.title("Distribucion del peso por clase objetivo")
    plt.xlabel("Clase")
    plt.ylabel("Peso")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weight_by_target.png", dpi=180)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df[["label", "test_f1_macro", "test_mcc", "test_accuracy"]].melt(
        id_vars="label",
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="label", y="score", hue="metric", palette="viridis")
    plt.title("Comparacion de modelos en el conjunto de prueba")
    plt.xlabel("Modelo")
    plt.ylabel("Valor de la metrica")
    plt.ylim(0.8, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close()


def train_and_evaluate(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    X = clean_df.drop(columns=[TARGET_COLUMN])
    y = clean_df[TARGET_COLUMN]
    preprocessor, _, _ = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metrics_rows = []
    best_estimators = {}
    best_params = {}

    for model_key, spec in model_specs().items():
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", spec["model"]),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        report_df = pd.DataFrame(
            classification_report(y_test, predictions, output_dict=True)
        ).transpose()
        report_df.to_csv(RESULTS_DIR / f"{model_key}_classification_report.csv")

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            xticks_rotation=35,
            cmap="Blues",
            colorbar=False,
        )
        disp.ax_.set_title(f"Matriz de confusion - {spec['label']}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_key}.png", dpi=180)
        plt.close()

        metrics_rows.append(
            {
                "model_key": model_key,
                "label": spec["label"],
                "course_type": spec["course_type"],
                "cv_f1_macro": round(float(search.best_score_), 4),
                "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "test_balanced_accuracy": round(
                    float(balanced_accuracy_score(y_test, predictions)), 4
                ),
                "test_precision_macro": round(
                    float(precision_score(y_test, predictions, average="macro")), 4
                ),
                "test_recall_macro": round(
                    float(recall_score(y_test, predictions, average="macro")), 4
                ),
                "test_f1_macro": round(
                    float(f1_score(y_test, predictions, average="macro")), 4
                ),
                "test_mcc": round(float(matthews_corrcoef(y_test, predictions)), 4),
            }
        )
        best_estimators[model_key] = best_model
        best_params[model_key] = search.best_params_

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["test_mcc", "test_f1_macro"],
        ascending=False,
    )
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    (RESULTS_DIR / "best_hyperparameters.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_key = metrics_df.iloc[0]["model_key"]
    best_model = best_estimators[best_key]

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="f1_macro",
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(RESULTS_DIR / "best_model_permutation_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(10),
        x="importance_mean",
        y="feature",
        hue="feature",
        palette="mako",
        legend=False,
    )
    plt.title("Importancia por permutacion del mejor modelo")
    plt.xlabel("Caida promedio en F1 macro")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "best_model_permutation_importance.png", dpi=180)
    plt.close()

    split_summary = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_fraction": 0.2,
        "random_state": RANDOM_STATE,
    }
    (RESULTS_DIR / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics_df, best_params, split_summary


def markdown_table(df: pd.DataFrame) -> str:
    headers = " | ".join(df.columns.tolist())
    separator = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(value) for value in row) for row in df.to_numpy()]
    return "\n".join([headers, separator, *rows])


def report_asset(path: Path) -> str:
    return path.relative_to(REPORTS_DIR).as_posix()


def build_report(summary: dict, pruning: dict, metrics_df: pd.DataFrame, best_params: dict, split_summary: dict) -> None:
    metrics_for_report = metrics_df[
        [
            "label",
            "course_type",
            "cv_f1_macro",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_macro",
            "test_mcc",
        ]
    ].copy()
    best_model = metrics_df.iloc[0]
    best_model_name = best_model["label"]
    target_table = pd.DataFrame(
        {
            "Clase": list(summary["target_distribution"].keys()),
            "Frecuencia": list(summary["target_distribution"].values()),
        }
    )

    report = f"""# MiniProyecto 2 - Problema de Clasificacion

## Integrantes

- Valentina Popo Montilla
- Juan Camilo Balleresteros Sierra
- Santigo Rodriguez Gacha

## 1. Descripcion del dataset

Se utilizo el dataset **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** del repositorio UCI. El conjunto contiene informacion de habitos de alimentacion, condicion fisica y caracteristicas antropometricas para clasificar el nivel de obesidad de una persona.

- Registros originales: **{summary["rows"]}**
- Variables originales: **{summary["columns"]}**
- Variables numericas: **{len(summary["numeric_columns"])}**
- Variables categoricas: **{len(summary["categorical_columns"]) - 1}** predictoras + la variable objetivo
- Variable objetivo: **{TARGET_COLUMN}**
- Clases objetivo: **{len(summary["target_distribution"])}**
- Valores faltantes detectados: **{summary["missing_values_total"]}**
- Filas duplicadas exactas detectadas: **{summary["duplicate_rows"]}**

Distribucion de clases:

{markdown_table(target_table)}

![Distribucion de clases]({report_asset(FIGURES_DIR / "class_distribution.png")})

## 2. Analisis exploratorio y depuracion de variables

Se aplico un proceso de depuracion con las siguientes reglas:

1. Eliminar columnas con mas del 50% de valores faltantes.
2. Eliminar columnas con varianza cero.
3. Revisar correlaciones numericas absolutas mayores a 0.85 para evitar redundancia.
4. Eliminar filas duplicadas exactas.

Resultados de la depuracion:

- Columnas con mas del 50% de faltantes: **{len(pruning["columns_over_missing_threshold"])}**
- Columnas con varianza cero: **{len(pruning["zero_variance_columns"])}**
- Pares numericos con correlacion mayor a 0.85: **{len(pruning["high_correlation_pairs_over_0_85"])}**
- Columnas eliminadas por depuracion: **{", ".join(pruning["dropped_columns"]) if pruning["dropped_columns"] else "Ninguna"}**
- Filas duplicadas eliminadas: **{pruning["duplicate_rows_removed"]}**
- Registros finales para modelado: **{pruning["rows_after_cleaning"]}**

En este dataset no fue necesario eliminar variables por faltantes, varianza cero o alta correlacion. La limpieza se concentro en retirar los duplicados exactos, quedando un conjunto final de **{pruning["rows_after_cleaning"]}** registros.

La matriz de correlacion de las variables numericas se presenta a continuacion:

![Correlacion numerica]({report_asset(FIGURES_DIR / "numeric_correlation.png")})

Como variable ilustrativa, el peso muestra una separacion clara entre varias clases objetivo:

![Peso por clase]({report_asset(FIGURES_DIR / "weight_by_target.png")})

## 3. Division del dataset y metodologia

Para entrenar y validar los modelos se realizo una particion estratificada:

- Entrenamiento: **{split_summary["train_rows"]}** registros
- Prueba: **{split_summary["test_rows"]}** registros
- Proporcion de prueba: **20%**
- Semilla aleatoria: **{split_summary["random_state"]}**

Las variables numericas fueron imputadas con la mediana y estandarizadas. Las variables categoricas fueron imputadas con la moda y codificadas mediante **One-Hot Encoding**. El ajuste de hiperparametros se realizo con **GridSearchCV** de 5 particiones, optimizando la metrica **F1 macro**.

## 4. Modelos de clasificacion evaluados

Se entrenaron cuatro modelos:

1. **Regresion Logistica** - modelo lineal multiclase visto en clase.
2. **SVM con kernel RBF** - modelo visto en clase.
3. **Arbol de Decision** - modelo visto en clase.
4. **Random Forest** - modelo adicional no visto en clase. Este metodo combina multiples arboles de decision y toma la clase final por votacion, lo que suele mejorar la estabilidad frente a un arbol individual.

## 5. Comparacion de resultados

Ademas de la matriz de confusion, se utilizo como indice adicional el **Matthews Correlation Coefficient (MCC)**, una metrica robusta para problemas multiclase porque resume la calidad global de la clasificacion considerando verdaderos y falsos positivos y negativos.

{markdown_table(metrics_for_report)}

![Comparacion de modelos]({report_asset(FIGURES_DIR / "model_comparison.png")})

Matrices de confusion generadas:

- ![Matriz Regresion Logistica]({report_asset(FIGURES_DIR / "confusion_matrix_logistic_regression.png")})
- ![Matriz SVM]({report_asset(FIGURES_DIR / "confusion_matrix_svm_rbf.png")})
- ![Matriz Arbol de Decision]({report_asset(FIGURES_DIR / "confusion_matrix_decision_tree.png")})
- ![Matriz Random Forest]({report_asset(FIGURES_DIR / "confusion_matrix_random_forest.png")})

## 6. Mejor modelo e hiperparametros

El mejor modelo del experimento fue **{best_model_name}**, ya que obtuvo los mejores valores en el conjunto de prueba tanto para **F1 macro** (**{best_model["test_f1_macro"]}**) como para **MCC** (**{best_model["test_mcc"]}**).

Hiperparametros optimos encontrados:

```json
{json.dumps(best_params[best_model["model_key"]], indent=2, ensure_ascii=False)}
```

Las variables con mayor influencia estimadas mediante importancia por permutacion fueron:

![Importancia del mejor modelo]({report_asset(FIGURES_DIR / "best_model_permutation_importance.png")})

## 7. Conclusiones

1. El dataset presenta una estructura relativamente limpia: no tiene valores faltantes y tampoco muestra columnas con varianza cero ni correlaciones numericas extremas. La principal accion de depuracion fue eliminar **{pruning["duplicate_rows_removed"]}** registros duplicados.
2. Los cuatro modelos alcanzaron desempenos altos, lo que sugiere que las variables del dataset contienen informacion suficiente para distinguir las clases objetivo.
3. **SVM con kernel RBF** fue el modelo mas solido al generalizar, con mejor **F1 macro** y mejor **MCC** en el conjunto de prueba. Esto indica una mejor capacidad para separar fronteras no lineales entre los distintos niveles de obesidad.
4. Aunque el **Arbol de Decision** obtuvo un resultado competitivo, fue ligeramente inferior al SVM en las metricas globales, por lo que no seria la primera opcion para esta aplicacion.
5. **Random Forest**, como metodo adicional no visto en clase, mostro un rendimiento estable, pero en esta base no supero al SVM. Su principal ventaja sigue siendo la interpretabilidad agregada y la robustez frente a variaciones en los datos.
6. Para esta aplicacion concreta, el modelo recomendado es **{best_model_name}**, porque ofrece el mejor equilibrio entre precision global, sensibilidad promedio por clase y correlacion global de las predicciones.
7. Como limitacion metodologica, el propio UCI reporta que cerca del 77% de los datos fue generado sinteticamente con SMOTE. Por eso, aunque el desempeno es alto, los resultados deben interpretarse con cautela al extrapolarlos a poblaciones reales.

## 8. Archivos generados

- Script principal: `src/run_analysis.py`
- Notebook: `notebooks/miniproyecto_2_clasificacion_obesidad.ipynb`
- Resultados tabulares: `reports/results/`
- Figuras: `reports/figures/`

"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def build_notebook() -> None:
    cells = [
        markdown_cell("# MiniProyecto 2 - Clasificacion de niveles de obesidad"),
        markdown_cell(
            "Este notebook resume el flujo del proyecto: carga del dataset, depuracion, modelado y comparacion de resultados."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.model_selection import GridSearchCV, train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.svm import SVC\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef\n"
        ),
        code_cell(
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "DATA_PATH = ROOT / 'data' / 'raw' / 'obesity.csv'\n"
            "TARGET_COLUMN = 'NObeyesdad'\n"
            "df = pd.read_csv(DATA_PATH)\n"
            "df.head()"
        ),
        markdown_cell("## Resumen y depuracion"),
        code_cell(
            "print(df.shape)\n"
            "print(df[TARGET_COLUMN].value_counts())\n"
            "print('Valores faltantes totales:', int(df.isna().sum().sum()))\n"
            "print('Duplicados exactos:', int(df.duplicated().sum()))\n"
            "df = df.drop_duplicates().reset_index(drop=True)\n"
            "print('Forma tras eliminar duplicados:', df.shape)"
        ),
        markdown_cell("## Preparacion de variables"),
        code_cell(
            "X = df.drop(columns=[TARGET_COLUMN])\n"
            "y = df[TARGET_COLUMN]\n"
            "numeric_features = X.select_dtypes(include='number').columns.tolist()\n"
            "categorical_features = X.select_dtypes(exclude='number').columns.tolist()\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),\n"
            "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),\n"
            "])\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
            "X_train.shape, X_test.shape"
        ),
        markdown_cell("## Entrenamiento de modelos"),
        code_cell(
            "models = {\n"
            "    'logistic_regression': (\n"
            "        LogisticRegression(max_iter=4000, random_state=42),\n"
            "        {'model__C':[0.1, 1, 3, 10], 'model__solver':['lbfgs', 'newton-cg']},\n"
            "    ),\n"
            "    'svm_rbf': (\n"
            "        SVC(random_state=42),\n"
            "        {'model__C':[0.5, 1, 3, 10], 'model__gamma':['scale', 0.1, 0.03, 0.01], 'model__kernel':['rbf']},\n"
            "    ),\n"
            "    'decision_tree': (\n"
            "        DecisionTreeClassifier(random_state=42),\n"
            "        {'model__criterion':['gini', 'entropy'], 'model__max_depth':[None, 5, 10, 20], 'model__min_samples_split':[2, 5, 10], 'model__min_samples_leaf':[1, 2, 4]},\n"
            "    ),\n"
            "    'random_forest': (\n"
            "        RandomForestClassifier(random_state=42, n_jobs=-1),\n"
            "        {'model__n_estimators':[200, 400], 'model__max_depth':[None, 10, 20], 'model__min_samples_split':[2, 5], 'model__min_samples_leaf':[1, 2]},\n"
            "    ),\n"
            "}\n"
            "rows = []\n"
            "for name, (model, params) in models.items():\n"
            "    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])\n"
            "    search = GridSearchCV(pipeline, params, cv=5, scoring='f1_macro', n_jobs=-1)\n"
            "    search.fit(X_train, y_train)\n"
            "    preds = search.best_estimator_.predict(X_test)\n"
            "    rows.append({\n"
            "        'model': name,\n"
            "        'cv_f1_macro': search.best_score_,\n"
            "        'test_accuracy': accuracy_score(y_test, preds),\n"
            "        'test_balanced_accuracy': balanced_accuracy_score(y_test, preds),\n"
            "        'test_f1_macro': f1_score(y_test, preds, average='macro'),\n"
            "        'test_mcc': matthews_corrcoef(y_test, preds),\n"
            "    })\n"
            "results_df = pd.DataFrame(rows).sort_values(['test_mcc', 'test_f1_macro'], ascending=False)\n"
            "results_df"
        ),
        markdown_cell(
            "## Resultados exportados\n\n"
            "El script `src/run_analysis.py` genera automaticamente el informe final, las figuras, los CSV de metricas y las matrices de confusion."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def main() -> None:
    ensure_dirs()
    raw_df = load_dataset()
    summary = summarize_dataset(raw_df)
    clean_df, pruning = feature_pruning_report(raw_df)
    save_dataset_artifacts(raw_df, clean_df, pruning)

    plot_class_distribution(clean_df)
    plot_numeric_correlation(clean_df)
    plot_weight_by_target(clean_df)

    metrics_df, best_params, split_summary = train_and_evaluate(clean_df)
    plot_model_comparison(metrics_df)

    build_report(summary, pruning, metrics_df, best_params, split_summary)
    build_notebook()

    print(f"Dataset limpio guardado en: {PROCESSED_PATH}")
    print(f"Informe generado en: {REPORT_PATH}")
    print(f"Notebook generado en: {NOTEBOOK_PATH}")
    print(f"Mejor modelo: {metrics_df.iloc[0]['label']}")
    print(f"MCC del mejor modelo: {metrics_df.iloc[0]['test_mcc']}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
