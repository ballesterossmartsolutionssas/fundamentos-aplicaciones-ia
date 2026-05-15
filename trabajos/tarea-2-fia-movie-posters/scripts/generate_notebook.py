from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tarea_2_fia_movie_posters_colab.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


cells = [
    md(
        """
        # Tarea 2 FIA - Multi-label Movie Poster Classification

        **Materia:** Fundamentos y Aplicaciones de Inteligencia Artificial

        **Objetivo:** clasificar generos de peliculas a partir de caratulas usando una CNN original.
        """
    ),
    md(
        """
        ## Idea central

        Esta tarea es **multi-label**: una pelicula puede tener varios generos simultaneamente. Por eso se usa salida `sigmoid` y perdida `binary_crossentropy`, no `softmax`.
        """
    ),
    code(
        """
        !pip -q install scikit-learn seaborn
        """
    ),
    code(
        """
        import os
        import re
        import random
        import shutil
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import classification_report, roc_curve, auc

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32
        EPOCHS = 15
        MIN_GENRE_COUNT = 120

        ROOT = Path("/content")
        DATA_ROOT = ROOT / "Movies-Poster_Dataset"
        OUTPUT_DIR = ROOT / "movie_poster_outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)
        """
    ),
    md("## 1. Descargar dataset"),
    code(
        """
        if DATA_ROOT.exists():
            shutil.rmtree(DATA_ROOT)
        !git clone --depth 1 https://github.com/laxmimerit/Movies-Poster_Dataset.git /content/Movies-Poster_Dataset
        """
    ),
    md("## 2. Leer CSV y encontrar imagenes"),
    code(
        """
        csv_candidates = list(DATA_ROOT.rglob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError("No se encontro CSV en el dataset")

        csv_path = csv_candidates[0]
        print("CSV:", csv_path)

        df = pd.read_csv(csv_path, encoding="latin-1")
        print(df.columns.tolist())
        df.head()
        """
    ),
    code(
        """
        def pick_column(columns, patterns):
            for pattern in patterns:
                for col in columns:
                    if re.search(pattern, col, flags=re.IGNORECASE):
                        return col
            return None

        genre_col = pick_column(df.columns, [r"genre"])
        imdb_col = pick_column(df.columns, [r"imdb", r"imdbid", r"imdb id"])
        title_col = pick_column(df.columns, [r"title"])

        if genre_col is None:
            raise ValueError("No se encontro columna de generos")

        print("Columna generos:", genre_col)
        print("Columna imdb:", imdb_col)
        print("Columna titulo:", title_col)

        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(DATA_ROOT.rglob(ext))

        image_by_stem = {p.stem.lower(): p for p in image_files}
        print("Imagenes encontradas:", len(image_files))

        def image_for_row(row):
            candidates = []
            if imdb_col is not None and pd.notna(row[imdb_col]):
                raw = str(row[imdb_col]).strip()
                candidates.extend([raw.lower(), raw.replace("tt", "").lower(), f"tt{raw}".lower()])
            if title_col is not None and pd.notna(row[title_col]):
                title = re.sub(r"[^a-zA-Z0-9]+", "", str(row[title_col])).lower()
                candidates.append(title)
            for key in candidates:
                if key in image_by_stem:
                    return str(image_by_stem[key])
            return None

        df["filepath"] = df.apply(image_for_row, axis=1)
        df = df.dropna(subset=["filepath", genre_col]).copy()
        print("Registros con imagen y genero:", len(df))
        df[[genre_col, "filepath"]].head()
        """
    ),
    md("## 3. Preparar etiquetas multi-label"),
    code(
        """
        def parse_genres(value):
            if pd.isna(value):
                return []
            parts = re.split(r"\\||,|/", str(value))
            return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]

        df["genres_list"] = df[genre_col].apply(parse_genres)
        all_genres = pd.Series([g for genres in df["genres_list"] for g in genres])
        genre_counts = all_genres.value_counts()
        display(genre_counts)

        selected_genres = genre_counts[genre_counts >= MIN_GENRE_COUNT].index.tolist()
        print("Generos seleccionados:", selected_genres)

        df["genres_list"] = df["genres_list"].apply(lambda values: [g for g in values if g in selected_genres])
        df = df[df["genres_list"].map(len) > 0].reset_index(drop=True)

        mlb = MultiLabelBinarizer(classes=selected_genres)
        y = mlb.fit_transform(df["genres_list"])
        label_cols = [f"genre_{g}" for g in mlb.classes_]
        labels_df = pd.DataFrame(y, columns=label_cols)
        data = pd.concat([df[["filepath"]].reset_index(drop=True), labels_df], axis=1)

        print("Dataset final:", data.shape)
        data.head()
        """
    ),
    md("## 4. Division train/validation/test"),
    code(
        """
        train_df, temp_df = train_test_split(data, test_size=0.30, random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)

        print("Train:", train_df.shape)
        print("Validation:", val_df.shape)
        print("Test:", test_df.shape)
        """
    ),
    md("## 5. ImageDataGenerator desde dataframe"),
    code(
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            zoom_range=0.12,
            width_shift_range=0.08,
            height_shift_range=0.08,
            horizontal_flip=True,
        )

        eval_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_dataframe(
            train_df,
            x_col="filepath",
            y_col=label_cols,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="raw",
            shuffle=True,
            seed=SEED,
        )

        val_gen = eval_datagen.flow_from_dataframe(
            val_df,
            x_col="filepath",
            y_col=label_cols,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="raw",
            shuffle=False,
        )

        test_gen = eval_datagen.flow_from_dataframe(
            test_df,
            x_col="filepath",
            y_col=label_cols,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="raw",
            shuffle=False,
        )
        """
    ),
    md("## 6. CNN original propuesta"),
    code(
        """
        def build_cnn(input_shape, n_labels):
            model = keras.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                layers.Conv2D(128, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                layers.Conv2D(256, 3, padding="same", activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.40),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.30),
                layers.Dense(n_labels, activation="sigmoid"),
            ])
            return model

        model = build_cnn((*IMG_SIZE, 3), len(label_cols))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                keras.metrics.AUC(name="auc", multi_label=True),
            ],
        )
        model.summary()
        """
    ),
    md("## 7. Entrenamiento"),
    code(
        """
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_auc", patience=4, mode="max", restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(OUTPUT_DIR / "best_movie_poster_cnn.keras", monitor="val_auc", mode="max", save_best_only=True),
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
        )

        pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
        """
    ),
    code(
        """
        hist = pd.DataFrame(history.history)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(hist["binary_accuracy"], label="train")
        ax[0].plot(hist["val_binary_accuracy"], label="val")
        ax[0].set_title("Binary accuracy")
        ax[0].legend()
        ax[1].plot(hist["loss"], label="train")
        ax[1].plot(hist["val_loss"], label="val")
        ax[1].set_title("Loss")
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
        plt.show()
        """
    ),
    md("## 8. Evaluacion"),
    code(
        """
        test_gen.reset()
        y_true = test_df[label_cols].to_numpy()
        y_prob = model.predict(test_gen, verbose=1)
        y_pred = (y_prob >= 0.35).astype(int)

        report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
        display(report_df)
        """
    ),
    code(
        """
        roc_rows = []
        plt.figure(figsize=(10, 8))
        for i, genre in enumerate(mlb.classes_):
            if len(np.unique(y_true[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            roc_rows.append({"genre": genre, "roc_auc": roc_auc})
            plt.plot(fpr, tpr, lw=1.5, label=f"{genre} AUC={roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title("ROC-AUC por genero")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "roc_auc_by_genre.png", dpi=160)
        plt.show()

        pd.DataFrame(roc_rows).sort_values("roc_auc", ascending=False).to_csv(OUTPUT_DIR / "roc_auc_scores.csv", index=False)
        """
    ),
    md("## 9. Probar una caratula externa"),
    code(
        """
        from google.colab import files
        from tensorflow.keras.utils import load_img, img_to_array

        uploaded = files.upload()
        image_path = next(iter(uploaded.keys()))

        img = load_img(image_path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        batch = np.expand_dims(arr, axis=0)
        probs = model.predict(batch)[0]
        top_idx = probs.argsort()[::-1][:5]

        plt.imshow(img)
        plt.axis("off")
        plt.show()

        print("Top 5 generos predichos:")
        for idx in top_idx:
            print(f"{mlb.classes_[idx]}: {probs[idx]:.4f}")
        """
    ),
    md(
        """
        ## Nota para entrega

        Descargar o compartir este notebook ejecutado. Para la sustentacion, usar la seccion final con una caratula externa y comparar los primeros 5 generos predichos con los generos reales de IMDB.
        """
    ),
]

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(
    json.dumps(
        {
            "cells": cells,
            "metadata": {
                "colab": {"provenance": []},
                "kernelspec": {
                    "display_name": "Python 3",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"Notebook generado: {NOTEBOOK}")
