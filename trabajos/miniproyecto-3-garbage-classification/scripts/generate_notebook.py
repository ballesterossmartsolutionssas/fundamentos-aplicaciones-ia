from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "miniproyecto_3_garbage_classification_colab.ipynb"


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
        # MiniProyecto 3 - Garbage Classification

        **Materia:** Fundamentos y Aplicaciones de Inteligencia Artificial

        **Dataset:** Garbage Classification, todas las categorias.

        **Integrantes:** Valentina Popo Montilla, Santiago Rodriguez Gacha y Juan Camilo Ballesteros.
        """
    ),
    md(
        """
        ## Objetivo

        Realizar clasificacion de imagenes usando tres arquitecturas de redes neuronales:

        1. MLP.
        2. CNN original.
        3. Transfer learning con MobileNetV2.

        Para cada modelo se genera matriz de confusion, classification report y ROC-AUC.
        """
    ),
    code(
        """
        !pip -q install kagglehub scikit-learn seaborn
        """
    ),
    code(
        """
        import os
        import json
        import shutil
        import random
        import zipfile
        from pathlib import Path

        import kagglehub
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, accuracy_score
        from sklearn.preprocessing import label_binarize

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32
        EPOCHS = 12
        DATASET_SLUG = "mostafaabla/garbage-classification"
        ROOT = Path("/content")
        RAW_DIR = ROOT / "garbage_raw"
        SPLIT_DIR = ROOT / "garbage_split"
        OUTPUT_DIR = ROOT / "garbage_outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)
        """
    ),
    md("## 1. Descargar dataset desde KaggleHub"),
    code(
        """
        dataset_path = kagglehub.dataset_download(DATASET_SLUG)
        RAW_DIR = Path(dataset_path)
        print("Path to dataset files:", RAW_DIR)
        """
    ),
    code(
        """
        def find_class_root(base_dir: Path) -> Path:
            candidates = []
            for path in base_dir.rglob("*"):
                if path.is_dir():
                    child_dirs = [p for p in path.iterdir() if p.is_dir()]
                    image_count = sum(1 for _ in path.rglob("*.jpg")) + sum(1 for _ in path.rglob("*.png")) + sum(1 for _ in path.rglob("*.jpeg"))
                    if len(child_dirs) >= 2 and image_count > 0:
                        candidates.append((path, len(child_dirs), image_count))
            if not candidates:
                raise FileNotFoundError("No se encontro una carpeta con subcarpetas de clases.")
            return sorted(candidates, key=lambda item: (item[1], item[2]), reverse=True)[0][0]

        CLASS_ROOT = find_class_root(RAW_DIR)
        classes = sorted([p.name for p in CLASS_ROOT.iterdir() if p.is_dir()])
        print("Carpeta de clases:", CLASS_ROOT)
        print("Numero de clases:", len(classes))
        print(classes)
        """
    ),
    md("## 2. Particion train/validation/test"),
    code(
        """
        def split_dataset(class_root: Path, split_dir: Path, train_ratio=0.70, val_ratio=0.15):
            if split_dir.exists():
                shutil.rmtree(split_dir)
            for split in ["train", "val", "test"]:
                (split_dir / split).mkdir(parents=True, exist_ok=True)

            for class_dir in sorted([p for p in class_root.iterdir() if p.is_dir()]):
                images = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                    images.extend(class_dir.glob(ext))
                random.shuffle(images)
                n = len(images)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                partitions = {
                    "train": images[:n_train],
                    "val": images[n_train:n_train + n_val],
                    "test": images[n_train + n_val:],
                }
                for split, files_ in partitions.items():
                    out_class = split_dir / split / class_dir.name
                    out_class.mkdir(parents=True, exist_ok=True)
                    for file in files_:
                        shutil.copy2(file, out_class / file.name)

        split_dataset(CLASS_ROOT, SPLIT_DIR)

        for split in ["train", "val", "test"]:
            count = sum(1 for _ in (SPLIT_DIR / split).rglob("*") if _.is_file())
            print(split, count)
        """
    ),
    code(
        """
        train_ds = keras.utils.image_dataset_from_directory(
            SPLIT_DIR / "train",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=True,
            seed=SEED,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            SPLIT_DIR / "val",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
        )
        test_ds = keras.utils.image_dataset_from_directory(
            SPLIT_DIR / "test",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
        )

        class_names = train_ds.class_names
        num_classes = len(class_names)
        print(class_names)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        test_ds = test_ds.cache().prefetch(AUTOTUNE)
        """
    ),
    md("## 3. Funciones de evaluacion"),
    code(
        """
        def get_true_and_pred(model, dataset):
            y_true = []
            y_prob = []
            for x_batch, y_batch in dataset:
                probs = model.predict(x_batch, verbose=0)
                y_prob.append(probs)
                y_true.append(y_batch.numpy())
            y_true = np.vstack(y_true)
            y_prob = np.vstack(y_prob)
            return y_true.argmax(axis=1), y_prob.argmax(axis=1), y_true, y_prob


        def evaluate_model(model, dataset, model_name):
            y_true_idx, y_pred_idx, y_true, y_prob = get_true_and_pred(model, dataset)

            report = classification_report(y_true_idx, y_pred_idx, target_names=class_names, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(OUTPUT_DIR / f"{model_name}_classification_report.csv")
            display(report_df)

            cm = confusion_matrix(y_true_idx, y_pred_idx)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.title(f"Matriz de confusion - {model_name}")
            plt.xlabel("Prediccion")
            plt.ylabel("Real")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{model_name}_confusion_matrix.png", dpi=160)
            plt.show()

            y_true_bin = label_binarize(y_true_idx, classes=np.arange(num_classes))
            plt.figure(figsize=(10, 8))
            roc_rows = []
            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                roc_rows.append({"class": class_name, "roc_auc": roc_auc})
                plt.plot(fpr, tpr, lw=1.5, label=f"{class_name} AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.title(f"ROC-AUC por clase - {model_name}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{model_name}_roc_auc.png", dpi=160)
            plt.show()

            metrics = {
                "model": model_name,
                "accuracy": accuracy_score(y_true_idx, y_pred_idx),
                "macro_f1": f1_score(y_true_idx, y_pred_idx, average="macro"),
            }
            pd.DataFrame(roc_rows).to_csv(OUTPUT_DIR / f"{model_name}_roc_auc_scores.csv", index=False)
            return metrics


        def plot_history(history, model_name):
            hist = pd.DataFrame(history.history)
            hist.to_csv(OUTPUT_DIR / f"{model_name}_history.csv", index=False)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(hist["accuracy"], label="train")
            ax[0].plot(hist["val_accuracy"], label="val")
            ax[0].set_title(f"Accuracy - {model_name}")
            ax[0].legend()
            ax[1].plot(hist["loss"], label="train")
            ax[1].plot(hist["val_loss"], label="val")
            ax[1].set_title(f"Loss - {model_name}")
            ax[1].legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{model_name}_history.png", dpi=160)
            plt.show()
        """
    ),
    md("## 4. Modelo 1: MLP"),
    code(
        """
        mlp_model = keras.Sequential([
            layers.Input(shape=(*IMG_SIZE, 3)),
            layers.Rescaling(1./255),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ])

        mlp_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        mlp_model.summary()
        mlp_history = mlp_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        plot_history(mlp_history, "mlp")
        mlp_metrics = evaluate_model(mlp_model, test_ds, "mlp")
        """
    ),
    md("## 5. Modelo 2: CNN original"),
    code(
        """
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
        ])

        cnn_model = keras.Sequential([
            layers.Input(shape=(*IMG_SIZE, 3)),
            layers.Rescaling(1./255),
            augmentation,
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
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation="softmax"),
        ])

        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        cnn_model.summary()
        cnn_history = cnn_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        plot_history(cnn_history, "cnn")
        cnn_metrics = evaluate_model(cnn_model, test_ds, "cnn")
        """
    ),
    md("## 6. Modelo 3: Transfer learning con MobileNetV2"),
    code(
        """
        base_model = keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(*IMG_SIZE, 3))
        x = augmentation(inputs)
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        transfer_model = keras.Model(inputs, outputs)

        transfer_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        transfer_model.summary()
        transfer_history = transfer_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        plot_history(transfer_history, "transfer_mobilenetv2")
        transfer_metrics = evaluate_model(transfer_model, test_ds, "transfer_mobilenetv2")
        """
    ),
    md("## 7. Comparacion final"),
    code(
        """
        metrics_df = pd.DataFrame([mlp_metrics, cnn_metrics, transfer_metrics]).sort_values("macro_f1", ascending=False)
        metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
        display(metrics_df)

        plt.figure(figsize=(8, 4))
        sns.barplot(data=metrics_df.melt(id_vars="model", value_vars=["accuracy", "macro_f1"]), x="model", y="value", hue="variable")
        plt.ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        plt.title("Comparacion de arquitecturas")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
        plt.show()

        transfer_model.save(OUTPUT_DIR / "best_transfer_mobilenetv2.keras")
        """
    ),
    md("## 8. Clasificar una imagen externa"),
    code(
        """
        from google.colab import files
        from tensorflow.keras.utils import load_img, img_to_array

        uploaded = files.upload()
        image_path = next(iter(uploaded.keys()))

        img = load_img(image_path, target_size=IMG_SIZE)
        arr = img_to_array(img)
        batch = np.expand_dims(arr, axis=0)
        probs = transfer_model.predict(batch)[0]
        top_idx = probs.argsort()[::-1][:5]

        plt.imshow(img)
        plt.axis("off")
        plt.show()

        print("Top 5 predicciones:")
        for idx in top_idx:
            print(f"{class_names[idx]}: {probs[idx]:.4f}")
        """
    ),
    md(
        """
        ## Observaciones finales

        Al terminar la ejecucion, descargar la carpeta `/content/garbage_outputs` si se necesitan los resultados como evidencia. El modelo recomendado para sustentar es MobileNetV2, porque aprovecha transferencia de aprendizaje y normalmente supera a la MLP y a la CNN entrenada desde cero.
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
