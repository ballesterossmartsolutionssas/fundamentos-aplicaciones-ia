# MiniProyecto 3 - Garbage Classification

Materia: **Fundamentos y Aplicaciones de Inteligencia Artificial**

Dataset asignado: **Garbage Classification - Todas las categorias**

Fuente: https://www.kaggle.com/mostafaabla/garbage-classification

## Integrantes

- Valentina Popo Montilla
- Santiago Rodriguez Gacha
- Juan Camilo Ballesteros

## Entregables

- `notebooks/miniproyecto_3_garbage_classification_colab.ipynb`: notebook principal para Google Colab.
- `reports/informe_miniproyecto_3_garbage.md`: informe base con metodologia, modelos, metricas y conclusiones.
- `scripts/generate_notebook.py`: script que genera el notebook en caso de necesitar regenerarlo.

## Modelos implementados

1. MLP sobre imagenes redimensionadas y normalizadas.
2. CNN convolucional original.
3. Transfer learning con MobileNetV2.

## Como usar en Colab

1. Abrir el notebook `notebooks/miniproyecto_3_garbage_classification_colab.ipynb`.
2. Ejecutar las celdas en orden.
3. Subir `kaggle.json` cuando Colab lo solicite.
4. Revisar los resultados exportados en `/content/garbage_outputs`.

El notebook genera automaticamente matrices de confusion, classification report y curvas ROC-AUC para cada arquitectura.
