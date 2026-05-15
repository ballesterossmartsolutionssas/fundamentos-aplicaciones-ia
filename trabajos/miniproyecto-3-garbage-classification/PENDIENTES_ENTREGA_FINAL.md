# Pendientes para entrega final - MiniProyecto 3 Garbage Classification

Este proyecto ya tiene estructura, notebook, modelos, descarga con `kagglehub` y evidencia de una corrida local rapida. Para entregar formalmente al profesor todavia falta cerrar estos puntos.

## Falta hacer

1. Ejecutar el notebook completo en Google Colab con GPU.
2. Mantener `EPOCHS = 12` si el tiempo alcanza; si Colab se demora demasiado, usar `EPOCHS = 8`.
3. Esperar a que terminen los tres modelos: MLP, CNN y MobileNetV2.
4. Descargar o conservar la carpeta `/content/garbage_outputs`.
5. Actualizar el informe con los resultados finales reales:
   - Accuracy.
   - Macro F1.
   - Matrices de confusion.
   - Classification report.
   - ROC-AUC.
6. Exportar el informe final a PDF.
7. Subir al repo el PDF final y, si pesan poco, los resultados principales generados por Colab.

## Estado actual

- El dataset ya descarga correctamente con `kagglehub.dataset_download("mostafaabla/garbage-classification")`.
- El notebook principal esta en `notebooks/miniproyecto_3_garbage_classification_colab.ipynb`.
- Ya existe una corrida local rapida en `execution/quick-local-run`.
- Esa corrida rapida sirve como prueba de funcionamiento, pero no reemplaza la ejecucion full para entrega final.

## Recomendacion

Cuando se vaya a cerrar la entrega, correr el notebook en Colab con GPU y usar los resultados finales de MobileNetV2 como modelo principal para sustentar, comparandolo contra MLP y CNN.
