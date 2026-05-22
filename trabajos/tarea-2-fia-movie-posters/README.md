# Tarea 2 FIA - Movie Poster Multi-label Classification

Materia: **Fundamentos y Aplicaciones de Inteligencia Artificial**

Tarea opcional: clasificacion multi-label de generos de peliculas usando caratulas entre 1985 y 2015.

Dataset: https://github.com/laxmimerit/Movies-Poster_Dataset.git

## Entregables

- `notebooks/tarea_2_fia_movie_posters_colab.ipynb`: notebook principal para Google Colab.
- `reports/informe_tarea_2_fia_movie_posters.md`: informe base del proceso.
- `reports/informe_tarea_2_fia_movie_posters.pdf`: informe exportado a PDF.
- `execution/final-colab-run`: resultados finales de la ejecucion en Google Colab con GPU T4.
- `scripts/generate_notebook.py`: script para regenerar el notebook.

## Que implementa

- Clonacion del dataset desde GitHub.
- Limpieza del CSV y asociacion de imagenes con sus generos.
- Procesamiento con `ImageDataGenerator.flow_from_dataframe`.
- CNN original para clasificacion multi-label.
- Evaluacion con classification report y ROC-AUC por genero.
- Celda para subir una caratula externa y mostrar los primeros 5 generos predichos.
- Celda final para descargar un ZIP con metricas, graficas y modelo entrenado.

## Uso

Abrir el notebook en Google Colab y ejecutar las celdas en orden. Los resultados se exportan en `/content/movie_poster_outputs` y se descargan al final como ZIP.

## Resultados finales

- Epocas: 10
- `val_binary_accuracy`: 0.8895
- `val_auc`: 0.6830
- `val_loss`: 0.2919
- ROC-AUC promedio por genero: 0.6756
