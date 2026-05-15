# Tarea 2 FIA - Movie Poster Multi-label Classification

Materia: **Fundamentos y Aplicaciones de Inteligencia Artificial**

Tarea opcional: clasificacion multi-label de generos de peliculas usando caratulas entre 1985 y 2015.

Dataset: https://github.com/laxmimerit/Movies-Poster_Dataset.git

## Entregables

- `notebooks/tarea_2_fia_movie_posters_colab.ipynb`: notebook principal para Google Colab.
- `reports/informe_tarea_2_fia_movie_posters.md`: informe base del proceso.
- `scripts/generate_notebook.py`: script para regenerar el notebook.

## Que implementa

- Clonacion del dataset desde GitHub.
- Limpieza del CSV y asociacion de imagenes con sus generos.
- Procesamiento con `ImageDataGenerator.flow_from_dataframe`.
- CNN original para clasificacion multi-label.
- Evaluacion con classification report y ROC-AUC por genero.
- Celda para subir una caratula externa y mostrar los primeros 5 generos predichos.

## Uso

Abrir el notebook en Google Colab y ejecutar las celdas en orden. Los resultados se exportan en `/content/movie_poster_outputs`.
