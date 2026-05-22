# Tarea 2 FIA - Clasificacion Multi-label de Generos de Peliculas

## Integrantes

- Valentina Popo Montilla
- Juan Camilo Balleresteros Sierra
- Santigo Rodriguez Gacha

## 1. Objetivo

Implementar una red neuronal convolucional original para clasificar generos de peliculas a partir de sus caratulas. La tarea se trabaja como **clasificacion multi-label**, porque una misma pelicula puede pertenecer a varios generos al mismo tiempo, por ejemplo drama, romance y comedia.

## 2. Dataset

Dataset usado: https://github.com/laxmimerit/Movies-Poster_Dataset.git

El dataset contiene caratulas de peliculas entre 1985 y 2015 y un archivo CSV con metadatos, incluyendo los generos asociados a cada pelicula. El notebook clona el repositorio, localiza automaticamente el CSV y relaciona cada fila con su imagen correspondiente.

## 3. Procesamiento de datos

El procesamiento implementado en el notebook realiza los siguientes pasos:

1. Clonacion del repositorio del dataset en Google Colab.
2. Lectura del archivo CSV y deteccion de columnas relevantes como genero, titulo e identificador IMDB.
3. Busqueda de archivos de imagen dentro del dataset.
4. Asociacion de cada registro con su caratula.
5. Separacion del campo de generos en una lista de etiquetas.
6. Conversion de generos a variables binarias con `MultiLabelBinarizer`.
7. Filtrado de generos con pocas muestras para mejorar estabilidad.
8. Division en entrenamiento, validacion y prueba.
9. Lectura de imagenes mediante `ImageDataGenerator.flow_from_dataframe`, como solicita la guia de la tarea.

## 4. Modelo CNN propuesto

La arquitectura propuesta es una CNN secuencial original compuesta por:

- Capas `Conv2D` para extraer patrones visuales de los posters.
- `BatchNormalization` para estabilizar el entrenamiento.
- `MaxPooling2D` para reducir dimensionalidad espacial.
- `GlobalAveragePooling2D` antes de las capas densas.
- `Dropout` para reducir sobreajuste.
- Capa final `Dense` con activacion `sigmoid`.

Se usa `sigmoid` en la salida porque cada genero se predice de forma independiente. La funcion de perdida usada es `binary_crossentropy`, apropiada para problemas multi-label. El modelo se entrena con `Adam`, `BinaryAccuracy` y `AUC` multi-etiqueta.

## 5. Evaluacion

El notebook genera automaticamente:

- Historial de entrenamiento en `training_history.csv`.
- Grafica de accuracy y loss en `training_curves.png`.
- Classification report por genero en `classification_report.csv`.
- ROC-AUC por genero en `roc_auc_scores.csv`.
- Grafica ROC-AUC en `roc_auc_by_genre.png`.
- Mejor modelo guardado como `best_movie_poster_cnn.keras`.
- ZIP descargable con todos los resultados de la ejecucion.

## 6. Prueba con caratula externa

El enunciado pide probar una caratula fuera del dataset y mostrar los primeros 5 generos predichos. El notebook incluye una celda con `files.upload()` para subir una imagen externa en Colab, acondicionarla al tamano usado por la red, normalizar sus pixeles y mostrar el top 5 de generos con mayor probabilidad.

## 7. Resultados finales de la ejecucion en Colab

La ejecucion final se realizo en Google Colab con GPU T4 durante 10 epocas. El entrenamiento genero historial, classification report, curvas de entrenamiento, ROC-AUC por genero y el mejor modelo guardado. Los archivos quedaron almacenados en `execution/final-colab-run`.

| Metrica | Valor final |
| --- | ---: |
| Epocas | 10 |
| `binary_accuracy` entrenamiento | 0.8921 |
| `auc` entrenamiento | 0.6819 |
| `loss` entrenamiento | 0.2883 |
| `val_binary_accuracy` | 0.8895 |
| `val_auc` | 0.6830 |
| `val_loss` | 0.2919 |
| ROC-AUC promedio por genero | 0.6756 |
| ROC-AUC mediano por genero | 0.6810 |
| F1 micro | 0.4288 |
| F1 ponderado | 0.2965 |
| F1 por muestras | 0.4252 |

Los mejores ROC-AUC por genero fueron:

| Genero | ROC-AUC |
| --- | ---: |
| Animation | 0.8441 |
| Comedy | 0.7941 |
| Horror | 0.7689 |
| Family | 0.7648 |
| Sci-Fi | 0.7516 |

Los generos con menor ROC-AUC fueron:

| Genero | ROC-AUC |
| --- | ---: |
| Music | 0.5091 |
| War | 0.5304 |
| Biography | 0.5663 |
| History | 0.5758 |
| Sport | 0.5961 |

La metrica de `binary_accuracy` es alta porque la tarea multi-label tiene muchas etiquetas negativas por imagen. Por eso, para interpretar la capacidad real del modelo es mas representativo revisar ROC-AUC por genero y la prediccion top 5 sobre una caratula externa. El classification report muestra que el umbral fijo de 0.35 favorece generos frecuentes como Drama y Comedy, mientras que varios generos minoritarios quedan con bajo recall. Este comportamiento es esperable en un dataset desbalanceado y puede mejorarse ajustando umbrales por genero o entrenando mas epocas con estrategias de balanceo.

## 8. Sustentacion breve

Para la sustentacion de maximo 5 minutos se recomienda explicar:

1. La diferencia entre clasificacion multiclase y multi-label.
2. Por que se usa `sigmoid` y `binary_crossentropy`.
3. Como `ImageDataGenerator.flow_from_dataframe` permite leer imagenes y etiquetas desde un dataframe.
4. La arquitectura CNN propuesta y el papel de convoluciones, pooling, batch normalization y dropout.
5. Las metricas generadas: binary accuracy, loss, classification report y ROC-AUC por genero.
6. La prueba con una caratula externa y los top 5 generos predichos.

## 9. Conclusiones

El enfoque implementado permite resolver una tarea multi-label realista donde cada poster puede activar varias etiquetas de genero. La CNN propuesta aprende patrones visuales asociados a composicion, paleta de color, presencia de rostros, texto y estilos graficos de los posters. El notebook permite guardar metricas, graficas, modelo entrenado y probar una caratula externa como exige el enunciado.

La ejecucion final en Colab confirma que el flujo completo funciona con GPU y 10 epocas. Los resultados son razonables para una CNN original entrenada desde cero sobre un problema multi-label desbalanceado: el modelo alcanza buen comportamiento global en `binary_accuracy`, un ROC-AUC promedio por genero cercano a 0.68 y mejores resultados en generos visualmente mas distinguibles como Animation, Comedy, Horror, Family y Sci-Fi.
