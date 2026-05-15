# Tarea 2 FIA - Clasificacion Multi-label de Generos de Peliculas

## 1. Objetivo

Implementar una red neuronal convolucional original para clasificar generos de peliculas a partir de sus caratulas. La tarea es **multi-label**, porque una misma pelicula puede pertenecer a varios generos al mismo tiempo.

## 2. Dataset

Dataset usado: https://github.com/laxmimerit/Movies-Poster_Dataset.git

El dataset contiene caratulas de peliculas y un archivo CSV con metadatos, entre ellos el campo de genero. Los generos se procesan como etiquetas multiples separadas por delimitadores como `|`.

## 3. Procesamiento

El notebook realiza los siguientes pasos:

1. Clona el repositorio del dataset.
2. Ubica automaticamente el CSV y la carpeta de imagenes.
3. Relaciona cada pelicula con su archivo de imagen.
4. Convierte la columna de generos a variables binarias mediante `MultiLabelBinarizer`.
5. Filtra generos con muy pocas muestras para mejorar estabilidad del entrenamiento.
6. Divide el dataset en entrenamiento, validacion y prueba.
7. Usa `ImageDataGenerator.flow_from_dataframe`, como solicita la guia de la tarea.

## 4. Modelo CNN propuesto

La arquitectura implementada es una CNN original con:

- Capas convolucionales `Conv2D`.
- Normalizacion por lotes `BatchNormalization`.
- Reduccion espacial con `MaxPooling2D`.
- Regularizacion `Dropout`.
- Capa final `Dense` con activacion `sigmoid`.

La activacion final es `sigmoid` porque se trata de clasificacion multi-label. La perdida usada es `binary_crossentropy`.

## 5. Evaluacion

Se generan:

- Classification report por genero.
- ROC-AUC por genero.
- Grafica de entrenamiento y validacion.
- Funcion para subir una caratula nueva y mostrar los primeros 5 generos predichos.

## 6. Sustentacion breve

Para la sustentacion de maximo 5 minutos se recomienda explicar:

1. La diferencia entre clasificacion multiclase y multi-label.
2. Por que se usa `sigmoid` y `binary_crossentropy`.
3. Como `ImageDataGenerator.flow_from_dataframe` permite leer imagenes y etiquetas desde un dataframe.
4. La arquitectura CNN propuesta.
5. La prueba con una caratula externa y los top 5 generos predichos.

## 7. Conclusiones

El enfoque permite resolver una tarea multi-label realista, donde cada poster puede activar varias etiquetas. La CNN aprende patrones visuales presentes en los posters, como paletas de color, rostros, composicion y elementos graficos asociados a generos cinematograficos.

El rendimiento final debe revisarse en el notebook luego de ejecutar el entrenamiento en Colab, especialmente las metricas ROC-AUC y el comportamiento de la prediccion sobre imagenes externas.
