# MiniProyecto 3 - Redes Neuronales Artificiales

## Garbage Classification

### Integrantes

- Valentina Popo Montilla
- Santiago Rodriguez Gacha
- Juan Camilo Ballesteros

## 1. Descripcion del problema

El objetivo del miniproyecto es resolver una tarea de clasificacion de imagenes usando redes neuronales artificiales. El dataset asignado es **Garbage Classification**, disponible en Kaggle, y se trabaja con todas sus categorias.

La tarea consiste en clasificar imagenes de residuos en clases como carton, plastico, papel, metal, vidrio, basura, ropa, zapatos, residuos biologicos y baterias, entre otras. Este tipo de problema es relevante para aplicaciones de reciclaje automatizado, separacion de residuos y sistemas de vision artificial aplicados al medio ambiente.

## 2. Dataset

Fuente: https://www.kaggle.com/mostafaabla/garbage-classification

Categorias esperadas del dataset:

- battery
- biological
- brown-glass
- cardboard
- clothes
- green-glass
- metal
- paper
- plastic
- shoes
- trash
- white-glass

El notebook inspecciona automaticamente las carpetas descargadas desde Kaggle y usa todas las categorias disponibles.

## 3. Division de datos

El dataset se divide en tres subconjuntos:

- Entrenamiento: 70%
- Validacion: 15%
- Prueba: 15%

La particion se realiza de forma estratificada por carpeta de clase para mantener representacion de todas las categorias.

## 4. Preprocesamiento

Las imagenes se redimensionan a 128 x 128 pixeles. Se normalizan los valores de pixeles al rango [0, 1] y se aplica data augmentation en los modelos CNN y transfer learning mediante rotaciones, zoom y volteo horizontal.

## 5. Arquitecturas implementadas

### 5.1 MLP

La primera arquitectura usa una red densa multicapa. Las imagenes se aplanan y pasan por capas densas con regularizacion `Dropout`. Este modelo sirve como linea base, aunque no aprovecha la estructura espacial de las imagenes.

### 5.2 CNN original

La segunda arquitectura es una red convolucional propia con bloques `Conv2D`, `MaxPooling2D`, `BatchNormalization` y `Dropout`. Esta arquitectura aprende filtros espaciales directamente sobre las imagenes.

### 5.3 Transfer learning

La tercera arquitectura usa **MobileNetV2** preentrenada en ImageNet como extractor de caracteristicas. Sobre esta base se agrega una cabeza de clasificacion adaptada al numero de categorias del dataset.

## 6. Evaluacion

Para cada arquitectura se calculan:

- Matriz de confusion.
- Classification Report.
- ROC-AUC curve por clase.
- Accuracy y macro F1 en el conjunto de prueba.

Los resultados se exportan automaticamente en `/content/garbage_outputs` al ejecutar el notebook.

## 7. Observaciones esperadas

Se espera que la MLP tenga el rendimiento mas bajo porque no explota patrones espaciales. La CNN deberia mejorar al aprender filtros visuales especificos del dataset. El modelo de transfer learning deberia alcanzar el mejor desempeno general porque reutiliza representaciones visuales aprendidas en un conjunto de imagenes mucho mas grande.

## 8. Conclusiones

El enfoque recomendado para esta aplicacion es usar transfer learning con MobileNetV2, especialmente si el dataset no es muy grande o si las clases presentan variaciones visuales importantes. La CNN original es una alternativa valida para comparar aprendizaje desde cero, mientras que la MLP funciona como linea base pedagogica.

La entrega principal es el notebook de Colab, que contiene la descarga del dataset, preparacion de datos, entrenamiento de las tres arquitecturas y evaluacion completa.
