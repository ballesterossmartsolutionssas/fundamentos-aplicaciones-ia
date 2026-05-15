# Ejecucion local rapida - Tarea 2 FIA

Fecha de ejecucion: 2026-05-15

Esta carpeta contiene evidencia de una corrida local rapida del cuaderno de clasificacion multi-etiqueta de posters de peliculas. La ejecucion se hizo para validar que el flujo de datos, el entrenamiento, la evaluacion y las metricas funcionan antes de correr el entrenamiento completo en Colab.

## Entorno

- Python: entorno local `C:\tf310`
- TensorFlow: 2.10.1
- Ejecucion: CPU
- Dataset: `Movies-Poster_Dataset` clonado desde GitHub

## Parametros de la corrida

- Filas usadas: 1800
- Entrenamiento: 1260 imagenes
- Validacion: 270 imagenes
- Prueba: 270 imagenes
- Tamano de imagen: 96 x 96
- Epocas: 2
- Arquitectura: CNN secuencial multi-etiqueta
- Funcion de perdida: `binary_crossentropy`
- Salida: activacion `sigmoid`

## Resultados principales

- `val_auc` final: 0.5191
- `val_binary_accuracy` final: 0.8758

Estos valores no se presentan como resultado final del proyecto, porque la corrida fue reducida. El cuaderno principal esta preparado para ejecutarse con mas epocas y tamano de imagen mayor en Colab.

## Archivos

- `quick_training_history.csv`: historial de entrenamiento por epoca.
- `quick_classification_report.csv`: precision, recall y F1 por genero.
- `quick_roc_auc_scores.csv`: ROC-AUC por genero.
- `quick_roc_auc_by_genre.png`: grafica de ROC-AUC por genero.
