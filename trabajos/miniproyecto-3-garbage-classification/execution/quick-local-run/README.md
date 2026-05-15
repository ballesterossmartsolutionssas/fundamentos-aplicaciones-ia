# Ejecucion local rapida - MiniProyecto 3

Fecha de ejecucion: 2026-05-15

Esta carpeta contiene evidencia de una corrida local rapida del proyecto **Garbage Classification**. La ejecucion valida el flujo completo con descarga del dataset, preparacion de datos, entrenamiento de tres modelos y evaluacion.

## Descarga del dataset

La descarga se realizo con KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("mostafaabla/garbage-classification")
```

Ruta local generada:

```text
C:\Users\ASUS\.cache\kagglehub\datasets\mostafaabla\garbage-classification\versions\1
```

## Parametros de la corrida

- Modelos: MLP, CNN y MobileNetV2.
- Clases: 12 categorias.
- Muestra balanceada: 60 imagenes por clase.
- Total usado: 720 imagenes.
- Entrenamiento: 504 imagenes.
- Validacion: 108 imagenes.
- Prueba: 108 imagenes.
- Tamano de imagen: 96 x 96.
- Epocas: 1.
- Entorno: TensorFlow 2.10.1 en CPU.

## Resultados principales

| Modelo | Accuracy test | Macro F1 test | Accuracy validacion |
| --- | ---: | ---: | ---: |
| MLP | 0.1111 | 0.0349 | 0.1204 |
| CNN | 0.1204 | 0.0671 | 0.1204 |
| MobileNetV2 | 0.5093 | 0.4845 | 0.4074 |

La corrida fue reducida para validar ejecucion local. Para el informe final se recomienda correr el notebook completo en Colab con mas epocas; aun asi, esta prueba confirma la expectativa metodologica: MobileNetV2 con transferencia de aprendizaje supera a MLP y CNN entrenadas desde cero en pocas epocas.

## Archivos

- `quick_model_comparison.csv`: comparacion resumida de los tres modelos.
- `quick_run_summary.json`: configuracion y resultados de la corrida.
- `*_history.csv`: historial de entrenamiento por modelo.
- `*_classification_report.csv`: metricas por clase.
- `*_confusion_matrix.png`: matriz de confusion por modelo.
