# Estado de ejecucion local - MiniProyecto 3

Fecha de intento: 2026-05-15

Se intento ejecutar localmente el flujo del MiniProyecto 3 con el dataset **Garbage Classification** de Kaggle:

```powershell
C:\tf310\Scripts\kaggle.exe datasets download -d mostafaabla/garbage-classification -p C:\fia_runs\garbage --unzip
```

## Resultado

La descarga no se pudo completar porque no existe un archivo de credenciales de Kaggle en:

```text
C:\Users\ASUS\.kaggle\kaggle.json
```

La salida del comando fue:

```text
KeyError: 'username'
```

## Interpretacion

El cuaderno del proyecto esta listo para ejecutarse, pero Kaggle exige autenticacion para descargar el dataset. Para correrlo en Colab o localmente se debe cargar un archivo `kaggle.json` valido de la cuenta de Kaggle.

## Pasos para ejecutar

1. Descargar `kaggle.json` desde la configuracion de cuenta de Kaggle.
2. En Colab, subir el archivo cuando el cuaderno lo pida.
3. En ejecucion local, ubicarlo en `C:\Users\ASUS\.kaggle\kaggle.json`.
4. Volver a ejecutar el cuaderno `notebooks/miniproyecto_3_garbage_classification_colab.ipynb`.

Cuando el dataset este disponible, el cuaderno entrena y compara MLP, CNN y MobileNetV2, genera matriz de confusion, reporte de clasificacion y prediccion para imagen externa.
