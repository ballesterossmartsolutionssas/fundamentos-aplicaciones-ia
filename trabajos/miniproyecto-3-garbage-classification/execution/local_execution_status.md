# Estado de ejecucion local - MiniProyecto 3

Fecha de ejecucion: 2026-05-15

Primero se intento descargar el dataset con Kaggle CLI:

```powershell
C:\tf310\Scripts\kaggle.exe datasets download -d mostafaabla/garbage-classification -p C:\fia_runs\garbage --unzip
```

Ese metodo fallo porque no existia `C:\Users\ASUS\.kaggle\kaggle.json`:

```text
KeyError: 'username'
```

Luego se probo la alternativa oficial indicada en KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("mostafaabla/garbage-classification")
print("Path to dataset files:", path)
```

## Resultado actualizado

La descarga con `kagglehub` funciono correctamente y dejo los archivos en:

```text
C:\Users\ASUS\.cache\kagglehub\datasets\mostafaabla\garbage-classification\versions\1
```

Por esto, el notebook fue actualizado para descargar el dataset con `kagglehub` y ya no depende de `kaggle.json`.
