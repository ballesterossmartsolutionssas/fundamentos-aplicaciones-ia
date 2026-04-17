# Fundamentos y Aplicaciones de IA

Repositorio general de la materia **Fundamentos y Aplicaciones de Inteligencia Artificial**.

Este repositorio queda como base para organizar entregas, notebooks, informes y codigo de la asignatura. Actualmente contiene el desarrollo del **MiniProyecto 2: clasificacion de niveles de obesidad**.

## Integrantes

- Valentina Popo Montilla
- Juan Camilo Balleresteros Sierra
- Santigo Rodriguez Gacha

## Dataset seleccionado

- **Nombre:** Estimation of Obesity Levels Based On Eating Habits and Physical Condition
- **Fuente:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
- **Instancias:** 2111
- **Variables predictoras:** 16
- **Variable objetivo real en el CSV:** `NObeyesdad`
- **Clases:** Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III

## Trabajo actual

El contenido actual del repositorio desarrolla un problema de clasificacion a partir del dataset seleccionado, cumpliendo con los entregables del miniproyecto:

1. Describir el dataset de forma detallada.
2. Analizar las variables del dataset.
3. Dividir los datos en entrenamiento y validacion.
4. Depurar variables estimadoras:
   - columnas con mas del 50% de vacios
   - varianza cero
   - variables altamente correlacionadas
   - otras reglas justificadas
5. Entrenar al menos tres modelos de clasificacion:
   - minimo dos vistos en clase
   - minimo uno no visto en clase, con explicacion
6. Ajustar hiperparametros.
7. Comparar modelos con matriz de confusion y al menos una metrica adicional no vista en clase.
8. Redactar conclusiones argumentadas sobre el mejor modelo para esta aplicacion.

## Estructura del repositorio

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- reports/
|-- src/
|   `-- download_data.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Inicio rapido

### 1. Crear entorno virtual

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### 2. Descargar el dataset

```powershell
python src\\download_data.py
```

Esto generara:

- `data/raw/obesity.csv`
- `data/raw/obesity_metadata.json`
- `data/raw/obesity_variables.csv`

Si quieren trabajar en notebook, pueden usar Google Colab o instalar Jupyter por separado mas adelante.

### 3. Ejecutar el trabajo completo

```powershell
python src\\run_analysis.py
```

Esto genera automaticamente:

- `data/processed/obesity_clean.csv`
- `reports/results/model_metrics.csv`
- `reports/results/best_hyperparameters.json`
- `reports/results/pruning_summary.json`
- `reports/figures/*.png`
- `reports/informe_miniproyecto_2.md`
- `notebooks/miniproyecto_2_clasificacion_obesidad.ipynb`

### 4. Exportar el informe a Word

El informe editable para subir se deja en:

- `reports/informe_miniproyecto_2.docx`
- `reports/informe_miniproyecto_2.pdf`

Para regenerarlo en este entorno se puede usar el runtime del workspace:

```powershell
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src\export_report_docx.py
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src\export_report_pdf.py
```

## Propuesta de modelos

Modelos sugeridos para cumplir el trabajo:

- Regresion logistica
- SVM
- Arbol de decision
- Random Forest o XGBoost como metodo adicional no visto en clase

## Recomendacion de flujo

1. Descargar y documentar el dataset.
2. Hacer EDA y limpieza.
3. Codificar variables categoricas.
4. Separar entrenamiento y prueba.
5. Construir un pipeline por modelo.
6. Ajustar hiperparametros con `GridSearchCV` o `RandomizedSearchCV`.
7. Comparar metricas y matrices de confusion.
8. Redactar conclusiones y limitaciones.

## Entregables sugeridos en este repo

- `notebooks/`: analisis exploratorio, limpieza y modelado
- `src/`: scripts reutilizables
- `reports/`: informe final en PDF o Word

## Nota importante del dataset

Segun la ficha del UCI, aproximadamente el 77% de los datos fue generado sinteticamente usando SMOTE y el 23% fue recolectado directamente de usuarios. Ese punto debe considerarse en la discusion de resultados y limitaciones del estudio.

La descripcion del sitio de UCI menciona `NObesity`, pero el archivo CSV publicado usa la columna `NObeyesdad`. En el codigo del proyecto se toma `NObeyesdad` como variable objetivo.
