# Pendientes para entrega final - Tarea 2 FIA Movie Posters

Este trabajo ya tiene estructura, notebook, modelo CNN multi-etiqueta, descarga/preparacion del dataset, corrida final en Colab y resultados exportados.

## Falta hacer para entrega final

- Revisar manualmente el notebook ejecutado en Colab antes de enviarlo, especialmente la salida de la prueba con caratula externa.
- Entregar el notebook ejecutado y el informe PDF final.

## Cerrado en esta revision

- Se corrigio el generador del notebook para que las celdas de codigo no queden con indentacion invalida.
- Se regenero `notebooks/tarea_2_fia_movie_posters_colab.ipynb` y se valido que sus celdas de codigo no tienen errores de sintaxis.
- Se ajusto el notebook a 10 epocas.
- Se agrego una celda final para descargar un ZIP con metricas, graficas y modelo entrenado desde Colab.
- Se ejecuto el notebook en Google Colab con GPU T4 durante 10 epocas.
- Se guardaron los resultados finales en `execution/final-colab-run`.
- Se actualizo el informe Markdown con metodologia, evaluacion, prueba de caratula externa y resultados finales.
- Se exporto un PDF actualizado en `reports/informe_tarea_2_fia_movie_posters.pdf`.

## Estado actual

- El notebook principal esta en `notebooks/tarea_2_fia_movie_posters_colab.ipynb`.
- Ya existe una corrida local rapida en `execution/quick-local-run`.
- La corrida final esta en `execution/final-colab-run`.
- La corrida final obtuvo `val_auc = 0.6830`, `val_binary_accuracy = 0.8895` y `val_loss = 0.2919`.

## Recomendacion

Para la sustentacion, enfatizar que la tarea es multi-etiqueta y que se evalua con `binary_crossentropy`, `sigmoid`, metricas por genero y prediccion top 5 sobre una caratula externa. Tambien conviene explicar que `binary_accuracy` puede ser alta por el desbalance de etiquetas negativas, por lo que ROC-AUC por genero es mas informativo.
