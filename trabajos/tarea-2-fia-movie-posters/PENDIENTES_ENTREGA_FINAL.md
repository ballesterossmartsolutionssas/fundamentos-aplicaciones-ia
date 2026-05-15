# Pendientes para entrega final - Tarea 2 FIA Movie Posters

Este trabajo ya tiene estructura, notebook, modelo CNN multi-etiqueta, descarga/preparacion del dataset y evidencia de una corrida local rapida. Para entrega formal todavia falta cerrar estos puntos.

## Falta hacer

1. Ejecutar el notebook completo en Google Colab con GPU o en local si se prefiere esperar mas tiempo.
2. Usar mas epocas que la corrida rapida; recomendacion: entre 8 y 12 epocas.
3. Guardar los resultados finales generados por el notebook:
   - Accuracy/binary accuracy.
   - Loss de validacion.
   - Classification report por genero.
   - ROC-AUC por genero.
   - Grafica de resultados.
4. Actualizar el informe con los resultados finales reales.
5. Exportar el informe final a PDF.
6. Subir al repo el PDF final y los resultados principales si no pesan demasiado.

## Estado actual

- El notebook principal esta en `notebooks/tarea_2_fia_movie_posters_colab.ipynb`.
- Ya existe una corrida local rapida en `execution/quick-local-run`.
- La corrida rapida valido que el flujo funciona, pero no debe tomarse como resultado final fuerte de entrega.

## Recomendacion

Para la entrega, correr el notebook en Colab con GPU y usar esos resultados finales en el informe. Si el tiempo es limitado, usar 8 epocas y explicar que la tarea es multi-etiqueta, por lo que se evalua con `binary_crossentropy`, `sigmoid` y metricas por genero.
