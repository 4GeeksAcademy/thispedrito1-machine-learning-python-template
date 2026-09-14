# CLAUDE.md

Repositorio de proyectos de Machine Learning del bootcamp de 4Geeks. Cada clase **reemplaza** el proyecto anterior (`src/app.py`, `src/utils.py`, `src/explore.ipynb`, `README.es.md`); el trabajo previo queda en el historial de git.

## Entorno

- Local (macOS arm64, 8 GB RAM, poco disco libre). Python del sistema es 3.14, pero TensorFlow solo publica wheels hasta 3.13: usar el entorno virtual `.venv` (Python 3.13 de Homebrew).
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- El devcontainer (Codespaces) usa Python 3.13.
- No hay tests ni linter configurados; verificar con pruebas de humo sobre datos pequeños.

## Convenciones

- `src/app.py`: constantes en mayúsculas arriba (rutas desde `PROJECT_ROOT`, `RANDOM_STATE = 42`), funciones pequeñas `build_*`/`prepare_*` y un `run_pipeline()` que devuelve un dict de métricas; `if __name__ == "__main__"` imprime el resumen.
- `src/explore.ipynb`: explica en español paso a paso e importa las funciones de `app.py`/`utils.py` (añade `src` a `sys.path`) en lugar de duplicar código.
- Datos en `data/raw` → `data/interim` → `data/processed`; modelos en `models/`. Datasets pesados van en `.gitignore`.
- Código y docstrings en inglés; textos del notebook y gráficos en español.

## Historial de clases

1. **Análisis de sentimiento** (`reviews.csv`).
2. **K-Means** sobre California Housing + clasificador supervisado (RandomForest) con las etiquetas de los clusters.
3. **RNA para clasificación de imágenes (perros vs. gatos)** — rama `feat/image-classifier-cnn`:
   - `prepare_dataset()` descarga `dogs-vs-cats.zip` (~580 MB, solo carpeta `train/` con 25.000 fotos `cat.N.jpg`/`dog.N.jpg`) y lo extrae directamente en `data/interim/dogs-vs-cats.noindex/{train,test}/{cat,dog}` (el sufijo `.noindex` evita que Spotlight indexe las 25.000 fotos, que consumía CPU y disco) (80/20 estratificado, marcador `.complete`, borra el zip).
   - Carga perezosa con `ImageDataGenerator.flow_from_directory()` (224×224, batch 16); 10% del train como validación para los callbacks, test intacto.
   - Modelo: `EfficientNetB0(weights=None)` + `GlobalAveragePooling2D` + `Dense(128)` + `Dense(2, softmax)`, Adam, `categorical_crossentropy`.
   - Callbacks: `ModelCheckpoint` + `EarlyStopping` (`val_loss`, `patience=3`, `restore_best_weights`). Mejor modelo en `models/efficientnet_dogs_vs_cats.keras` (~51 MB).
   - Gotchas: Keras 3 ya no exporta `ImageDataGenerator` → importar desde `tensorflow.keras.preprocessing.image`; `fit_generator` no existe → `model.fit`; EfficientNet normaliza internamente → **no** usar `rescale=1/255`.
   - Entrenamiento completo en CPU (MacBook Air M1 sin ventilador): ~24 min por época; ~17 min con un ventilador externo, porque el chip deja de frenarse por calor. La prueba de humo sobreestimó (~70 min).
   - Monitorizar temperatura/recursos sin sudo con `macmon pipe` (instalado con Homebrew).
   - **Resultado del entrenamiento completo (2026-09-14):** 18 épocas (~5,5 h), EarlyStopping restauró la época 15 (val_loss 0,182, val_accuracy 93,4%). **Test: accuracy 94,3%, loss 0,150** con 5.000 imágenes. Historial por época en `models/training_history.json` (el script no lo guardaba; se extrajo del log). Sobreajuste visible desde la época 16 (train 97% vs validación ~92%).
   - Notebook ejecutado con salidas (figuras en JPEG para que pese <1 MB): solo reentrena si no existe el modelo (`TRAIN_MODEL`); si existe, carga `models/training_history.json`. Incluye matriz de confusión (190 gatos→perro, 93 perros→gato), fotos de test con predicción y los 9 errores más confiados, y conclusiones. `evaluate_model` predice el test en una sola pasada; `run_pipeline` guarda el historial con `save_training_summary`.
   - Para reejecutar el notebook: `cd src && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb` (~1 min con el modelo ya entrenado).
