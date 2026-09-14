import json
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from keras import ops
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.losses import categorical_crossentropy
from keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Keras 3 no longer exports ImageDataGenerator; TensorFlow still exposes the legacy class.
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator


PROJECT_ROOT = Path(__file__).parents[1]
DATA_URL = "https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip"
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "dogs-vs-cats.zip"
# The ".noindex" suffix keeps macOS Spotlight from indexing the 25,000 images.
DATASET_PATH = PROJECT_ROOT / "data" / "interim" / "dogs-vs-cats.noindex"
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
MODELS_PATH = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_PATH / "efficientnet_dogs_vs_cats.keras"
HISTORY_PATH = MODELS_PATH / "training_history.json"

CLASSES = ["cat", "dog"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42
# Marker written once the split finishes, so an interrupted extraction is redone.
COMPLETE_MARKER = DATASET_PATH / ".complete"


def download_dataset(zip_path: Path = ZIP_PATH) -> Path:
    """Download the dogs-vs-cats zip when it is not already on disk."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print("Descargando dataset (~580 MB)...")
        urlretrieve(DATA_URL, zip_path)
    return zip_path


def prepare_dataset(zip_path: Path = ZIP_PATH, delete_zip: bool = True) -> Path:
    """Extract the images straight into train/test folders with one subfolder per class.

    flow_from_directory() infers each label from the folder name, so the final
    layout is DATASET_PATH/{train,test}/{cat,dog}/<image>.jpg.
    """
    if COMPLETE_MARKER.exists():
        return DATASET_PATH

    download_dataset(zip_path)
    shutil.rmtree(DATASET_PATH, ignore_errors=True)
    rng = random.Random(RANDOM_STATE)

    with zipfile.ZipFile(zip_path) as archive:
        image_names = [
            name
            for name in archive.namelist()
            if name.startswith("dogs-vs-cats/train/") and name.endswith(".jpg")
        ]
        for label in CLASSES:
            class_images = sorted(
                name for name in image_names if Path(name).name.startswith(f"{label}.")
            )
            if not class_images:
                raise ValueError(f"No images found for class '{label}' in {zip_path}")
            rng.shuffle(class_images)
            test_count = int(len(class_images) * TEST_SIZE)

            for index, name in enumerate(class_images):
                split_path = TEST_PATH if index < test_count else TRAIN_PATH
                target = split_path / label / Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    COMPLETE_MARKER.touch()
    if delete_zip:
        zip_path.unlink()
    return DATASET_PATH


def sample_images(label: str, count: int = 9, split_path: Path = TRAIN_PATH) -> list[Path]:
    """Return the first `count` image paths of a class, for visual inspection."""
    return sorted((split_path / label).glob("*.jpg"))[:count]


def build_generators(
    batch_size: int = BATCH_SIZE,
) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """Create the train, validation and test iterators that load images lazily.

    EfficientNetB0 rescales pixels internally, so images stay in the 0-255 range.
    Train and validation use two generators with the same validation_split: the
    subset split is deterministic, so both see disjoint files, but only the
    training subset gets random flips.
    """
    common = {
        "target_size": IMAGE_SIZE,
        "classes": CLASSES,
        "class_mode": "categorical",
        "batch_size": batch_size,
    }
    train_generator = ImageDataGenerator(horizontal_flip=True, validation_split=VALIDATION_SPLIT)
    validation_generator = ImageDataGenerator(validation_split=VALIDATION_SPLIT)
    test_generator = ImageDataGenerator()

    trdata = train_generator.flow_from_directory(
        TRAIN_PATH, subset="training", shuffle=True, seed=RANDOM_STATE, **common
    )
    valdata = validation_generator.flow_from_directory(
        TRAIN_PATH, subset="validation", shuffle=False, **common
    )
    tsdata = test_generator.flow_from_directory(TEST_PATH, shuffle=False, **common)
    return trdata, valdata, tsdata


def build_model() -> Sequential:
    """Build and compile the EfficientNetB0 architecture given in the instructions."""
    model = Sequential()
    model.add(Input(shape=(*IMAGE_SIZE, 3)))
    model.add(EfficientNetB0(include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=len(CLASSES), activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_callbacks(checkpoint_path: Path = BEST_MODEL_PATH) -> list:
    """Create the callbacks that keep the best model on disk and stop training early."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )
    return [checkpoint, early_stopping]


def evaluate_model(model_path: Path = BEST_MODEL_PATH, tsdata: DirectoryIterator | None = None) -> dict:
    """Load the saved best model and predict the untouched test set in a single pass.

    tsdata must not shuffle, so the predictions line up with tsdata.classes and
    tsdata.filepaths. The loss is the same categorical cross-entropy used in training.
    """
    if tsdata is None:
        _, _, tsdata = build_generators()
    best_model = load_model(model_path)
    probabilities = best_model.predict(tsdata, verbose=0)
    predictions = probabilities.argmax(axis=1)
    labels = tsdata.classes
    one_hot_labels = np.eye(len(CLASSES))[labels]
    return {
        "test_loss": float(ops.convert_to_numpy(categorical_crossentropy(one_hot_labels, probabilities)).mean()),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "report": classification_report(
            labels, predictions, target_names=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "filepaths": tsdata.filepaths,
    }


def save_training_summary(
    history: dict,
    results: dict,
    image_counts: dict,
    output_path: Path = HISTORY_PATH,
) -> dict:
    """Store per-epoch metrics and test scores so they can be plotted without retraining."""
    summary = {
        "history": {
            metric: [float(value) for value in history[metric]]
            for metric in ("loss", "accuracy", "val_loss", "val_accuracy")
        },
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "test_loss": results["test_loss"],
        "test_accuracy": results["test_accuracy"],
        "images": image_counts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_pipeline(epochs: int = EPOCHS) -> dict:
    prepare_dataset()
    trdata, valdata, tsdata = build_generators()

    model = build_model()
    history = model.fit(
        trdata,
        validation_data=valdata,
        epochs=epochs,
        callbacks=build_callbacks(),
    )

    results = evaluate_model(BEST_MODEL_PATH, tsdata)
    image_counts = {"train": trdata.samples, "validation": valdata.samples, "test": tsdata.samples}
    summary = save_training_summary(history.history, results, image_counts)
    return {**results, **summary}


if __name__ == "__main__":
    results = run_pipeline()
    images = results["images"]
    print(f"Images: train={images['train']}, validation={images['validation']}, test={images['test']}")
    print(f"Epochs run: {len(results['history']['loss'])} (best: {results['best_epoch']})")
    print(f"Test loss: {results['test_loss']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.3f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
