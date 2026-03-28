# ============================================================
#  AlexNet adaptada para CIFAR-10
#  Autora : Sofia Avila
#  Curso  : Aprendizaje Profundo — Universidad del Rosario
# ============================================================

# ── 1. Importaciones ─────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

os.makedirs("results", exist_ok=True)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# ── 2. Carga y preprocesamiento ──────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

val_size = int(0.1 * len(X_train))
X_val,   y_val   = X_train[:val_size], y_train[:val_size]
X_train, y_train = X_train[val_size:], y_train[val_size:]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


# ── 3. Visualizar muestras ───────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for ax in axes.flat:
    idx = np.random.randint(len(X_train))
    ax.imshow(X_train[idx])
    ax.set_title(CLASS_NAMES[y_train[idx][0]], fontsize=8)
    ax.axis("off")
plt.suptitle("Muestras CIFAR-10", fontweight="bold")
plt.tight_layout()
plt.savefig("results/sample_images.png", dpi=150)
plt.show()


# ── 4. Data Augmentation ─────────────────────────────────────
datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.1,
)
datagen.fit(X_train)


# ── 5. Arquitectura AlexNet ──────────────────────────────────
#
#  AlexNet original fue disenada para 224x224 (ImageNet).
#  Adaptacion para CIFAR-10 (32x32):
#    - UpSampling2D: 32 -> 64 px
#    - BatchNorm en lugar de LRN
#
def build_alexnet(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential(name="AlexNet_CIFAR10")

    model.add(layers.Input(shape=input_shape))
    model.add(layers.UpSampling2D(size=(2, 2)))           # 32 -> 64

    # Bloque 1
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4),
                            padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Bloque 2
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1),
                            padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Bloque 3 — tres conv seguidas
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Clasificador FC
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


model = build_alexnet()
model.summary()


# ── 6. Compilar ──────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print(f"Parametros totales: {model.count_params():,}")


# ── 7. Entrenamiento ─────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 128

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath="results/alexnet_best.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)
print("Entrenamiento finalizado.")


# ── 8. Curvas de entrenamiento ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train", color="steelblue")
axes[0].plot(history.history["val_accuracy"], label="Val",   color="coral", linestyle="--")
axes[0].set_title("Accuracy por Epoca", fontweight="bold")
axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train", color="steelblue")
axes[1].plot(history.history["val_loss"], label="Val",   color="coral", linestyle="--")
axes[1].set_title("Loss por Epoca", fontweight="bold")
axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.show()


# ── 9. Evaluacion en Test ────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = y_test.flatten()

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\nReporte de Clasificacion:")
print(report)


# ── 10. Matriz de confusion ──────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Matriz de Confusion — AlexNet CIFAR-10", fontweight="bold")
ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.show()


# ── 11. Predicciones de ejemplo ──────────────────────────────
indices = np.random.choice(len(X_test), 16, replace=False)
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for ax, idx in zip(axes.flat, indices):
    ax.imshow(X_test[idx])
    pred  = CLASS_NAMES[y_pred[idx]]
    true  = CLASS_NAMES[y_true[idx]]
    color = "green" if pred == true else "red"
    ax.set_title(f"P:{pred}\nT:{true}", fontsize=7, color=color, fontweight="bold")
    ax.axis("off")
plt.suptitle("Predicciones (verde=correcto, rojo=incorrecto)", fontweight="bold")
plt.tight_layout()
plt.savefig("results/sample_predictions.png", dpi=150)
plt.show()


# ── 12. Guardar metricas ─────────────────────────────────────
metrics = {
    "test_loss":         float(test_loss),
    "test_accuracy":     float(test_acc),
    "best_val_accuracy": float(max(history.history["val_accuracy"])),
    "epochs_trained":    len(history.history["accuracy"]),
    "total_parameters":  model.count_params(),
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open("results/metrics.txt", "w") as f:
    f.write("AlexNet CIFAR-10 — Metricas Finales\n")
    f.write("=" * 40 + "\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\n--- Reporte por Clase ---\n")
    f.write(report)

print("\nMetricas guardadas en results/")
for fname in sorted(os.listdir("results")):
    size = os.path.getsize(f"results/{fname}")
    print(f"  {fname}  ({size/1024:.1f} KB)")


# ── 13. Guardar modelo ───────────────────────────────────────
model.save("results/alexnet_cifar10_final.keras")
print("Modelo guardado en results/alexnet_cifar10_final.keras")
