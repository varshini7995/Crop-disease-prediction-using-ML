import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Flatten, Concatenate, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# CONFIGURATION
# ===============================
DATASET_PATH = "./dataset/"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

# ===============================
# DATA GENERATOR
# ===============================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
CLASS_NAMES = list(train_gen.class_indices.keys())

# ===============================
# INPUT LAYER
# ===============================
input_img = Input(shape=(128, 128, 3))

# ===============================
# 1️⃣ CUSTOM CNN BRANCH
# ===============================
cnn = Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
cnn = MaxPooling2D()(cnn)
cnn = Conv2D(64, (3,3), activation="relu", padding="same")(cnn)
cnn = MaxPooling2D()(cnn)
cnn = Conv2D(128, (3,3), activation="relu", padding="same")(cnn)
cnn = MaxPooling2D()(cnn)
cnn_features = Flatten()(cnn)

# ===============================
# 2️⃣ VGG-16 BRANCH (MANUAL WEIGHTS)
# ===============================
vgg = VGG16(
    input_tensor=input_img,
    weights=None,
    include_top=False
)
vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

for layer in vgg.layers:
    layer.trainable = False

vgg_features = Flatten()(vgg.output)

# ===============================
# FEATURE FUSION
# ===============================
merged_features = Concatenate()([cnn_features, vgg_features])

x = Dense(512, activation="relu")(merged_features)
x = Dense(256, activation="relu")(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

# ===============================
# FINAL HYBRID MODEL
# ===============================
model = Model(inputs=input_img, outputs=output)
model.summary()

# ===============================
# COMPILE MODEL
# ===============================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# CHECKPOINT
# ===============================
checkpoint = ModelCheckpoint(
    "best_hybrid_cnn_vgg16.h5",
    monitor="val_loss",
    save_best_only=True
)

# ===============================
# TRAIN MODEL
# ===============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# ===============================
# SAVE FINAL MODEL
# ===============================
model.save("hybrid_cnn_vgg16_final.h5")
print("✅ Hybrid CNN + VGG16 model saved successfully")

# ===============================
# PLOT ACCURACY
# ===============================
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Hybrid CNN + VGG16 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ===============================
# PLOT LOSS
# ===============================
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Hybrid CNN + VGG16 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ===============================
# EVALUATION
# ===============================
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid CNN + VGG16 Confusion Matrix")
plt.show()

