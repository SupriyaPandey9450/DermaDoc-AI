# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Dataset path
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# ✅ CNN Model (UPDATED INPUT PART)
model = Sequential([
    Input(shape=(128, 128, 3)),   # 👈 IMPORTANT FIX

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_data,
    validation_data=test_data,
    epochs=30
)

# ✅ SAVE MODEL (DEPLOYMENT SAFE)
model.save("skin_disease_model", save_format="tf")   # 👈 BEST for Hugging Face

# (Optional backup)
model.save("skin_disease_model.keras")

# Save labels
with open("class_labels.json", "w") as f:
    json.dump(list(train_data.class_indices.keys()), f)

print("✅ Model training complete & saved safely")
