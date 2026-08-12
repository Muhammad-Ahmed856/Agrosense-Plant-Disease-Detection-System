"""
AgroSense — Transfer Learning Training Script (MobileNetV2)
============================================================
Uses MobileNetV2 pre-trained on ImageNet instead of training from scratch.

WHY THIS IS BETTER:
  - MobileNetV2 already knows edges, textures, shapes from 14M images
  - Works much better on real-world (Google) images out of the box
  - Reaches ~97-98% accuracy vs ~90-95% with custom CNN
  - Trains faster — only the top layers need heavy training

Works on:
  • Kaggle Notebook (GPU T4 / T4x2)  ← recommended, ~15 mins
  • Local machine (CPU or GPU)        ← CPU will take ~2-3 hours

Kaggle usage:
  1. Create a new Kaggle Notebook
  2. Add dataset: "vipoooool/new-plant-diseases-dataset"
  3. Enable GPU: Session options → Accelerator → GPU T4 x2
  4. Paste this entire script and run

Output files (download from Kaggle Output tab):
  plant_model.h5
  class_names.json
→ Place both in your local  agrosense/model/  folder
"""

import os, json, sys, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D,
    Dropout, BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# ══════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════

ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE:
    DATASET_ROOT = "/kaggle/input/new-plant-diseases-dataset"
    OUTPUT_DIR   = "/kaggle/working"
    print("🌐 Running on Kaggle")
else:
    print("💻 Running locally — downloading dataset via KaggleHub...")
    import kagglehub
    DATASET_ROOT = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    OUTPUT_DIR   = "model"
    print(f"✅ Dataset path: {DATASET_ROOT}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "plant_model.h5")
NAMES_PATH = os.path.join(OUTPUT_DIR, "class_names.json")

# ══════════════════════════════════════════════════════════════════
# 2. GPU SETUP
# ══════════════════════════════════════════════════════════════════

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    print(f"⚡ Strategy: {strategy.__class__.__name__}")
else:
    strategy = tf.distribute.get_strategy()
    print("⚠️  No GPU — running on CPU (will be slow)")

# ══════════════════════════════════════════════════════════════════
# 3. CONFIG
# MobileNetV2 requires minimum 96×96 input. We use 224×224 which
# is its native size — gives best feature extraction quality.
# ══════════════════════════════════════════════════════════════════

IMG_SIZE        = (224, 224)   # MobileNetV2 native size — do not change
BASE_BATCH_SIZE = 32           # per GPU (MobileNetV2 is heavier than custom CNN)
BATCH_SIZE      = BASE_BATCH_SIZE * max(1, len(gpus))

# Two-phase learning rates:
PHASE1_LR = 1e-3   # Phase 1: train only the new head (fast)
PHASE2_LR = 1e-5   # Phase 2: fine-tune top layers of MobileNetV2 (slow, careful)

PHASE1_EPOCHS = 10  # Head-only training
PHASE2_EPOCHS = 20  # Fine-tuning (EarlyStopping will cut short)

# How many layers from the top of MobileNetV2 to unfreeze in Phase 2
UNFREEZE_LAYERS = 30

print(f"\n⚙️  Config:")
print(f"   IMG_SIZE      : {IMG_SIZE}  (MobileNetV2 native)")
print(f"   BATCH_SIZE    : {BATCH_SIZE}")
print(f"   Phase 1 LR   : {PHASE1_LR}  ({PHASE1_EPOCHS} epochs, head only)")
print(f"   Phase 2 LR   : {PHASE2_LR}  ({PHASE2_EPOCHS} epochs, fine-tune top {UNFREEZE_LAYERS} layers)")

# ══════════════════════════════════════════════════════════════════
# 4. FIND DATASET PATHS
# ══════════════════════════════════════════════════════════════════

def find_split(root, split_name):
    """Walk directory tree to find train/ or valid/ regardless of nesting."""
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath).lower() == split_name:
            if any(os.path.isdir(os.path.join(dirpath, d)) for d in os.listdir(dirpath)):
                return dirpath
    return None

train_path = find_split(DATASET_ROOT, "train")
val_path   = find_split(DATASET_ROOT, "valid")

if not train_path or not val_path:
    print("\n❌ Could not find train/ or valid/. Dataset structure:")
    for root, dirs, files in os.walk(DATASET_ROOT):
        level = root.replace(DATASET_ROOT, "").count(os.sep)
        if level < 3:
            print("  " * level + os.path.basename(root) + "/")
    sys.exit(1)

print(f"\n📂 train → {train_path}")
print(f"📂 valid → {val_path}")

# ══════════════════════════════════════════════════════════════════
# 5. LOAD DATASETS
# ══════════════════════════════════════════════════════════════════

print("\n📂 Loading datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)
class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"✅ {NUM_CLASSES} classes found")

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_total = tf.data.experimental.cardinality(val_dataset).numpy()
val_size  = int(val_total * 0.5)
val_data  = val_dataset.take(val_size)
test_data = val_dataset.skip(val_size)

# ══════════════════════════════════════════════════════════════════
# 6. AUGMENTATION + PREPROCESSING
#
# IMPORTANT: MobileNetV2 needs its own preprocess_input() function
# instead of simple /255.0 division. It scales pixels to [-1, 1].
# This is what makes it work well on real-world images.
# ══════════════════════════════════════════════════════════════════

augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomTranslation(0.15, 0.15),
    tf.keras.layers.RandomBrightness(0.3),
    tf.keras.layers.RandomContrast(0.3),
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_train(x, y):
    x = augment_layer(x, training=True)
    x = preprocess_input(x)   # MobileNetV2 normalization: scales to [-1, 1]
    return x, y

def preprocess_eval(x, y):
    x = preprocess_input(x)   # same normalization, no augmentation
    return x, y

train_data = (
    train_dataset
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_data  = val_data.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_data = test_data.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ══════════════════════════════════════════════════════════════════
# 7. BUILD MODEL
#
# Transfer Learning has two phases:
#
# PHASE 1 — Feature Extraction
#   Freeze all MobileNetV2 layers. Only train the new head we add.
#   The backbone already knows how to extract features — we just
#   teach the head to map those features to our 38 disease classes.
#   Fast and stable. Runs for PHASE1_EPOCHS epochs.
#
# PHASE 2 — Fine-Tuning
#   Unfreeze the top UNFREEZE_LAYERS of MobileNetV2 and retrain
#   at a very low learning rate. This adjusts the backbone's top
#   layers slightly to better suit plant disease images specifically.
#   Must use a low LR or it will destroy the pre-trained weights.
# ══════════════════════════════════════════════════════════════════

print("\n🏗️  Building MobileNetV2 transfer learning model...")

with strategy.scope():

    # Load MobileNetV2 backbone — pre-trained on ImageNet, without its original top
    backbone = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,      # remove ImageNet classification head
        weights="imagenet"      # use pre-trained weights
    )
    backbone.trainable = False  # freeze all backbone layers for Phase 1
    print(f"✅ MobileNetV2 loaded — {len(backbone.layers)} layers, all frozen for Phase 1")

    # Build our classification head on top
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = backbone(inputs, training=False)  # training=False keeps BatchNorm frozen
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="agrosense_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

trainable = sum(1 for l in model.layers if l.trainable)
total     = len(model.layers)
print(f"   Trainable layers : {trainable} / {total}")
print(f"   Total parameters : {model.count_params():,}")

# ══════════════════════════════════════════════════════════════════
# 8. PHASE 1 — HEAD-ONLY TRAINING
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print(f"🚀 PHASE 1 — Training head only ({PHASE1_EPOCHS} epochs)")
print(f"{'='*55}")

class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._t
        va = logs.get("val_accuracy", 0) * 100
        vl = logs.get("val_loss", 0)
        print(f"  ⏱  Epoch {epoch+1}: {elapsed:.0f}s | val_acc={va:.2f}% | val_loss={vl:.4f}")

phase1_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EpochTimer(),
]

t0 = time.time()
history1 = model.fit(
    train_data,
    epochs=PHASE1_EPOCHS,
    validation_data=val_data,
    callbacks=phase1_callbacks,
    verbose=1
)
p1_time = time.time() - t0
p1_best = max(history1.history["val_accuracy"]) * 100
print(f"\n✅ Phase 1 done in {p1_time/60:.1f} min — best val_acc: {p1_best:.2f}%")

# ══════════════════════════════════════════════════════════════════
# 9. PHASE 2 — FINE-TUNING
# Unfreeze top layers of MobileNetV2 and train at very low LR
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print(f"🔧 PHASE 2 — Fine-tuning top {UNFREEZE_LAYERS} MobileNetV2 layers")
print(f"{'='*55}")

with strategy.scope():
    # Unfreeze top UNFREEZE_LAYERS of the backbone
    backbone.trainable = True
    for layer in backbone.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False   # keep lower layers frozen

    unfrozen = sum(1 for l in backbone.layers if l.trainable)
    print(f"   Unfrozen backbone layers : {unfrozen} / {len(backbone.layers)}")

    # Recompile with a much lower learning rate — critical for fine-tuning
    # High LR here would destroy the pre-trained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

phase2_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EpochTimer(),
]

t0 = time.time()
history2 = model.fit(
    train_data,
    epochs=PHASE2_EPOCHS,
    validation_data=val_data,
    callbacks=phase2_callbacks,
    verbose=1
)
p2_time = time.time() - t0
p2_best = max(history2.history["val_accuracy"]) * 100
print(f"\n✅ Phase 2 done in {p2_time/60:.1f} min — best val_acc: {p2_best:.2f}%")

# ══════════════════════════════════════════════════════════════════
# 10. EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════════

print("\n📊 Evaluating on held-out test set...")
test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"\n   Test Accuracy : {test_acc*100:.2f}%")
print(f"   Test Loss     : {test_loss:.4f}")
print(f"   Phase 1 best  : {p1_best:.2f}%")
print(f"   Phase 2 best  : {p2_best:.2f}%")
print(f"   Total time    : {(p1_time+p2_time)/60:.1f} minutes")

# ══════════════════════════════════════════════════════════════════
# 11. SAVE
# ══════════════════════════════════════════════════════════════════

model.save(MODEL_PATH)
with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\n💾 Model saved       → {MODEL_PATH}")
print(f"💾 Class names saved  → {NAMES_PATH}")

if ON_KAGGLE:
    print("\n" + "="*60)
    print("📥  DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("1. Click the [Output] tab on the right side panel")
    print("2. Download: plant_model.h5")
    print("3. Download: class_names.json")
    print("4. On your local machine, place both files in:")
    print("      agrosense/model/")
    print("5. Run: python app.py")
    print("="*60)
else:
    print("\n✅ Done! Run: python app.py")