import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from training.data_loader import get_data_loaders, calculate_class_weights

# Ensure reproducible training results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Configure TensorFlow to run deterministically if possible
    # (Note: Some GPU operations are inherently non-deterministic)
    tf.config.experimental.enable_op_determinism()

def setup_mixed_precision():
    """Enables mixed precision training if a GPU is available to speed up training."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Set memory growth to prevent TF from locking all VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("[System] GPU acceleration detected. Mixed precision (float16) enabled.")
        except Exception as e:
            print(f"[System] Warning: Failed to initialize GPU config: {str(e)}")
    else:
        print("[System] No GPU detected. Training will run on CPU in float32 precision.")

def build_model(model_type: str, num_classes: int = 4, image_size: int = 224) -> tf.keras.Model:
    """
    Builds an EfficientNet-based transfer learning model with integrated
    data augmentation and a custom classification head.
    """
    # 1. Image augmentation pipeline (only active during training)
    img_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ], name="img_augmentation")

    # Input layer
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    
    # Apply data augmentation
    x = img_augmentation(inputs)
    
    # 2. Base Model selection
    if model_type.lower() == "efficientnet_b0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=x
        )
    elif model_type.lower() == "efficientnet_b3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_tensor=x
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'efficientnet_b0' or 'efficientnet_b3'")
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Custom head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # IMPORTANT: The output activation must be float32 to prevent numeric underflow in mixed precision
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"BrainTumorClassifier_{model_type}")
    return model, base_model

def plot_history(history, save_path: str):
    """Generates and saves loss and accuracy plots."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#4A90E2", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#FF4B4B", linewidth=2)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#4A90E2", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#FF4B4B", linewidth=2)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Saved metrics plot to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classification model.")
    parser.add_argument("--model_type", type=str, default="efficientnet_b3", choices=["efficientnet_b0", "efficientnet_b3"], help="Model architecture.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs_head", type=int, default=10, help="Epochs to train only the classification head.")
    parser.add_argument("--epochs_ft", type=int, default=15, help="Epochs for fine-tuning the model.")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for classification head.")
    parser.add_argument("--lr_ft", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_mixed_precision()

    # Image sizes: 224 for B0, 224/300 for B3. We use 224 consistently for uniform test pipelines.
    image_size = 224

    # Setup directories
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[Training] Loading datasets...")
    train_ds, test_ds = get_data_loaders(image_size=image_size, batch_size=args.batch_size)
    class_weights = calculate_class_weights()

    print(f"[Training] Initializing transfer learning with {args.model_type}...")
    model, base_model = build_model(args.model_type, num_classes=4, image_size=image_size)

    # Compile model for head-only training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_head),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Stage 1: Train Custom Classification Head
    print("\n=== Stage 1: Training Custom Classification Head ===")
    
    model_checkpoint_path = os.path.join(models_dir, "best_model.h5")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    history_head = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Stage 2: Fine-Tuning
    # Unfreeze the top layers of the base model for fine-tuning
    if args.epochs_ft > 0:
        print("\n=== Stage 2: Fine-tuning Base Model ===")
        base_model.trainable = True
        
        # Unfreeze all layers except BatchNormalization layers
        # Unfreezing BN layers can disrupt mean/variance statistics
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                
        # Re-compile the model with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_ft),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("Re-compiled model for fine-tuning. Layer status:")
        print(f"Base model trainable: {base_model.trainable}")

        history_ft = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=args.epochs_ft,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # Merge histories for plotting
        history = history_ft
        for key in history_head.history.keys():
            history_head.history[key].extend(history_ft.history[key])
        final_history = history_head
    else:
        final_history = history_head

    # Plot and save performance charts
    plot_path = os.path.join(artifacts_dir, "training_plots.png")
    plot_history(final_history, plot_path)

    print(f"\n[Training Completed] Best model saved successfully to: {model_checkpoint_path}")

if __name__ == "__main__":
    main()
