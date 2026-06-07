# Information: Training data pipeline module.
# Importance: Configures tf.data.Dataset loaders with performance prefetching and computes class weights to combat dataset imbalances.

import os
import tensorflow as tf
from typing import Tuple, Dict, Optional

# Constants for default paths
DEFAULT_TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Brain Tumor data", "Training"))
DEFAULT_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Brain Tumor data", "Testing"))

def get_data_loaders(
    train_dir: str = DEFAULT_TRAIN_DIR,
    test_dir: str = DEFAULT_TEST_DIR,
    image_size: int = 224,
    batch_size: int = 32
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates tf.data.Dataset pipelines for training and testing images.
    
    Args:
        train_dir: Directory containing training images grouped by subdirectories.
        test_dir: Directory containing testing images grouped by subdirectories.
        image_size: Target height/width for resizing images.
        batch_size: Batch size for training.
        
    Returns:
        train_ds: Prefetched tf.data.Dataset for training.
        test_ds: Prefetched tf.data.Dataset for testing.
    """
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Dataset directories not found. Please ensure they exist at:\n"
            f"Training: {train_dir}\nTesting: {test_dir}"
        )

    # In Keras, class_names are sorted alphabetically, which maps to:
    # 0 = glioma, 1 = meningioma, 2 = notumor, 3 = pituitary
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",  # Categorical for multi-class classification
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=42
    )

    # Load testing/validation dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=False  # Do not shuffle test set for easy evaluation
    )

    # Prefetch for GPU memory optimization and thread performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds

def calculate_class_weights(train_dir: str = DEFAULT_TRAIN_DIR) -> Optional[Dict[int, float]]:
    """
    Computes class weights to combat class imbalance.
    Class weight = total_samples / (num_classes * class_samples)
    """
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    counts = []
    
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        if os.path.exists(cls_path):
            counts.append(len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
        else:
            counts.append(0)

    total_samples = sum(counts)
    if total_samples == 0:
        return None

    num_classes = len(class_names)
    class_weights = {}
    
    for idx, count in enumerate(counts):
        if count > 0:
            class_weights[idx] = total_samples / (num_classes * count)
        else:
            class_weights[idx] = 1.0  # Fallback to neutral weight if no samples

    print(f"[DataLoader] Calculated Class Counts: {dict(zip(class_names, counts))}")
    print(f"[DataLoader] Calculated Class Weights: {class_weights}")
    
    return class_weights
