import numpy as np
from tensorflow.keras.utils import to_categorical
import segmentation_models as sm

# Mapping based on the logic in the original implementation
# Final 10 classes:
# 0 background
# 1 rice
# 2 green beans
# 3 french fries
# 4 carrots
# 5 shrimp
# 6 steak
# 7 onion
# 8 tomato
# 9 egg
LABEL_MAP = {
    66: 1,  # rice
    95: 2,  # green beans
    3: 3,   # french fries if already 3 in dataset
    84: 4,  # carrots
    56: 5,  # shrimp
    46: 6,  # steak
    93: 7,  # onion
    73: 8,  # tomato
    24: 9   # egg
}

KEEP_LABELS = set(LABEL_MAP.keys()) | {3}

def remap_mask_to_10_classes(mask):
    remapped = np.zeros_like(mask)

    for old_label, new_label in LABEL_MAP.items():
        remapped[mask == old_label] = new_label

    # preserve fries if already labeled as 3
    remapped[mask == 3] = 3

    return remapped

def filter_nonempty_samples(images, masks, min_labeled_pixels=300):
    filtered_images = []
    filtered_masks = []

    for img, mask in zip(images, masks):
        labeled_count = np.sum(mask > 0)
        if labeled_count > min_labeled_pixels:
            filtered_images.append(img)
            filtered_masks.append(mask)

    return np.array(filtered_images), np.array(filtered_masks)

def prepare_masks_for_training(masks, n_classes=10):
    masks = np.expand_dims(masks, axis=3)
    masks_cat = to_categorical(masks, num_classes=n_classes)
    masks_cat = masks_cat.reshape((masks.shape[0], masks.shape[1], masks.shape[2], n_classes))
    return masks_cat

def preprocess_images(images, backbone="vgg16"):
    preprocess_input = sm.get_preprocessing(backbone)
    return preprocess_input(images)

def prepare_10_class_data(train_images, train_masks, test_images, test_masks, n_classes=10):
    train_masks_remapped = np.array([remap_mask_to_10_classes(mask) for mask in train_masks])
    test_masks_remapped = np.array([remap_mask_to_10_classes(mask) for mask in test_masks])

    train_images_filtered, train_masks_filtered = filter_nonempty_samples(
        train_images, train_masks_remapped, min_labeled_pixels=300
    )
    test_images_filtered, test_masks_filtered = filter_nonempty_samples(
        test_images, test_masks_remapped, min_labeled_pixels=300
    )

    X_train = preprocess_images(train_images_filtered, backbone="vgg16")
    X_test = preprocess_images(test_images_filtered, backbone="vgg16")

    y_train = prepare_masks_for_training(train_masks_filtered, n_classes=n_classes)
    y_test = prepare_masks_for_training(test_masks_filtered, n_classes=n_classes)

    return X_train, y_train, X_test, y_test
