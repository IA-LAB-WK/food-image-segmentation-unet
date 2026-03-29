import os
import glob
import cv2
import numpy as np

def load_images(image_dir, image_size=(128, 128), file_pattern="*.jpg"):
    images = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, file_pattern)))

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_size)
        images.append(img)

    return np.array(images)

def load_masks(mask_dir, image_size=(128, 128), file_pattern="*.png"):
    masks = []
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, file_pattern)))

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    return np.array(masks)

def load_images_and_masks(image_dir, mask_dir, image_size=(128, 128)):
    images = load_images(image_dir=image_dir, image_size=image_size)
    masks = load_masks(mask_dir=mask_dir, image_size=image_size)
    return images, masks
