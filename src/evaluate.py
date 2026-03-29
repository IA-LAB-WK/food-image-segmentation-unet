import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU

def evaluate_model(y_test, predicted_masks, n_classes=10):
    y_true = np.argmax(y_test, axis=3)

    iou_metric = MeanIoU(num_classes=n_classes)
    iou_metric.update_state(y_true.flatten(), predicted_masks.flatten())
    mean_iou = iou_metric.result().numpy()

    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou

def save_prediction_examples(X_test, y_test, predicted_masks, output_dir="outputs/sample_predictions", num_examples=3):
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.argmax(y_test, axis=3)

    for i in range(min(num_examples, len(X_test))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(X_test[i])
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(y_true[i], cmap="viridis")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(predicted_masks[i], cmap="viridis")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"prediction_{i+1}.png"))
        plt.close()
