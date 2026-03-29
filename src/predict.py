import numpy as np

def predict_masks(model, X_test):
    predictions = model.predict(X_test)
    predicted_masks = np.argmax(predictions, axis=3)
    return predicted_masks

def count_predicted_pixels(predicted_mask):
    unique_pixels, counts = np.unique(predicted_mask, return_counts=True)
    return dict(zip(unique_pixels, counts))
