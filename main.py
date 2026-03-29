from src.data_loading import load_images_and_masks
from src.preprocessing import prepare_10_class_data
from src.train_model import build_unet_model, train_model
from src.predict import predict_masks
from src.evaluate import evaluate_model, save_prediction_examples

def main():
    image_size = (128, 128)
    n_classes = 10

    train_images, train_masks = load_images_and_masks(
        image_dir="data/raw/train_images",
        mask_dir="data/raw/train_masks",
        image_size=image_size
    )

    test_images, test_masks = load_images_and_masks(
        image_dir="data/raw/test_images",
        mask_dir="data/raw/test_masks",
        image_size=image_size
    )

    X_train, y_train, X_test, y_test = prepare_10_class_data(
        train_images, train_masks, test_images, test_masks, n_classes=n_classes
    )

    model = build_unet_model(n_classes=n_classes, backbone="vgg16")
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=50,
        batch_size=8,
        model_save_path="models/vgg_backbone_10_classes_50epochs.hdf5"
    )

    predictions = predict_masks(model, X_test)
    evaluate_model(y_test, predictions, n_classes=n_classes)
    save_prediction_examples(X_test, y_test, predictions, output_dir="outputs/sample_predictions")

if __name__ == "__main__":
    main()
