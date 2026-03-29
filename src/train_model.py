import numpy as np
import tensorflow as tf
import segmentation_models as sm

def build_unet_model(n_classes=10, backbone="vgg16", learning_rate=1e-4):
    activation = "softmax"
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    dice_loss = sm.losses.DiceLoss(class_weights=np.full(n_classes, 0.25))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + focal_loss

    metrics = [
        "accuracy",
        sm.metrics.IOUScore(threshold=0.5),
        sm.metrics.FScore(threshold=0.5)
    ]

    model = sm.Unet(
        backbone,
        encoder_weights="imagenet",
        classes=n_classes,
        activation=activation
    )

    model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=8, model_save_path=None):
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    if model_save_path:
        model.save(model_save_path)

    return history
