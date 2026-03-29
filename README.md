
# Food Image Segmentation with U-Net

This project explores the use of **U-Net**, a deep learning architecture most commonly used in **medical image segmentation**, in a different domain: **food image segmentation**.

The dataset for trainning and testing can be found from Kaggle FoodSeg103: https://www.kaggle.com/datasets/fontainenathan/foodseg103

U-Net is widely known in the medical field for tasks such as tumor segmentation, organ boundary detection, and tissue identification because it performs well at **pixel-level classification**. This project applies that same idea to food images, where the goal is to identify **where each food item appears in the image**, not just whether it is present. The presentation and code show this project using a reduced 10-class setup for food segmentation

## Project Overview

The model performs **semantic segmentation** on food images using a **U-Net with a VGG16 backbone**. Instead of assigning one label to an entire image, the model predicts a mask so each pixel belongs to a food category or the background.

This repository includes:
- data loading for train and test images/masks
- preprocessing and class remapping
- training a U-Net segmentation model
- generating predicted masks
- evaluating segmentation performance


## Why U-Net for Food Segmentation

Although U-Net is most often associated with **medical imaging**, its encoder-decoder structure also makes it a strong fit for segmentation problems outside medicine. In this project, it was used for **food segmentation**, showing that the same architecture can be adapted to identify food regions and boundaries in meal images.

This creates opportunities for applications such as:
- food item localization
- portion-region analysis
- meal composition understanding
- future nutrition-related computer vision tasks

## 10-Class Setup

The presentation focuses on the reduced **10-class version** of the project:

- Background = pixel 0
- Rice = pixel 1
- Green Beans = pixel 2
- French Fries = pixel 3
- Carrots = pixel 4
- Shrimp = pixel 5
- Steak = pixel 6
- Onion = pixel 7
- Tomato = pixel 8
- Egg = pixel 9

These mappings are shown directly in the project presentation.

## Model Architecture

- **Architecture:** U-Net
- **Backbone:** VGG16
- **Framework:** Keras / TensorFlow
- **Input size:** 128 x 128
- **Activation:** Softmax
- **Optimizer:** Adam
- **Loss:** Dice Loss + Categorical Focal Loss
- **Metrics:** Accuracy, IoU Score, F-Score
- **Epochs:** 50
- **Batch size:** 8

The implementation file shows the 10-class training pipeline and saving the trained model as `vgg_backbone_10_classes_50epochs.hdf5`. 

## Results

The presentation shows the model results visually using:
- the original image
- the ground truth mask
- the predicted mask

This side-by-side comparison demonstrates that the model is able to learn meaningful food-region boundaries and separate food items from the background. The prediction examples in the presentation are labeled as **Image**, **Ground Truth Mask**, and **Predicted Mask**.

For the **10-class model**, the training screenshot from the presentation shows the final results at the end of training were approximately:

- **Epoch:** 50/50
- **Validation Accuracy:** 0.8422
- **Validation IoU Score:** 0.5732
- **Validation F1-Score:** 0.6290
- **Validation Loss:** 0.3052

Across the final epochs, validation accuracy stayed around **0.84–0.845**, and validation IoU stayed around **0.55–0.57**, showing fairly stable performance near the end of training.

## Repository Structure

```bash
food-image-segmentation-unet/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── models/
│   └── vgg_backbone_10_classes_50epochs.hdf5
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
└── full_implementation.py
