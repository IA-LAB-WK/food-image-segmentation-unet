
"""
Created on Sun Mar 29 12:28:00 2024

@author: ismaelal-hadhrami
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import keras
import segmentation_models as sm
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt


SIZE_X = 128 
SIZE_Y = 128
n_classes= 104 #Number of classes for segmentation


########################################################
#script_directory = os.getcwd()
images = []
#directory = os.path.join(script_directory, '.spyder\FoodSeg103\Images\img_dir/train')
directory = '.spyder-py3/FoodSeg103/Images/img_dir/train/'
file_pattern = '*.jpg'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X)) 
    images.append(img)
images = np.array(images)

plt.imshow(images[0])

##########################################
masks = []
directory = '.spyder-py3/FoodSeg103/Images/ann_dir/train/'
file_pattern = '*.png'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    mask = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE )       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X)) 
    masks.append(mask)
masks = np.array(masks)

#for k in range(masks.shape[0]):
#    for y in range(masks.shape[1]):
#        for x in range(masks.shape[2]):
#            if masks[k,y,x] >5:
#                masks[k,y,x] = 0
                
#new_train_masks = []
#new_train_images = []

#for k in range(masks.shape[0]):
#    if np.any(masks[k]):
#        count = 0
#        for y in range(masks.shape[1]):
#            for x in range(masks.shape[2]):
#               if masks[k,y,x] > 0:
#                    count = count + 1
 #       if count > 200:           
 #           new_train_masks.append(masks[k])
 #           new_train_images.append(images[k])
            
#new_train_masks = np.array(new_train_masks)
#new_train_images= np.array(new_train_images)
#############################################################
images_test = []
directory = '.spyder-py3/FoodSeg103/Images/img_dir/test/'
file_pattern = '*.jpg'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X)) 
    images_test.append(img)
images_test = np.array(images_test)

###################################
masks_test = []
directory = '.spyder-py3/FoodSeg103/Images/ann_dir/test/'
file_pattern = '*.png'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    mask = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE )       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X)) 
    masks_test.append(mask)
masks_test = np.array(masks_test)

#for k in range(masks_test.shape[0]):
#    for y in range(masks_test.shape[1]):
 #       for x in range(masks_test.shape[2]):
 #           if masks_test[k,y,x] >5:
#                masks_test[k,y,x] = 0
                
#new_test_masks = []
#new_test_images = []

#for k in range(masks_test.shape[0]):
#    if np.any(masks_test[k]):
#        count = 0
 #       for y in range(masks_test.shape[1]):
#            for x in range(masks_test.shape[2]):
 #               if masks_test[k,y,x] > 0:
 #                   count = count + 1
 #       if count > 200:           
 #           new_test_masks.append(masks_test[k])
  #          new_test_images.append(images_test[k])
            
#new_test_masks = np.array(new_test_masks)
#new_test_images= np.array(new_test_images)

#plt.imshow(new_test_images[0])
#plt.imshow(new_test_masks[0])

##############################
#new_train_masks = np.expand_dims(new_train_masks, axis=3)
#new_test_masks= np.expand_dims(new_test_masks, axis=3)
##############################
masks = np.expand_dims(masks, axis=3)
masks_test = np.expand_dims(masks_test, axis=3)


from keras.utils import to_categorical
train_masks_cat = to_categorical(masks, num_classes=104)
train_masks_cat = train_masks_cat.reshape((masks.shape[0], masks.shape[1], masks.shape[2], 104))



test_masks_cat = to_categorical(masks_test, num_classes=104)
test_masks_cat = test_masks_cat.reshape((masks_test.shape[0], masks_test.shape[1], masks_test.shape[2], 104))
#########################

#np.unique(new_train_masks)
np.unique(masks)

#new_train_images = new_train_images/255.0
#new_test_images = new_test_images/255.0

########################
activation ='softmax'
LR = 0.0001
optim = keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss(class_weights=np.full(104,0.25))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
#########################################

BACKBONE1 = 'vgg16'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

X_train1 = preprocess_input1(images)
X_test1 = preprocess_input1(images_test)

######################################
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=104, activation=activation)
model1.compile(optim, total_loss, metrics=metrics)


history1 = model1.fit(X_train1,train_masks_cat , 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test1, test_masks_cat))


model1.save('vgg_backbone_104_classes_50epochs.hdf5')
######################################################################
pred = model1.predict(X_test1[:2])
pred_argmax = np.argmax(pred[1], axis=2)
cat_example = pred[0]
one_pred = pred[1]
real_mask = test_masks_cat[1]
plt.imshow(X_test1[1])

plt.imshow(np.argmax(test_masks_cat[1], axis=2))
plt.imshow(pred_argmax)
pred_argmax.shape
print(np.unique(pred_argmax))

unique_pixels, counts = np.unique(pred_argmax, return_counts=True)

pixel_counts= dict(zip(unique_pixels, counts))

for pixel, count in pixel_counts.items():
    print(f'{pixel}: {count}')

##############################################################################
pred2 = model1.predict(X_test1[:20])
pred2_argmax = np.argmax(pred2[19], axis=2)

plt.imshow(np.argmax(test_masks_cat[19], axis=2))
plt.imshow(pred2_argmax)

unique_pixels2, counts2 = np.unique(pred2_argmax, return_counts=True)

pixel_counts2= dict(zip(unique_pixels2, counts2))

for pixel, count in pixel_counts2.items():
    print(f'{pixel}: {count}')
#######################################################################
from keras.metrics import MeanIoU
n_classes = 104
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(np.argmax(test_masks_cat[1], axis=2), pred_argmax)
print("Mean IOU = ", IOU_keras.result().numpy())

##################################################################################
#New model on only 5 items


SIZE_X = 128 
SIZE_Y = 128
n_classes= 10 #Number of classes for segmentation


########################################################
#script_directory = os.getcwd()
images2 = []
#directory = os.path.join(script_directory, '.spyder\FoodSeg103\Images\img_dir/train')
directory = '.spyder-py3/FoodSeg103/Images/img_dir/train/'
file_pattern = '*.jpg'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X)) 
    images2.append(img)
images2 = np.array(images2)

plt.imshow(images2[0])

##########################################
masks2 = []
directory = '.spyder-py3/FoodSeg103/Images/ann_dir/train/'
file_pattern = '*.png'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    mask = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE )       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X)) 
    masks2.append(mask)
masks2 = np.array(masks2)


#############################################################
images_test2 = []
directory = '.spyder-py3/FoodSeg103/Images/img_dir/test/'
file_pattern = '*.jpg'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X)) 
    images_test2.append(img)
images_test2 = np.array(images_test2)

###################################
masks_test2 = []
directory = '.spyder-py3/FoodSeg103/Images/ann_dir/test/'
file_pattern = '*.png'

image_paths = glob.glob(f'{directory}/{file_pattern}')
sorted_image_path = sorted(image_paths)

for image_path in sorted_image_path:
    mask = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE )       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X)) 
    masks_test2.append(mask)
masks_test2 = np.array(masks_test2)


for k in range(masks2.shape[0]):
    for y in range(masks2.shape[1]):
        for x in range(masks2.shape[2]):
            #emptying slot 1,2,4,5
            if masks2[k,y,x] == 1:
                masks2[k,y,x]= 0
            if masks2[k,y,x]== 2:
                masks2[k,y,x]=0
            if masks2[k,y,x]== 4:
                masks2[k,y,x]=0
            if masks2[k,y,x]== 5:
                masks2[k,y,x]=0
            if masks2[k,y,x]== 6:
                masks2[k,y,x]=0
            if masks2[k,y,x]== 7:
                masks2[k,y,x]=0
            if masks2[k,y,x]== 8: 
                masks2[k,y,x]=0
            if masks2[k,y,x]== 9:
                masks2[k,y,x]=0
                 
                 
for k in range(masks2.shape[0]):
    for y in range(masks2.shape[1]):
        for x in range(masks2.shape[2]):
            #emptying slot 1,2,4,5
            if masks2[k,y,x] == 66:
                masks2[k,y,x]= 1
            if masks2[k,y,x]== 95:
                masks2[k,y,x]= 2
            if masks2[k,y,x]== 84:
                 masks2[k,y,x]= 4
            if masks2[k,y,x]== 56:
                 masks2[k,y,x]=5 
            if masks2[k,y,x]== 46:
                 masks2[k,y,x]=6
            if masks2[k,y,x]== 93:
                 masks2[k,y,x]=7   
            if masks2[k,y,x]== 73:
                 masks2[k,y,x]=8
            if masks2[k,y,x]== 24:
                 masks2[k,y,x]=9
 
    
 
for k in range(masks2.shape[0]):
    for y in range(masks2.shape[1]):
       for x in range(masks2.shape[2]):
           if masks2[k,y,x] >9:
                masks2[k,y,x] = 0
                

np.unique(masks2)

###########################################
new_train_masks2 = []
new_train_images2 = []
###########################################

for k in range(masks2.shape[0]):
   if np.any(masks2[k]):
       count = 0
       for y in range(masks2.shape[1]):
            for x in range(masks2.shape[2]):
               if masks2[k,y,x] > 0:
                   count = count + 1
       if count > 300:           
           new_train_masks2.append(masks2[k])
           new_train_images2.append(images2[k])
           
##################################################################           
           
for k in range(masks_test2.shape[0]):
    for y in range(masks_test2.shape[1]):
        for x in range(masks_test2.shape[2]):
            #emptying slot 1,2,4,5
            if masks_test2[k,y,x] == 1:
                masks_test2[k,y,x]= 0
            if masks_test2[k,y,x]== 2:
                masks_test2[k,y,x]=0
            if masks_test2[k,y,x]== 4:
                masks_test2[k,y,x]=0
            if masks_test2[k,y,x]== 5:
                masks_test2[k,y,x]=0
            if masks_test2[k,y,x]== 6:
                masks_test2[k,y,x]=0
            if masks_test2[k,y,x]== 7:
                masks_test2[k,y,x]=0
            if masks_test2[k,y,x]== 8: 
                masks_test2[k,y,x]=0
            if masks_test2[k,y,x]== 9:
                masks_test2[k,y,x]=0
                 
                 
for k in range(masks_test2.shape[0]):
    for y in range(masks_test2.shape[1]):
        for x in range(masks_test2.shape[2]):
            #emptying slot 1,2,4,5
            if masks_test2[k,y,x] == 66:#RICE
                masks_test2[k,y,x]= 1
            if masks_test2[k,y,x]== 95:#GREEN BEANS
                masks_test2[k,y,x]= 2
            if masks_test2[k,y,x]== 84:#CARROT
                 masks_test2[k,y,x]= 4
            if masks_test2[k,y,x]== 56:#SHRIMP
                 masks_test2[k,y,x]=5 
            if masks_test2[k,y,x]== 46:#STEAK
                 masks_test2[k,y,x]=6
            if masks_test2[k,y,x]== 93:#ONION
                 masks_test2[k,y,x]=7   
            if masks_test2[k,y,x]== 73:#TOMATO
                 masks_test2[k,y,x]=8
            if masks_test2[k,y,x]== 24:#EGG
                 masks_test2[k,y,x]=9
 
    
 
for k in range(masks_test2.shape[0]):
    for y in range(masks_test2.shape[1]):
       for x in range(masks_test2.shape[2]):
           if masks_test2[k,y,x] >9:
                masks_test2[k,y,x] = 0
                
#################################
new_test_masks2 = []
new_test_images2 = []  
#################################

for k in range(masks_test2.shape[0]):
   if np.any(masks_test2[k]):
       count = 0
       for y in range(masks_test2.shape[1]):
            for x in range(masks_test2.shape[2]):
               if masks_test2[k,y,x] > 0:
                   count = count + 1
       if count > 300:           
           new_test_masks2.append(masks_test2[k])
           new_test_images2.append(images_test2[k])
           
###############################################################
new_train_images2= np.array(new_train_images2)
new_train_masks2 = np.array(new_train_masks2)
new_test_images2= np.array(new_test_images2)
new_test_masks2 = np.array(new_test_masks2)



new_train_images2.shape
new_train_masks2.shape
new_test_images2.shape
new_test_masks2.shape

###############################
new_train_masks2 = np.expand_dims(new_train_masks2, axis=3)
new_test_masks2 = np.expand_dims(new_test_masks2, axis=3)

#new_train_images2= new_train_images2[:1600]
#new_train_masks2= new_train_masks2[:1600]


from keras.utils import to_categorical
train_masks_cat2 = to_categorical(new_train_masks2, num_classes=10)
train_masks_cat2 = train_masks_cat2.reshape((new_train_masks2.shape[0], new_train_masks2.shape[1], new_train_masks2.shape[2], 10))



test_masks_cat2 = to_categorical(new_test_masks2, num_classes=10)
test_masks_cat2 = test_masks_cat2.reshape((new_test_masks2.shape[0], new_test_masks2.shape[1], new_test_masks2.shape[2], 10))
#########################

activation ='softmax'
LR = 0.0001
optim = keras.optimizers.Adam(LR)

dice_loss1 = sm.losses.DiceLoss(class_weights=np.full(10,0.25))
focal_loss1 = sm.losses.CategoricalFocalLoss()
total_loss1 = dice_loss1 + (1 * focal_loss1)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
#########################################

BACKBONE2 = 'vgg16'
preprocess_input1 = sm.get_preprocessing(BACKBONE2)

X_train2 = preprocess_input1(new_train_images2)
X_test2 = preprocess_input1(new_test_images2)

######################################
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=10, activation=activation)
model2.compile(optim, total_loss1, metrics=metrics)


history1 = model2.fit(X_train2,train_masks_cat2 , 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test2, test_masks_cat2))



model2.save('vgg_backbone_10_classes_50epochs.hdf5')
############################################################

pred2 = model2.predict(X_test2[:30])
pred2_argmax = np.argmax(pred2[9], axis=2)

plt.imshow(np.argmax(test_masks_cat2[9], axis=2))
plt.imshow(pred2_argmax)

plt.imshow(X_test2[9])

unique_pixels2, counts2 = np.unique(pred2_argmax, return_counts=True)

pixel_counts2= dict(zip(unique_pixels2, counts2))

for pixel, count in pixel_counts2.items():
    print(f'{pixel}: {count}')
    
    
