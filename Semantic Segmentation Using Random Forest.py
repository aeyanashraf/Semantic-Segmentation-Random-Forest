
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle

from keras.models import Sequential, Model
from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16


print(os.listdir("F:/python/images"))

SIZE_X = 1024
SIZE_Y = 996


train_images = []

for directory_path in glob.glob("F:/python/images/train_images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
      
      
train_images = np.array(train_images)

train_masks = [] 
for directory_path in glob.glob("F:/python/images/train_masks"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        train_masks.append(mask)
        
train_masks = np.array(train_masks)

X_train = train_images
y_train = train_masks
y_train = np.expand_dims(y_train, axis=3)

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))


for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()


features=new_model.predict(X_train)


square = 8
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()

X=features
X = X.reshape(-1, X.shape[3])  

Y = y_train.reshape(-1)


dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

dataset = dataset[dataset['Label'] != 29]
#Redefine X and Y for Random Forest
X_for_RF = dataset.drop(labels = ['Label'], axis=1)
Y_for_RF = dataset['Label']


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_for_RF, Y_for_RF)

filename = 'RF_model_NB.sav'
pickle.dump(gnb, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))

test_img = cv2.imread('F:/python/images/Train_images/Sandstone_Versa0400.tif', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)


X_test_feature = new_model.predict(test_img)
X_test_feature = X_test_feature.reshape(-1, X_test_feature.shape[3])

prediction = loaded_model.predict(X_test_feature)


prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='gray')
plt.imsave('F:/python/images/test_images/segmented.jpg', prediction_image, cmap='gray')



