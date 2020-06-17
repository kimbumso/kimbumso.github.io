---
layout: post
title: "CT Lung Segmentation"
description: 
headline: 
modified: 2020-06-16
category: Example
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# Generate Image

[CT Lung Segmentation 링크](https://github.com/kairess/CT_lung_segmentation)

[paper 링크](-)

[원문 링크](-)

[dataset 링크](https://www.kaggle.com/kmader/finding-lungs-in-ct-data)

<img src="{{ site.url }}/images/practice/tf.jpg">

*이탤릭* **볼드** ***이탤릭볼드***

## ***Workflow stages***
1. Question or problem definition.
2. Acquire training and testing data.
3. Wrangle, prepare, cleanse the data.
4. Analyze, identify patterns, and explore the data.
5. Model, predict and solve the problem.
6. Visualize, report, and present the problem solving steps and final solution.
7. Supply or submit the results.
 
기본적으로 설치되어 있어야하는 패키지는 `아래 코드` 를 사용한다.

~~~python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize

import os, glob

from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
~~~

## Data 전처리

~~~python

img_list = sorted(glob.glob('2d_images/*.tif'))
mask_list = sorted(glob.glob('2d_masks/*.tif'))

IMG_SIZE = 256

x_data, y_data = np.empty((2, len(img_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

for i, img_path in enumerate(img_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    x_data[i] = img
    
for i, img_path in enumerate(mask_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    y_data[i] = img
    
y_data /= 255.

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_data[12].squeeze(), cmap='gray')
ax[1].imshow(y_data[12].squeeze(), cmap='gray')

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)

np.save('dataset/x_train.npy', x_train)
np.save('dataset/y_train.npy', y_train)
np.save('dataset/x_val.npy', x_val)
np.save('dataset/y_val.npy', y_val)

~~~


## Load Dataset
# Encoder 차원 축소를 통해 핵심요소 추출 (DownSampling, ex MaxPooling2D)
# Decoder 압축된 정보로부터 차원 확장을 통해 원하는 정보로 복원 ( UpsSampling , Upsampling2D)

~~~python

x_train = np.load('dataset/x_train.npy')  # CT
y_train = np.load('dataset/y_train.npy')  # Mask
x_val = np.load('dataset/x_val.npy')s  # CT정답
y_val = np.load('dataset/y_val.npy')  # Mask 정답

~~~

<img src="{{ site.url }}/images/practice/CT_Lung_Segmentation/Screenshot_2020-06-17-10-10-18.png">

## 모델 사용

~~~python

dinputs = Input(shape=(256, 256, 1))

net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(128, activation='relu')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])

model.summary()

# tf 2.x 부터는

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=3 , activation='relu', padding='same', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.UpSampling2D(size=2))
model.add(layers.Conv2D(128, kernel_size=3, activation='sigmoid', padding='same'))
model.add(layers.UpSampling2D(size=2))
model.add(layers.Conv2D(64, kernel_size=3, activation='sigmoid', padding='same'))
model.add(layers.UpSampling2D(size=2))
model.add(layers.Conv2D(1, kernel_size=3 , activation='sigmoid', padding='same'))

model.summary()
~~~

<img src="{{ site.url }}/images/practice/CT_Lung_Segmentation/Screenshot_2020-06-17-10-14-52.png">

## Train

~~~python

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
])

~~~

<img src="{{ site.url }}/images/practice/CT_Lung_Segmentation/Screenshot_2020-06-17-13-58-01.png">

## Evaluation

~~~python

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')

~~~

<img src="{{ site.url }}/images/practice/CT_Lung_Segmentation/Screenshot_2020-06-17-13-58-54.png">

<img src="{{ site.url }}/images/practice/CT_Lung_Segmentation/Screenshot_2020-06-17-13-59-14.png">