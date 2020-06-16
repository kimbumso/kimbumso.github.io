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


## 라벨링

~~~python

age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

~~~

<img src="{{ site.url }}/images/practice/Age_Gender_Classification/Screenshot_2020-06-16-17-05-57.png">

## 모델 사용

~~~python

detector = dlib.get_frontal_face_detector()  # 얼굴 인식 모듈

age_net = cv2.dnn.readNetFromCaffe(
          'models/deploy_age.prototxt', 
          'models/age_net.caffemodel')  # caffe 로 작된된 모델을 로드
gender_net = cv2.dnn.readNetFromCaffe(
          'models/deploy_gender.prototxt', 
          'models/gender_net.caffemodel')  # caffe 로 작된된 모델을 로드


~~~

<img src="{{ site.url }}/images/practice/Age_Gender_Classification/Screenshot_2020-06-16-17-06-37.png">

## Main

~~~python

img_list = glob.glob('img/*.jpg')  # test할 이미지 불러오기

for img_path in img_list:
  img = cv2.imread(img_path)  # 이미지 로드

  faces = detector(img)  # 얼굴을 찾음

  for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()  # 얼굴의 위치정보

    face_img = img[y1:y2, x1:x2].copy()  # 이미지에서 얼굴만 추출

    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)  # blobFromImage 바이너리 데이터로 변환 

    # predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()  # 테스트 하기 위한 값
    gender = gender_list[gender_preds[0].argmax()]  # softmax로 모델의 output이 나오기 때문에 확률값으로 나옴 이걸 argmax로 정수형으로 변환. [0.7, 0.3] -> [1, 0]

    # predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()  # 테스트 하기 위한 값
    age = age_list[age_preds[0].argmax()]  # softmax로 모델의 output이 나오기 때문에 확률값으로 나옴 이걸 argmax로 정수형으로 변환. [0.7, 0.3] -> [1, 0]

    # visualize
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
    overlay_text = '%s %s' % (gender, age)
    cv2.putText(img, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=1, color=(0,0,0), thickness=10)  # 나이와 성별을 putText를 이용해서 씀
    cv2.putText(img, overlay_text, org=(x1, y1),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

~~~

<img src="{{ site.url }}/images/practice/Age_Gender_Classification/Screenshot_2020-06-16-17-08-45.png">

## Test

~~~python

  cv2.imshow('img', img)
  cv2.imwrite('result/%s' % img_path.split('/')[-1], img)

  key = cv2.waitKey(0) & 0xFF
  if key == ord('q'):
    break

~~~

<img src="{{ site.url }}/images/practice/Age_Gender_Classification/Screenshot_2020-06-16-17-09-43.png">