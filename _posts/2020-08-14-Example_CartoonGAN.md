---
layout: post
title: "CartoonGAN"
description: 
headline: 
modified: 2020-08-14
category: Example
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# 출처

[CartoonGAN 링크](https://www.youtube.com/watch?v=4tkhKTY6R7g)

[paper 링크](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf)

[model 링크](http://vllab1.ucmerced.edu/~yli62/CartoonGAN/)

<img src="{{ site.url }}/images/practice/tf.jpg">

*이탤릭* **볼드** ***이탤릭볼드***

기본적으로 설치되어 있어야하는 패키지는 `아래 코드` 를 사용한다.

~~~python
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append('/content/drive/My Drive/Youtube_Practice/')  # local package import를 위해 경로 지정

from CartoonGAN.network import Transformer  # 사용자 module
~~~

## 모델은 `pretrain`된 모델을 사용한다.

## http://vllab1.ucmerced.edu/~yli62/CartoonGAN/

~~~python
model = Transformer.Transformer()
model.load_state_dict(torch.load('/content/drive/My Drive/Youtube_Practice/CartoonGAN/pretrained_model/Shinkai_net_G_float.pth'))
model.eval()
print('Model loaded!')
~~~

## 이미지 전처리

~~~python
img_size = 450
img_path = '/content/drive/My Drive/Youtube_Practice/CartoonGAN/test_img/test.jpg'

img = cv2.imread(img_path)

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size, 2),
    transforms.ToTensor()
])

img_input = T(img).unsqueeze(0)

img_input = -1 + 2 * img_input # 0 - 1 -> -1 - +1

plt.figure(figsize=(16, 10))
plt.imshow(img[:, :, ::-1])  # bgr ->rgb
~~~

<img src="{{ site.url }}/images/practice/CarttonGAN/20200814_130255.png">

## 적용
### Shinkai_net_G_float.pth

~~~python
img_output = model(img_input)

img_output = (img_output.squeeze().detach().numpy() + 1.) / 2.
img_output = img_output.transpose([1, 2, 0]) # pytorch(채널, 높이, 너비) -> matplotlib(높이, 너비, 채널)

fig, axes = plt.subplots(1, 2, figsize=(16, 16))
axes[0].imshow(img[:, :, ::-1])
axes[1].imshow(img_output[:, :, ::-1])
~~~

<img src="{{ site.url }}/images/practice/CarttonGAN/01.png">

### Hayao_net_G_float.pth

~~~python
img_output = model(img_input)

img_output = (img_output.squeeze().detach().numpy() + 1.) / 2.
img_output = img_output.transpose([1, 2, 0]) # pytorch(채널, 높이, 너비) -> matplotlib(높이, 너비, 채널)

fig, axes = plt.subplots(1, 2, figsize=(16, 16))
axes[0].imshow(img[:, :, ::-1])
axes[1].imshow(img_output[:, :, ::-1])
~~~

<img src="{{ site.url }}/images/practice/CarttonGAN/02.png">

### Paprika_net_G_float.pth

~~~python
img_output = model(img_input)

img_output = (img_output.squeeze().detach().numpy() + 1.) / 2.
img_output = img_output.transpose([1, 2, 0]) # pytorch(채널, 높이, 너비) -> matplotlib(높이, 너비, 채널)

fig, axes = plt.subplots(1, 2, figsize=(16, 16))
axes[0].imshow(img[:, :, ::-1])
axes[1].imshow(img_output[:, :, ::-1])
~~~

<img src="{{ site.url }}/images/practice/CarttonGAN/03.png">