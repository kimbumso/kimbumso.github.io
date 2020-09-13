---
layout: post
title: "YOLO4"
description: 
headline: 
modified: 2020-09-13
category: DataScience
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# YOLO4 사용

<p>&nbsp;</p>

[YOLO4 참고 강의 링크](https://www.youtube.com/watch?v=D6mj_T8K_bo)

[YOLO4 시연 강의 링크](https://www.youtube.com/watch?v=hxwEqXCgQO4&t=335s)

[GIT 링크](https://github.com/kimbumso/tensorflow-yolov4-tflite)

[paper 링크](https://arxiv.org/abs/2004.10934)


<p>&nbsp;</p>

## GIT에서 LOCAL로 code 내려받기

<p>&nbsp;</p>

## PAKAGE import
~~~python
!pip install -r requirements.txt 
~~~

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_170122.png">

<p>&nbsp;</p>

## pretrain 된 Model 다운받기

~~~python

#model.py
Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
~~~

<p>&nbsp;</p>

## DarkNet 모델을 Tensorflow 모델로 변환히기

~~~python
## yolov4
!python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
~~~

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_170313.png">

<p>&nbsp;</p>

## 이미지 및 동영상 테스트

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_170433.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_172628.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_173027.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_173113.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/YOLO4/20200913_173156.png">