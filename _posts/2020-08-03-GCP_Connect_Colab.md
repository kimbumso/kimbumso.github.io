---
layout: post
title: "GCP bucket Colab 연동"
description: 
headline: 
modified: 2020-08-03
category: GCP
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---


[GCP TPU 관련 링크](https://beomi.github.io/2020/02/26/Train-BERT-from-scratch-on-colab-TPU-Tensorflow-ver/)

[paper 링크](-)

[dataset 링크](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

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
 

## GSC 버킷 접근

~~~python

from google.colab import auth
auth.authenticate_user()

~~~


## GCP 버킷에서 Colab으로 data 가져오기

~~~python

! mkdir datasets
!gsutil -m cp -R gs://bskim_bucket/datasets/BSR/BSDS500/data/images datasets
~~~

