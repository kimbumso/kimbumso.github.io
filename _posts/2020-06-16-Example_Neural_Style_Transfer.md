---
layout: post
title: "Neural Style Transfer"
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

# 출처

[Neural Style Transfer 링크](https://github.com/anishathalye/neural-style)

[paper 링크](https://arxiv.org/pdf/1610.07629.pdf)

[원문 링크](https://www.anishathalye.com/2015/12/19/an-ai-that-can-mimic-any-artist/)

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
import os
import math
import re
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import numpy as np
import scipy.misc

from stylize import stylize
~~~

모델은 `pretrain`된 모델을 사용한다.