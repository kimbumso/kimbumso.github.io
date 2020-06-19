---
layout: post
title: "TFRecord 사용"
description: 
headline: 
modified: 2020-06-19
category: DataScience
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

[tf.data.TFRecordDataset 관련 링크](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)

[paper 링크](-)

[참고자료 링크](https://blog.naver.com/4u_olion/221578711592)

[dataset 링크](-)


<img src="{{ site.url }}/images/practice/tf.jpg">


기본적으로 설치되어 있어야하는 패키지는 `아래 코드` 를 사용한다.

~~~python

from google.colab import auth, drive  # GCP Bucket연동
from tensorflow.keras.utils import Progbar

auth.authenticate_user()  # GCP 연동을 위한 인증

import re, sys, time
import numpy as np
from matplotlib import pyplot as plt
if 'google.colab' in sys.modules: # Colab-only Tensorflow version selector
  %tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
~~~

## GCP Bucket 연동
~~~python

!git clone https://github.com/GoogleCloudPlatform/gcsfuse.git

!echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
!apt -qq update
!apt -qq install gcsfuse

~~~

## Bucket에서 Colab으로 Data를 가져오기 위한 폴더 생성 및 Bucket 지정 Bucket안에 Kaggle API Key 있는지 확인
~~~python

!mkdir folderOnColab
!gcsfuse bskim_kaggle_bucket folderOnColab

~~~

## kaggle API에 Kaggle Key를 이동

~~~python

!mkdir -p ~/.kaggle
!mv folderOnColab/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json	

~~~


## Data 셋 가져오기 
~~~python

!wget --show-progress --continue -O /content/shakespeare.txt http://www.gutenberg.org/files/100/100-0.txt

!head -n5 /content/shakespeare.txt
!echo "..."
!shuf -n5 /content/shakespeare.txt

~~~


## TPU 연동

~~~python

import re, sys, time
import numpy as np
import os
from matplotlib import pyplot as plt
if 'google.colab' in sys.modules: # Colab-only Tensorflow version selector
  %tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
elif len(gpus) > 1: # multiple GPUs in one VM
    strategy = tf.distribute.MirroredStrategy(gpus)
else: # default strategy that works on CPU and single GPU
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

~~~

~~~python

import distutils  # 버전딸리는것들 찾기
if distutils.version.LooseVersion(tf.__version__) < '1.14':
    raise Exception('This notebook is compatible with TensorFlow 1.14 or higher, for TensorFlow 1.13 or lower please use the previous version at https://github.com/tensorflow/tpu/blob/r1.13/tools/colab/shakespeare_with_tpu_and_keras.ipynb')

~~~

# TFrecord 

#### TFRecord 파일은 텐서플로우의 학습 데이타 등을 저장하기 위한 바이너리 데이타 포맷으로, 구글의 Protocol Buffer 포맷으로 데이타를 파일에 Serialize 하여 저장

~~~python

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

image_reader = ImageReader()
image_data = tf.gfile.GFile(path, 'rb').read()
height, width = image_reader.read_image_dims(sess, image_data)


#확장자는 tfrecord로 맞추도록 하자.
with tf.python_io.TFRecordWriter('path_to.tfrecord') as writer:
    example = dataset_utils.image_to_tfexample(
              image_data, b'png', height, width, label)
    writer.write(example.SerializeToString())

~~~

위 코드를 요약하자면, Image를 Read하여 Byte string 값으로 표현한다. 그리고, image의 height와 width 값을 구한다.

이렇게 구해진 3가지값과 Label 정보, 그리고 image format 형태를 tfrecord로 표현하기 위한 example을 만든다. 즉, Image의 정보를 String으로 가장 간단히 표현한 것이다.

~~~python

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

~~~

example은  이미지 format에 대해 image key에 대해서 encoded는 bytestring 값을,

format은 image format유형을, label은 class key를 추가하여 그 안에 label이라는 항목에 추가한다. 

즉, 위에서 만들어진 tfrecord feature format은 image라는 root key를 가지며, 그 안에 encoded부터 witdth까지 구성된다.

~~~python

{ "image" : {
	"encoded" : bytes string feature,
	"format"  : image format,
	"class"   : { label : image label },
	"height"  : image height,
	"width"   : image width 
 }
}

~~~