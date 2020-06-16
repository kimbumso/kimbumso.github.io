---
layout: post
title: "LSTM Fantasy Novel"
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

[LSTM Fantasy Novel 링크](https://github.com/kairess/deep_fantasy_novel)

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
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np

import random
import sys
import io
import re
~~~

## 전처리

~~~python

# path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

path = 'barkers.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

text = re.sub(r'<.*>', '', text)
text = re.sub(r'\n', ' ', text)
text = re.sub(r' +', ' ', text)

# Compute Character Indices
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))  # 원핫인코딩으로 class화 함

## Vectorize Sentences
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])  # input 40글자씩
    next_chars.append(text[i + maxlen])  # output 1글자
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)  # input
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)  # output

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1  # one-hot encording
    y[i, char_indices[next_chars[i]]] = 1  # one-hot encording

~~~

## 모델 생성

~~~python

print('Build model...')
model = Sequential()
model.add(LSTM(1024, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))

~~~

## 콜백 생성

~~~python


def sample(preds, temperature=1.0):  # Loop를 방지하기 위해 확률을 조작. 긍정문 같은 예, 아니오 등이 반복되는 것을 방지
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):  # 에폭이 끝날 때마다 실행
    print('\n----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)  # 랜덤으로 40개의 시드를 생성
#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]  # 힌트
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):  # 400개의 글자를 예측
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)  # 람다콜백에 등록

~~~


## Train

~~~python

model.fit(x, y, batch_size=128, epochs=60, callbacks=[print_callback])

~~~
