---
layout: post
title: "Generate Image"
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

[Generate Image 링크](https://github.com/kairess/genetic_image)

[paper 링크](-)

[원문 링크](-)

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
import cv2, random, os, sys
import numpy as np
from copy import deepcopy
from skimage.measure import compare_mse
import multiprocessing as mp
~~~



## data 가져오기

~~~python
filepath = sys.argv[1]
filename, ext = os.path.splitext(os.path.basename(filepath))

img = cv2.imread(filepath)  # 이미지를 BGR 형식으로 읽음
height, width, channels = img.shape
~~~

## hyperparameters 설정

~~~python

n_initial_genes = 50  # 첫번째 세대의 유전자 갯수
n_population = 50  # 한 세대당 유전자 그룹 수
prob_mutation = 0.01  # 돌연변이가 발생할 확률
prob_add = 0.3  # 유전자 그룹에 원이 추가될 확률
prob_remove = 0.2  # 유전자 그룹에 원이 제거될 확률

min_radius, max_radius = 5, 15
save_every_n_iter = 100  # 이미지를 100세대당 저장
~~~



## 유전자 클래스 지정

~~~python

class Gene():
  def __init__(self):
    self.center = np.array([random.randint(0, width), random.randint(0, height)])  # 캔버스를 넘지 않도록 랜덤으로 지정
    self.radius = random.randint(min_radius, max_radius)  
    self.color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])  # B , G , R 순서로 랜덤으로 작성

  def mutate(self):  # 변형하는 코드
    mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100  # 돌연변이의 폭은 가우시안 분포로 평균 15에 표준편차 4 의 숫자를 100으로 나눈 값을 사용

    r = random.uniform(0, 1)
    if r < 0.33: # radius
      self.radius = np.clip(random.randint(
        int(self.radius * (1 - mutation_size)),
        int(self.radius * (1 + mutation_size))
      ), 1, 100)
    elif r < 0.66: # center
      self.center = np.array([
        np.clip(random.randint(
          int(self.center[0] * (1 - mutation_size)),
          int(self.center[0] * (1 + mutation_size))),
        0, width),
        np.clip(random.randint(
          int(self.center[1] * (1 - mutation_size)),
          int(self.center[1] * (1 + mutation_size))),
        0, height)
      ])
    else: # color
      self.color = np.array([
        np.clip(random.randint(
          int(self.color[0] * (1 - mutation_size)),
          int(self.color[0] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[1] * (1 - mutation_size)),
          int(self.color[1] * (1 + mutation_size))),
        0, 255),
        np.clip(random.randint(
          int(self.color[2] * (1 - mutation_size)),
          int(self.color[2] * (1 + mutation_size))),
        0, 255)
      ])

~~~

## 함수 작성

~~~python

# compute fitness
def compute_fitness(genome):
  out = np.ones((height, width, channels), dtype=np.uint8) * 255  # 얼마나 잘 적응했는지 판별하는 함수. 255로 채워서 백색으로 나옴

  for gene in genome:  # 원을 그림
    cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(int(gene.color[0]), int(gene.color[1]), int(gene.color[2])), thickness=-1)  # thickness=-1 원의 색을 채우기 위해 

  # mean squared error
  fitness = 255. / compare_mse(img, out)  # 원본이미지(img), 생성이미지(out) 의 차이를 구함 mse가 작으면 좋기 때문에 역수를 취해줌

  return fitness, out

# compute population
def compute_population(g):
  genome = deepcopy(g)
  # mutation
  if len(genome) < 200:
    for gene in genome:
      if random.uniform(0, 1) < prob_mutation:
        gene.mutate()
  else:
    for gene in random.sample(genome, k=int(len(genome) * prob_mutation)):  # 랜덤으로 뽑아서 변이를 시키는게 속도와 적중이 좋음
      gene.mutate()

  # add gene
  if random.uniform(0, 1) < prob_add:
    genome.append(Gene())  # 원이 ex 50개면 -> 51개로 됨

  # remove gene
  if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
    genome.remove(random.choice(genome)) # 원이 ex 50개면 -> 49개로 됨

  # compute fitness
  new_fitness, new_out = compute_fitness(genome)  # 새로운 유전자의 점수를 측정해서 return

  return new_fitness, genome, new_out


~~~


## Main

~~~python

if __name__ == '__main__':
  os.makedirs('result', exist_ok=True)

  p = mp.Pool(mp.cpu_count() - 1)  # 병렬처리를 위해서 pool 지정

  # 1st gene
  best_genome = [Gene() for _ in range(n_initial_genes)] 

  best_fitness, best_out = compute_fitness(best_genome)

  n_gen = 0

  while True:
    try:
      results = p.map(compute_population, [deepcopy(best_genome)] * n_population)
    except KeyboardInterrupt:
      p.close()
      break

    results.append([best_fitness, best_genome, best_out])

    new_fitnesses, new_genomes, new_outs = zip(*results)

    best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)  # fitness 점수에 따라 내림차순으로 정렬

    best_fitness, best_genome, best_out = best_result[0]

    # end of generation
    print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
    n_gen += 1

    # visualize
    if n_gen % save_every_n_iter == 0:
      cv2.imwrite('result/%s_%s.jpg' % (filename, n_gen), best_out)

    cv2.imshow('best out', best_out)
    if cv2.waitKey(1) == ord('q'):
     p.close()
     break

~~~


<img src="{{ site.url }}/images/practice/Generic_Image/Screenshot_2020-06-16-11-28-04.png">