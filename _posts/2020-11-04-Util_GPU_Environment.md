---
layout: post
title: "GPU Environment"
description: 
headline: 
modified: 2020-11-04
category: util
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# GPU 환경 세팅

<p>&nbsp;</p>

## GPU 병렬 프로세싱을 위한 기본 용어 정리.

### 1 Nvidia GPU  
#### Tesla - Kepler - Maxwell - Pascal(2016) - Volta(2017, V100) - Turing(2018, T4)
#### V100(고성능) vs T4 (저전력, 비용대비 고효율)

<p>&nbsp;</p>

### 2 PCI-E, PSU (전력상의 이유로 서버 셋업시 고려)
#### PCI-E switch를 통해서 GPU 자원 share. (16line, 8line) 
#### UPI(Ultra Path Interconnect) / QPI(Quick Path Interconnect)
#### NVLINK (GPU의 효과적인 커넥트를 위한 기술, 인텔은 제공 안함. IBM은 가능)
#### NVLINK TOPOLOGY(8개이상 GPU를 사용할때 Bottleneck에 강점을 보임, 8개이상부터는 NVSwitch를 통해서 제공)
#### GPUDierect RDMA(GPU <-> Third Party Device)
#### DGX-1 (8개 GPU, 위의 기술 통합)
#### DGX-2 (16개 GPU, 위의 기술 통합)

<p>&nbsp;</p>

### 3 CUDA Architecture (병렬처리 software flatform)
#### CUDA-X , 다양한 프레임워크에 잘 연동하여 가속화된 라이브러리를 제공
#### 다양한 도메인별로 지원. 
#### Vision, Speech, Audio, NLP  - 하단
#### DeepLearning Framework - 중단 
#### Nvidia Deep Learning SDK - 상단

<p>&nbsp;</p>

### 4 DATA CENTER
#### RACK당 최소 20kW 이상의 전력공급 검토가 필요.
#### Network - RDMA가 TCP보다 매우 효과적임. 
#### TCP로 구성할때는 L2로 구성하는게 좋음. (스위치의 부하를 잘 계산해서 pot 구성)
#### 2~4개의 GPU당 1개의 Network card를 꽂는게 좋음.
#### LATENCY, BANDWIDTH (IB, RoCE, Ethernet)
#### Storage 고려사항 (Capacity, IOPS, Bandwidth)
#### GPFS(딥러닝에선 비효율), HDFS(딥러닝의 워크로드와는 다름), NFS(보편적으로 많이 사용)

<p>&nbsp;</p>

### 5 TREND(year, PetaFlop/s-day)
#### (2012 0.01) AlexNet, Dropout 
#### (2014 0.001) DQN 
#### (2014 0.01) Visualizing and Understanding Conv Nets
#### (2015 0.1) VGG, Seq2Seq, GoogleNet
#### (2016 0.1) DeepSpeech2, ResNets
#### (2017 0.1) Wave2Letter
#### (2017 1.0) GNMT, Xception
#### (2017, 10.0) Neural Architecture Search
#### (2018, 1.0) WaveNet-Tacotron
#### (2019, 1.0) BERT, WaveGlow, Jasper
#### (2020, 100.0) GPT2

<p>&nbsp;</p>

#
## 래블업 Backend AI.
### ML Framework(오픈소스, 클라우드기반)

[래블업 링크](https://www.lablup.com/home)

[래블업 깃허브 링크](https://github.com/lablup/backend.ai)

<p>&nbsp;</p>

### 1. Git 다운로드

<p>&nbsp;</p>

### 2. html 기반으로 사용. docker Portainer와 유사하게 사용하면됨

<p>&nbsp;</p>

### 3. python version(3.6까지 지원?되는거같음), cuda version(10.2까지 되는거같음) 확인후 환경 세팅.

<p>&nbsp;</p>

### 4. terminal이나 Jupyter 사용 가능.

<img src="{{ site.url }}/images/Docker/2020_11_04/backendAI.png">

<p>&nbsp;</p>

