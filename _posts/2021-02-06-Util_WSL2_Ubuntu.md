---
layout: post
title: "WSL2 + Ubuntu 설치"
description: 
headline: 
modified: 2021-02-06
category: util
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# WSL2 + Ubuntu 설치

- Windows Subsystem for Linux2로 윈도우 10에서도 Linux 환경을 사용할 수 있게 됨.

- WSL2가 windows 와 linux 환경을 bridge 하여 가상환경을 세팅하는 것보다 속도가 매우 빠름

- linux 환경에서 Cuda GPU 를 사용할 수 있음.

<p>&nbsp;</p>

## 1. Windows Insider 설치

- 개발자 모드의 윈도우로 사용

[Window Insider 링크](https://insider.windows.com/en-us/)

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20125011.png">

- 개발자 등록 후 다운로드

[Window Insider 가이드 링크](https://insider.windows.com/en-us/getting-started)

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20125311.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20125425.png">


<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20125513.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20125810.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20125859.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20144425.png">

<p>&nbsp;</p>

- 잘 설치

## 2. WSL2

[WSL2 설치 가이드 링크](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20144706.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20144730.png">

<p>&nbsp;</p>


## 3.1 windows Ubuntu

- 설치가이드를 easy를 따라가면 자동으로 ubuntu가 설치된다.

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20145132.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20145410.png">


~~~python
# WSL에 붙어있는 SW 목록
#wsl --list --vorbose
wsl -l -v

# default를 2로 설정해준다.(build 21301 판에서는 기본 default 2)
wsl --set-default-version 2

# 혹시나 1로 설정되어있으면 2로 바꾸어주기
wsl --set-version Ubuntu 2

~~~

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20150044.png">

- ubuntu shell 을 열어서 버전 확인

~~~python
# 우분투 버전확인인
cat /etc/issue

# 또는
lsb_release -a
~~~

## 3.2 Docker Desktop for Windows

[Docker Desktop 설치 링크](https://hub.docker.com/editions/community/docker-ce-desktop-windows)

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20150440.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20150601.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20150613.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20150904.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20151136.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20151329.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20152225.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20152312.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20152335.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20152519.png">


- 잘 설치해주도록 한다

## 4. WSL 용 CUDA

[참고링크](https://dailylime.kr/2020/06/wsl2%EC%97%90%EC%84%9C-ubuntu%EC%99%80-cuda-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/)

[WSL CUDA 설치 링크](https://developer.nvidia.com/cuda/wsl/download)

- 없으면 위에 링크에서 다운받고 있으면 skip 한다.

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20153443.png">



~~~python
nvidia-smi
~~~
<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20153851.png">

- NVIDIA Container Toolkit 설치

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20154406.png">


~~~python
#Ubuntu Shell 에서 실행
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container-experimental.list | sudo tee /etc/apt/sources.list.d/libnvidia-container-experimental.list

sudo sed -i "s/archive.ubuntu.com/mirror.kakao.com/g" /etc/apt/sources.list

sudo apt update && sudo apt install -y nvidia-docker2
~~~

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20154832.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20155200.png">


## 5. Tensorflow-GPU and Jupyter

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20155752.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20155818.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20155834.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20155919.png">

~~~python

#GPU Test
sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark

# Jupyter notebook 실행행
docker run -u $(id -u):$(id -g) -it --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter
~~~


<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20160007.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20160028.png">



- 안타깝지만 필자는 cuda가 동작하는 gpu 컴을 가지고 있지않아서 gpu local에서는 gpu사용은 불가하다..ㅠㅠ
- local에서 이미지와 test code만 만들고 gpu가 구비되어있는 환경에서 training 하도록 한다.