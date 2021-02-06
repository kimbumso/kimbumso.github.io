---
layout: post
title: "docker mount"
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

# Docker Mount

- Docker를 사용하다보면 이미지를 삭제할때 DATA도 같이 삭제가됨. 

- 그래서 외부로 Data path를 빼서 저장해줘야함

- 그러나 성능상의 이유로 mnt/c 의 windows 경로는 쓰지말라고 하는데 현재는 퍼포먼스가 필요하지 않으니 그냥 사용한다.

https://docs.docker.com/docker-for-windows/wsl/#best-practices

<p>&nbsp;</p>

~~~python

docker run -u $(id -u):$(id -g) -it --rm -v /mnt/c/Bskim_Project:/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter

~~~

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20172549.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20182712.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20182734.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20182820.png">
