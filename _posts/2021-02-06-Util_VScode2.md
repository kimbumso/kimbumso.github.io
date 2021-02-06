---
layout: post
title: "code-server 설치2"
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

# VSCODE Server 사용 

<p>&nbsp;</p>

- 휴대폰등 다양한 기기에서 Vscode를 사용
- vscode가 Electron Framework 기반이기 때문에 가능
- We have a script to install code-server for Linux, macOS and FreeBSD. 해당 os에서만 사용가능

## 그러나 docker를 쓰면 단 2줄의 코드로 사용할 수있다

~~~python
mkdir -p ~/.config

docker run -it --name code-server -p 127.0.0.1:8080:8080  -v "$HOME/.config:/home/coder/.config"  -v "$PWD:/home/coder/project"  -u "$(id -u):$(id -g)"   -e "DOCKER_USER=$USER"   codercom/code-server:latest
~~~

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20224214.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20224252.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20224338.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20224406.png">

<p>&nbsp;</p>
