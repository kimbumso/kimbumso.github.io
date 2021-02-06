---
layout: post
title: "Ubuntu GUI"
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

# WSL2 Ubuntu GUI

- 가끔 linux 기반에서만 돌아가는 프로그램들이 간혹 있다. 

- 이 경우 terminal 환경에서 사용해도 되나 GUI로 직관적으로 볼 수도 있다.

https://sourceforge.net/projects/vcxsrv/

<p>&nbsp;</p>

## 1. xfce4 설치

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20210048.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20210446.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20210505.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20210547.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20210601.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20210616.png">

## 2. Ubuntu Setting 

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20213128.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20213442.png">

<p>&nbsp;</p>


<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20214835.png">


~~~python

sudo apt update && sudo apt -y upgrade

sudo apt install build-essential

sudo apt install net-tools

sudo apt install xrdp -y && sudo systemctl enable xrdp

#  xfce4 by installing the xubuntu-desktop and then some gtk2 libraries
sudo apt install -y tasksel

sudo tasksel install xubuntu-desktop

sudo apt install gtk2-engines

~~~

- 만약 위대로 했는데 창이 안열린다면 방화벽의 가능성이 높다. 방화벽을 해제해주도록 하자

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/Inked%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20214206_LI.jpg">


<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20214753.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/WSL2/20210206/%ED%99%94%EB%A9%B4%20%EC%BA%A1%EC%B2%98%202021-02-06%20214859.png">

<p>&nbsp;</p>
