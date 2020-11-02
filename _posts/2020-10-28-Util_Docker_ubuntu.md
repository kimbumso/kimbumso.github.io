---
layout: post
title: "Ubuntu Docker basic setting "
description: 
headline: 
modified: 2020-10-28
category: util
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# Docker Ubuntu 사용 

<p>&nbsp;</p>

## windows 에서 docker로 Tensorflow 사용시 GPU 환경 사용이 아직 안됨.
## 추가로 windows에서 사용할 수 있도록 Oracle VritualBox를 설치해서 사용해야함.
## ubuntu 20.04로 update 하면서 환경 재 설정
#
### 1. vim editor 설치

~~~bash
$ sudo apt-get update

$ sudo apt-get install vim

$ vi ~/.vimrc

set number    " line 표시
set ai    " auto indent
set si " smart indent
set cindent    " c style indent
set shiftwidth=4    " 자동 공백 채움 시 4칸
set tabstop=4    " tab을 4칸 공백으로
set ignorecase    " 검색 시 대소문자 무시
set hlsearch    " 검색 시 하이라이트
set nocompatible    " 방향키로 이동 가능
set fileencodings=utf-8,euc-kr    " 파일 저장 인코딩 : utf-8, euc-kr
set fencs=ucs-bom,utf-8,euc-kr    " 한글 파일은 euc-kr, 유니코드는 유니코드
set bs=indent,eol,start    " backspace 사용가능
set ruler    " 상태 표시줄에 커서 위치 표시
set title    " 제목 표시
set showmatch    " 다른 코딩 프로그램처럼 매칭되는 괄호 보여줌
set wmnu    " tab 을 눌렀을 때 자동완성 가능한 목록
syntax on    " 문법 하이라이트 on
filetype indent on    " 파일 종류에 따른 구문 강조
set mouse=a    " 커서 이동을 마우스로 가능하도록
~~~
#

### 2. ubuntu 한글 키보드 설치

~~~bash
$ sudo apt-get update

$ sudo apt-get install fcitx-hangul

~~~
 이후 설정에서 한국어(101/104키 호환 key 및 fcitx 입력기를 설정해준 후 재부팅)

#
### 3. default python 설정(3.8.5)
~~~bash
$ ls /usr/bin/ | grep python

dh_python2
python
python-config
python2
python2-config
python2.7
python2.7-config
python3
python3-config
python3-futurize
python3-pasteurize
python3.8            # 요놈 사용 
python3.8-config
x86_64-linux-gnu-python2-config
x86_64-linux-gnu-python2.7-config
x86_64-linux-gnu-python3-config
x86_64-linux-gnu-python3.8-config
~~~

~~~bash
$ sudo update-alternatives --config python

$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 2
~~~

~~~bash
$ sudo update-alternatives --config python

# config 설정에서 3.8 인 1을 선택해준다. 
$ python --version
Python 3.8.5

# 주의점. ubuntu에는 python에 의존하는 pakage가 매우 많기 때문에 
# python을 지우거나하면 의존성이 깨져서 os를 밀고 다시 깔아야 하는 경우가 많이 생김.
# 최대한 default python은 건들지 않고 venv를 이용
~~~
#
### 4. docker 설치 

~~~bash
$ apt update & apt upgrade

# 필수 패키지 설치
$ sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# key 인증
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# docker repository 등록 
$ sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"

# 도커 설치 
$ sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io

# 도커 설치확인 
$ docker --version

# 부팅시 자동 도커 실행
$ sudo systemctl enable docker && service docker start

~~~
# 5. docker portainer 세팅
~~~bash
# portainer 폴더생성 
$ mkdir -p /data/portainer

# 도커 portainer 실행
$ docker run --name portainer -p 9000:9000 -d --restart always -v /data/portainer:/data -v /var/run/docker.sock:/var/run/docker.sock portainer/portainer

~~~

<img src="{{ site.url }}/images/Docker/2020_10_28/docker01.png">

<p>&nbsp;</p>

#
### 6. docker compose 설치

~~~bash
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

$ sudo chmod +x /usr/local/bin/docker-compose

$ sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

$ docker-compose -version

~~~
#
### 7. Pycharm, Vscode, MysqlWorkbench(community 버전) 설치.
ubuntu software에서 설치

#
### 8. PlayOnLunx 설치 (wine)
ubuntu software에서 설치
5.0 이상부터는 windows 7 이 default이나 해당 ver 쓰면 최근 windows exe 파일
쓸수 있는게 별로 없으므로 windows 10으로 설치. 그러나 해당버전도 현재 버그가 너무많다..

설치해서 kakaotalk , kiwoom 영웅문 설치
추후 api 연동 진행
#
