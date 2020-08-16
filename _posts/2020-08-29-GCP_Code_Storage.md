---
layout: post
title: "google Cloud Source Repositories 사용"
description: 
headline: 
modified: 2020-08-29
category: GCP
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# Cloud Source Repositories 사용

<img src="{{ site.url }}/images/practice/tf.jpg">

<p>&nbsp;</p>

<p>AWS Free Tier가 만료됨에 따라 기존 source를 GCP로 옮기기로 함..</p>

<p>Bucket도 같이 옮겼어야 했는데 DB만 옮기고 S3내 bucket은 안옮겨서 수집한 이미지 파일은 다 날라감..</p>

<p>&nbsp;</p>

<img src="{{ site.url }}/images/GCP/code_storage/20200829_210527.png">

<p>SSH KEY 관리</p>

<p>&nbsp;</p>

<img src="{{ site.url }}/images/GCP/code_storage/20200829_210819.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/GCP/code_storage/20200829_210910.png">

<p>config 관리</p>

<p>&nbsp;</p>

<img src="{{ site.url }}/images/GCP/code_storage/20200829_211013.png">

known_hosts에 ip가 있으면 push가 안되므로 해당 부분에 data가 있으면 지워주도록 함.

<p>source 업로드</p>

git init

git remote add google ssh://{config에 저장된 내 호스트}/p/{내 프로젝트 이름}/r/{스토리지 이름}

git remote -v  (remote 확인)

git commit -m "first"

git push --set-upstream google master

<p>&nbsp;</p>

<img src="{{ site.url }}/images/GCP/code_storage/20200829_211531.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/GCP/code_storage/20200829_211603.png">

<p>GCP에 APP ENGINE으로 배포를 위한 django config 정보 수정은 추후에 작성..</p>