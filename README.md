# newhiwoong.github.io

[![licenses](https://img.shields.io/badge/licenses-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)[![results](https://img.shields.io/badge/results-Web-blue.svg)](https://opensource.org/licenses/MIT)  

> [BLOG OF KIWOONG'S](https://newhiwoong.github.io)의 CODE





## Github 블로그 제작법 및 팁

1. [개인 블로그 1 : 시작 계기](https://newhiwoong.github.io/webdevelopment/first-post)

2. [Github로 자신만의 블로그를 만드는 방법](https://newhiwoong.github.io/webdevelopment/%EB%B8%94%EB%A1%9C%EA%B7%B8-%EC%A0%9C%EC%9E%91%EB%B2%95)

3. [Disqus로 Github 블로그 댓글 기능 추가](https://newhiwoong.github.io/webdevelopment/Disqus%EB%A1%9C-Github-%EB%B8%94%EB%A1%9C%EA%B7%B8-%EB%8C%93%EA%B8%80-%EA%B8%B0%EB%8A%A5-%EC%B6%94%EA%B0%80)

4. [Github 블로그를 인터넷에서 검색할 수 있게 만들기](https://newhiwoong.github.io/webdevelopment/Github-%EB%B8%94%EB%A1%9C%EA%B7%B8%EB%A5%BC-%EA%B2%80%EC%83%89-%EA%B0%80%EB%8A%A5%ED%95%98%EA%B2%8C-%EB%A7%8C%EB%93%A4%EA%B8%B0)

5. [Jekyll Github 블로그에 Google Analytics 적용법](https://newhiwoong.github.io/webdevelopment/%EB%B8%94%EB%A1%9C%EA%B7%B8%EC%97%90-Google-Analytics-%EC%A0%81%EC%9A%A9%EB%B2%95)

6. [Markdown Editor 추천 Mark Text](https://newhiwoong.github.io/%EA%B8%B0%ED%83%80%20%EC%A0%95%EB%B3%B4%20%EA%B3%B5%EC%9C%A0/Markdown-Editor-%EC%B6%94%EC%B2%9C-Mark-Text)

7. [수준급의 Github README.md 작성하기](https://newhiwoong.github.io/%EA%B8%B0%ED%83%80%20%EC%A0%95%EB%B3%B4%20%EA%B3%B5%EC%9C%A0/%EC%88%98%EC%A4%80%EA%B8%89%EC%9D%98-Github-README.md-%EC%9E%91%EC%84%B1%ED%95%98%EA%B8%B0)



### Post 제작 윗부분

```
---
layout: post
title: "개인 블로그1 : 시작 계기" #post의 제목
description:         
headline: 
modified: 2019-01-23           #post의 날짜
category: webdevelopment       #post의 카테고리
tags: [jekyll]                 #post의 태그
imagefeature: cover10.jpg      #post의 앞에 나오는 사진
mathjax: 
chart: 
comments: true                 #post의 댓글 사용 여부
share: true                    #post의 댓글 공유가능 여부
featured: true                 #post가 중요한 글인지 여부
---
```

---

위 방식대로 post의 시작 부분을 수정하여 사용하자.



### 블로그 올리기

```
git add *
git commit -m "yyyy-mm-dd-commit-post"
git push origin master
```

---

post를 작성하면 위와 같이 `git push` 작업을 해서 블로그에 올리자.



## Base code

> https://github.com/hmfaysal/Notepad
