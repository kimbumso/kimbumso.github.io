---
layout: post
title: "Flutter Settings basic"
description: 
headline: 
modified: 2021-01-24
category: App
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---


# Flutter

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/flutterApp/20210123/20210123_102817.png">

<p>&nbsp;</p>

## What is Flutter?
### Flutter - 가장 빠른 아름다운 네이티브 앱 이라고 공식 홈페이지에서 설명..

제목|설명
---|---
<span style="color:red; font-size:1em;">빠른 개발</span>|Stateful Hot Reload
<span style="color:blue; font-size:1em;">표현력 있고 유연한 UI</span>|계층형 아키텍처를 통해 완벽한 커스터마이징
<span style="color:green; font-size:1em;">네이티브 수준의 성능</span>|Flutter 위젯은 플랫폼별 차이를 통합하여 모두에서 네이티브 수준의 성능으로 제공

<p>&nbsp;</p>

[Flutter Structure 참고](https://github.com/tadaspetra/flutter_starter_templates)

[dart cp949 (네이버 크롤링 decoding)](https://github.com/jjangga0214/dart-cp949)

[Web scraper ( 플러터 크롤링)](https://github.com/tusharojha/web_scraper)

[플러터 UI Template](https://github.com/mitesh77/Best-Flutter-UI-Templates)

[참고 강의 링크 (Provider)](https://www.youtube.com/watch?v=c3WIBiEHVas&list=PLGJ958IePUyD8ODM2vlQlbmLGCp-s9l-n&index=8)

[참고 강의 링크 3 (Heavy Fran 유튜브 채널 : framework)](https://www.youtube.com/channel/UCqxo_5t5-_Uhq9TfhTAat0A)

<p>&nbsp;</p>

## Flutter 기본구조

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/flutterApp/20210124/20210124_180150.png">

<p>&nbsp;</p>

###  기본설치시 android, ios, build, lib, test 기본 폴더 제공 기본소스는 lib폴더에서 작성

<span style="color:red; font-size:1em;">
    1. 이미지 및 미디어 관련 파일을 넣기위해 assets 폴더 추가 <br>
    2. 각종 외부 라이브러리는 pubspec.yaml 에서 import. 어떤 라이브러리가 있는지는 pub.dev site에서 확인<br>
    3. 앱 test 시 lib에 있는 main.dart 가 실행.<br>
    4. code작성 template로 위에 flutter structure 대로 작성<br>
    5. provider 사용하여 변수관리<br>
    6. lib 는 config, domain(모델관리), provider, screen(view 관리)로 구성<br>
    7. image 사용을 위해선  pubspec.yaml 에서 assets: 에서 경로를 추가해야함<br>
    8. flutter는 android 27 이상부터 작동하므로 앱 adk 를 올릴 때 해당 버전 이상부터 하도록 설정(android studio)<br>
    9. firebase 설정(각종 로그인 정보를 저장하기 위해)<br>
    10. view는 Screen 과 Widget으로 나누어 작성. Screen에 위젯을 불러와 그리는 구조<br>
    11. Widget은 StatefulWidget과 StatelessWidget이 있는데 상황에 맞게 적절하게 섞어서 써야함.<br>
</span>

###  Provider 사용 요령 
<span style="color:green; font-size:1em;">
    1. 모델 클래스 생성<br>
    2. 프로바이더 생성 및 로직 작성<br>
    3. main 에 MultiProvider에 작성 프로바이더 등록<br>
    4. Screen 에서 사용<br>
</span>

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/flutterApp/20210124/20210124_181820.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/flutterApp/20210124/20210124_181921.png">

<p>&nbsp;</p>

<img src="https://storage.googleapis.com/bskim_bucket/gitBlog/flutterApp/20210124/20210124_181955.png">
