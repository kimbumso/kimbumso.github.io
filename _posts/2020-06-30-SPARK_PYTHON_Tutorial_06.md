---
layout: post
title: "SPARK PYTHON Tutorial6"
description: 
headline: 
modified: 2020-06-30
category: SPARK
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# SPARK

[yes24 도서 링크](http://www.yes24.com/Product/Goods/77356185?scode=032&OzSrank=12)

[paper 링크](-)

[GIT 소스 링크](https://github.com/sparktraining/spark_using_python)

[acon 링크](http://www.acornpub.co.kr/book/data-spark-python#toc)



## SPARK Tutorial
1. 빅데이터, 하둡 및 스파크 소개
2. 스파크 배포
3. 스파크 클러스터 아키텍처의 이해
4. 스파크 프로그래밍 기초 학습
5. 스파크 코어 API를 사용한 고급 프로그래밍
6. 스파크로 SQL 및 NoSQL 프로그래밍하기
7. 스파크를 사용한 스트림 처리 및 메시징
8. 스파크를 사용한 데이터 과학 및 머신러닝 소개


<img src="{{ site.url }}/images/Spark/spark-logo-trademark.png">


*이탤릭* **볼드** ***이탤릭볼드***

## ***Overview***

## HDFS
* 대용량 데이터을 분산시키고 저장하고 관리하는 하둡 분산 파일 시스템(HDFS)
1. 다수의 리눅스 서버에 설치되어 운영
2. 저장하고자 하는 파일을 블록 단위로 나누어 분산된 서버에 저장
3. 네임노드(NameNode), 다수의 데이터노드(DataNode)로 구성

## YARN
프로세싱 또는 리소스 스케줄링 서브시스템
* 하둡의 데이터 처리를 제어하고 조율
1. 클라이언트가 리소스 매니저에게 응용프로그램 제출
2. 리소스 매니저는 충분한 용량을 가진 노드 매니저에 애플리케이션 마스터 프로세스 할당
3. 애플리케이션 마스터는 노드 매니저에서 실행할 리소스 매니저와 작업 컨테이너를 협상, 응용 프로그램의 작업 컨테이너를 호스팅하는 노드 매니저로 프로세스 전달
4. 노드 매니저는 작업 시도 상태와 진행상황을 애플리케이션 마스터에 보고
5. 애플리케이션 마스터는 진행률과 응용 프로그램의 상태를 리소스 매니저에 보고
6. 리소스 메니저는 응용프로그램 진행률, 상태 및 결과를 클라이언트에 보고


# 스파크 코어 API를 사용한 고급 프로그래밍

## 브로드캐스트 변수
공유변수로 읽기 전용 변수로 스파크 클러스터에서 작업자 노드가 사용할 수 있음.
이 변수는 드라이버에 의해 설정된 후에만 읽을 수 있음.
1. 브로드캐스트 변수를 사용하면 셔플 연산 팔요 x
2. 요율적이고 확장 가능한 피어 투 피어 배포 매커니즘 사용
3. 작업당 한 번씩 복제하는 대신 작업자당 한 번씩 데이터를 복제.(스파크 응용 프로그램에 수천개의 작업이 잇을 수 있으므로 이것은 매우 중요)
4. 많은 작업을 여러 번 다시 할 수 있음
5. 직렬화된 객체로 효율적으로 읽힘

## 어크뮬레이터
공유변수로 브로드캐스트와 달리 업데이트 가능. 
처리된 레코드 수를 계산하거나 조작된 레코드 수를 추적하는 등의 용도로 사용
다른 유형의 레코드를 의도적으로 계산할 때도 사용.(ex 로그 이벤트의 매핑)


## 파티셔닝
RDD 변환에서 파티션을 구성할 수 있으며 효율적인 파티션 제어는 효율 좋은 퍼포먼스를 냄

## RDD 리니지
RDD 또는 그 파티션을 생성하기 위해 수행된 일련의 변환
오류 발생 시 모든 단계의 모든 RDD를 재평가할 수 있으므로 복원력을 제공

## RDD 캐싱
모든 상위 RDD를 포함하는 스파크 RDD는 일반적으로 동일한 세션 또는 응용 프로그램에서 호출된 각 액션에 대해 다시 계산. RDD를 캐싱하면 데이터가 메모리에 지속되는데 동일한 루틴은 후속 액션이 호출될 때 재평가 없이 여러 번 재사용 가능.

## RDD 유지 
캐시된 파티션은 스파크 작업자 노드의 실행자 JVM에 있는 메모리에 저장.



<img src="{{ site.url }}/images/Spark/fourth/20200630_164357.png">