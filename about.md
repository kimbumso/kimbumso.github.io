---
layout: page
permalink: /about/index.html
title: 데이터 엔지니어
tags: [Bumso, kim, rlaqjath]
imagefeature: fourseasons.jpg
chart: true
---
<div>
    <figure>
	    <img src="{{ site.url }}/images/gguji.jpeg" alt="농부">
	    <img src="{{ site.url }}/images/developer.png" alt="개발자">
        <img src="{{ site.url }}/images/invester.jpg" alt="투자자">
	    <figcaption>My Futures</figcaption>
    </figure>
</div>
{% assign total_words = 0 %}
{% assign total_readtime = 0 %}
{% assign featuredcount = 0 %}
{% assign statuscount = 0 %}

{% for post in site.posts %}
    {% assign post_words = post.content | strip_html | number_of_words %}
    {% assign readtime = post_words | append: '.0' | divided_by:200 %}
    {% assign total_words = total_words | plus: post_words %}
    {% assign total_readtime = total_readtime | plus: readtime %}
    {% if post.featured %}
    {% assign featuredcount = featuredcount | plus: 1 %}
    {% endif %}
{% endfor %}

<!--
This is my personal blog. It currently has {{ site.posts | size }} posts in {{ site.categories | size }} categories which combinedly have {{ total_words }} words, which will take an average reader ({{ site.wpm }} WPM) approximately <span class="time">{{ total_readtime }}</span> minutes to read. {% if featuredcount != 0 %}There are <a href="{{ site.url }}/featured">{{ featuredcount }} featured posts</a>, you should definitely check those out.{% endif %} The most recent post is {% for post in site.posts limit:1 %}{% if post.description %}<a href="{{ site.url }}{{ post.url }}" title="{{ post.description }}">"{{ post.title }}"</a>{% else %}<a href="{{ site.url }}{{ post.url }}" title="{{ post.description }}" title="Read more about {{ post.title }}">"{{ post.title }}"</a>{% endif %}{% endfor %} which was published on {% for post in site.posts limit:1 %}{% assign modifiedtime = post.modified | date: "%Y%m%d" %}{% assign posttime = post.date | date: "%Y%m%d" %}<time datetime="{{ post.date | date_to_xmlschema }}" class="post-time">{{ post.date | date: "%d %b %Y" }}</time>{% if post.modified %}{% if modifiedtime != posttime %} and last modified on <time datetime="{{ post.modified | date: "%Y-%m-%d" }}" itemprop="dateModified">{{ post.modified | date: "%d %b %Y" }}</time>{% endif %}{% endif %}{% endfor %}. The last commit was on {{ site.time | date: "%A, %d %b %Y" }} at {{ site.time | date: "%I:%M %p" }} [UTC](http://en.wikipedia.org/wiki/Coordinated_Universal_Time "Temps Universel Coordonné").
-->

<h1 align="center">
<a href="http://dbdjango-dev.ap-northeast-2.elasticbeanstalk.com/"> bumso_Resume </a>  
</h1>


## [INTRODUCTION]()

저의 목표는 모든 개인에게 맞춘 세상을 만드는 것입니다. 예를 들어 사람마다 소설이나 시 등의 글이나 여러 음악에는 어느 정도 양의 한계가 존재합니다. 자신이 좋아하는 작가의 글을 다 읽고 다음 글이 나올 때까지 몇 년 이상이 걸리는 경우도 있습니다. 우리는 같은 것을 계속해서 보는 것이 아닌 새로운 것을 보길 원합니다. 음악도 같습니다. 이는 작업 시간의 한계로 어쩔 수 없습니다. 이를 해결하기 위해 기존에 있는 글이나 음악을 DNN으로 사용자의 취향에 맞는 새로운 글이나 음악을 자체 생성을 해주며 더욱 개인화된 경험을 제공해주는 등 획일화된 서비스가 아닌 각각의 개인에게 맞춘 서비스를 제공하고 싶습니다.

## [Skills]()

### Language
Python, Java

### Framework
Pandas, NumPy, scikit-learn, KoNLPy, [TensorFlow](https://github.com/kimbumso)

<h2>Connect</h2>
✉️ [rlaqjath1@gmail.com]()  
🌐 [https://github.com/kimbumso](https://github.com/kimbumso)
