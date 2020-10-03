---
layout: page
permalink: /about/index.html
title: 데이터 엔지니어
tags: [Bumso, kim, rlaqjath]
imagefeature: fourseasons.jpg
chart: true
---
<table align='center'>
    <tr>
        <td>
            <figure>
	            <img src="{{ site.url }}/images/gguji.jpeg" alt="농부">
                <figcaption>My Futures</figcaption>
            </figure>
        </td>
        <td>
            <figure>
	            <img src="{{ site.url }}/images/developer.png" alt="개발자">
                <figcaption>My Futures</figcaption>
            </figure>
        </td>
        <td>
            <figure>
	            <img src="{{ site.url }}/images/invester.jpg" alt="투자자">
                <figcaption>My Futures</figcaption>
            </figure>
        </td>
    </tr>
</table>

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

저의 목표는 인공지능에 의한 자동화된 세상을 만드는 것입니다. 여러 산업군에서 반복적이고 힘들고 위험한 곳에서 일하는 분들이 많이 있고 사고도 많이 발생하고 있습니다. 이런 부문에서 더 이상 사람이 리스크에 희생되는 것이 아닌 새로운 사회를 만들어 가는 것을 원합니다. 인지 기술의 발달은 판단에 있어 더 합리적인 근거를 제시하고 있습니다. 자동화 기술의 발달은 지루하고 단순한 작업의 근로환경의 변화를 암시하고 있습니다. 예측 기술의 발달은 미래의 불확실성을 감소시켜 안정성을 높여주고있습니다. 이러한 AI 기술은 지속적으로 발전하고 개선이 이루어지고 있으며 이런 개선작업에 참여하여 더 좋은 서비스를 제공하는 개발자로 거듭나고 싶습니다.

## [Skills]()

### Language
Python, Java, Mysql, Javascript

### Framework
Pytorch, Django, Hadoop, AWS, GCP, Hadoop, scikit-learn, [TensorFlow](https://github.com/kimbumso)

<h2>Connect</h2>
✉️ [rlaqjath1@gmail.com]()  
🌐 [https://github.com/kimbumso](https://github.com/kimbumso)
