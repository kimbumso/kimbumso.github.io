---
layout: post
title: "Ubuntu Docker basic setting "
description: 
headline: 
modified: 2020-11-30
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
## 1. Docker Compose setting

### 기본 구조
<img src="{{ site.url }}/images/Docker/2020_11_30/스크린샷, 2020-12-01 09-33-23.png">


### docker compose 배포시 기본적인 시나리오는 docker-compose.yml 이 담당하며 이 구성에 의해 docker image 가 구성됨.

~~~python 
version: '3'
services:
    nginx:
        container_name: nginx
        build: ./nginx
        image: docker-server/nginx
        restart: always
        ports:
          - "80:80"  # 80번 포트는 기본 포트의 접근. 즉 따로 포트없이 http://hostname/  식의 접근이 가능.
        volumes:
          - ./server_dev:/srv/docker-server
          - ./log:/var/log/nginx
          - ./nginx/uwsgi_params:/etc/nginx/uwsgi_params
        depends_on:
          - django  # 의존성 주입. 

    django:
        container_name: django
        build: ./server_dev
        image: docker-server/django
        restart: always
        command: uwsgi --ini uwsgi.ini
        volumes:
          - ./server_dev:/srv/docker-server
          - ./log:/var/log/uwsgi

~~~


### uwsgi_params 를 가지고 docker-compose가 uwsgi를 사용할 수 있게끔 설정

~~~python

uwsgi_param  QUERY_STRING       $query_string;
uwsgi_param  REQUEST_METHOD     $request_method;
uwsgi_param  CONTENT_TYPE       $content_type;
uwsgi_param  CONTENT_LENGTH     $content_length;

uwsgi_param  REQUEST_URI        $request_uri;
uwsgi_param  PATH_INFO          $document_uri;
uwsgi_param  DOCUMENT_ROOT      $document_root;
uwsgi_param  SERVER_PROTOCOL    $server_protocol;
uwsgi_param  REQUEST_SCHEME     $scheme;
uwsgi_param  HTTPS              $https if_not_empty;

uwsgi_param  REMOTE_ADDR        $remote_addr;
uwsgi_param  REMOTE_PORT        $remote_port;
uwsgi_param  SERVER_PORT        $server_port;
uwsgi_param  SERVER_NAME        $server_name;

~~~

### Nginx 세팅은 Dockerfile이 각종 config 파일을 제어함으로 구성됨.

~~~python
# Dockerfile
FROM nginx:latest

COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx-app.conf /etc/nginx/sites-available/


RUN mkdir -p /etc/nginx/sites-enabled/\
	&& ln -s /etc/nginx/sites-available/nginx-app.conf /etc/nginx/sites-enabled/

CMD ["nginx", "-g", "daemon off;"]

~~~

#
~~~python
# nignx.conf
user root; 
worker_processes auto; 
pid /run/nginx.pid;
 
events { 
    worker_connections 1024; 
}

http {

     sendfile on;
     tcp_nopush on;
     tcp_nodelay on;
     keepalive_timeout 65;
     types_hash_max_size 2048;

     uwsgi_read_timeout 600;
     uwsgi_send_timeout 600;
     uwsgi_connect_timeout 600; #time out을 맞춰줘야 가끔 발생하는 502 error를 회피할수 있음.

     include /etc/nginx/mime.types;
     default_type application/octet-stream;


     ssl_protocols TLSv1 TLSv1.1 TLSv1.2; 
     ssl_prefer_server_ciphers on; 


     access_log /var/log/nginx/access.log;
     error_log /var/log/nginx/error.log;

     gzip on;
     gzip_disable "msie6";


    include /etc/nginx/sites-enabled/*;
}

~~~

#

~~~python
# nginx-app.conf
upstream uwsgi{
    ip_hash;
    server unix:/srv/docker-server/django.sock; # 소켓통신할 소켓지정
}

server {
    listen 80;
    server_name localhost;
    charset utf-8;
    client_max_body_size 128M;  # 메시지 버퍼로 사이즈가 작으면 통신에 실패할 수있음.
    location /static {
		alias /srv/docker-server/.static_root;  # docker내 static폴더로 web에서 사용하는 css,js, image 등을 제어할 수있게끔 루트설정. 이부분이 안되면 local에서는 잘 동작하는데 docker 배포시 이미지나 각종 css가 깨질 수 있음.
	}
    location /media {
		alias /srv/docker-server/server_dev/media;  # mdedia 폴더를 지정하는 것으로 이건 추후에 AWS S3나 GCP bucket으로 빼줄 필요가 있을거 같음. docker에서 db까지 제어가 되는게 아니라서 생성시마다 migrate를 해줘야 하는 이슈가있음.
	}
    location / {
		uwsgi_pass  uwsgi;
		include  /etc/nginx/uwsgi_params;
	}


}

server_tokens off;
~~~
#
~~~python
# nginx-app.mediax.conf
upstream uwsgi{
    server unix:/srv/docker-server/django.sock;
}

server {
    listen 80;
    server_name localhost;
    charset utf-8;
    client_max_body_size 128M;

    location / {
		uwsgi_pass  uwsgi;
		include  uwsgi_params;
	}

}



~~~