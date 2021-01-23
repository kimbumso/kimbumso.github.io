---
layout: post
title: "Heroku django rest-API "
description: 
headline: 
modified: 2020-12-03
category: util
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# HEROKU를 통한 django APP 배포 

<p>&nbsp;</p>

## HEROKU는 무료 웹 배포 서비스
## 일정(월 1000시간, 500M의 기본 stroage를 제공) 서비스를 제공해주어 작은 프로젝트 배포에 좋음
## 어느정도 서버에 요청이 없으면 서버가 sleep 하므로 주기적으로 ping등을 날려줄 필요가 있음.
## 기본적으로 postgreSQL을 사용하며 비용을 지불하고 다른 DB를 사용할 수 있음.
#
## 1. app setting

### 기본 구조
<img src="{{ site.url }}/images/Heroku/first/스크린샷, 2020-12-03 13-19-45.png">


### django app을 구성하기위해 기본적인 가상환경에서 APP을 만들어줌

~~~bash
django-admin startproject server_dev
django-admin startapp quiz

# app폴더에서 requirements.txt를 읽어서 가상환경에 설치해줌
pip install -r requirements.txt

certifi==2020.6.20
chardet==3.0.4
configparser==5.0.1
dj-database-url==0.5.0
Django==2.2.6
django-ckeditor==5.8.0
django-cors-headers==3.5.0
django-heroku==0.3.1
django-js-asset==1.2.2
django-storages==1.7.2
djangorestframework==3.12.2
docutils==0.15.2
gunicorn==20.0.2
idna==2.10
jmespath==0.9.4
mysqlclient==1.4.4
Pillow==8.0.0
psycopg2==2.8.6
psycopg2-binary==2.8.6
python-dateutil==2.8.0
pytz==2019.3
requests==2.24.0
s3transfer==0.2.1
six==1.12.0
sqlparse==0.3.0
urllib3==1.25.6
whitenoise==5.2.0

## psycop2 를 설치할 때 이슈가 있는데 이는 local에 postgreSQL이 깔려있지 않으면 에러가 발생할 수있음. postgreSQL을 설치하면서 필요한 파일들을 import 해줌

~~~


### 배포를 위한 환경설정 파일을 설정 Procfile과 runtime.txt가 필요함

~~~python
# Procfile
web: gunicorn server_dev.wsgi --log-file -

# runtime.txt
python-3.8.5 # 자신에게 맞는 버전 설치

~~~

### Main Setting 설정 server_dev/setting.py

~~~python
# 장고 app key를 다음과 같은 형태로 변경
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'js4ha*cg35%@c83v^m7lfy8k586wp#-@s2tdpoh7(&)1kif(@&')

# 디버그 설정
DEBUG = bool( os.environ.get('DJANGO_DEBUG', True))

# 접근 호스트 열어주기
ALLOWED_HOSTS = ['*']

# 앱에 ckeditor와 rest_framework, 앞으로 추가하게될 앱들을 등록
INSTALLED_APPS = [
    'ckeditor',
    'ckeditor_uploader',
    'rest_framework',
    'quiz',
]

# 미들웨어에 whitenoise 추가. static 파일을 관리해주는 프로그램
MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware',
]

# static 파일 관리
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = '/static/'

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/'

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# ck uploader
CKEDITOR_UPLOAD_PATH = 'uploads/'  # create directory in media directory
CKEDITOR_RESTRICT_BY_USER = True

CKEDITOR_CONFIGS = {
    "default": {
        "removePlugins": "stylesheetparser",
    }
}

# Extra places for collectstatic to find static files.
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, 'static'),
    os.path.join(BASE_DIR, 'media')
)

# db 관리
import dj_database_url
db_from_env = dj_database_url.config(conn_max_age=500)
DATABASES['default'].update(db_from_env)

~~~

# git에 업로드하지 않을 파일을 .gitignore에 설정
~~~python
# Django
*.log
*.pot
*.pyc
__pycashe__/
*.py[cod]

# Distribution / pakaging
.Python
venv/
.env

# Editor Directories and files
.vscode
.idea
*.sue
*.ntvs*
*.njsproj
*.sln

# DB
db.sqlite3
.DS_Store
~~~

# 배포
~~~bash
git init
git add .
git commit -m '메시지를 넣도록함.'
git push origin master 

# heroku 로그인
heroku login
# 로그인 url에 가서 로그인을 실행한후 다시 터미널로 돌아온 후 프로젝트 생성
heroku create '앱이름'

heroku git:remote -a '앱이름'

git push heroku master

# 중간에 collectstatic으로 오류가 발생하면
heroku config:set DISABLE_COLLECTSTATIC=1
# 이후 다시 배포

# 배포가 성공적이면 migrate를 해줘야함
heroku run python manage.py migrate

# 헤로쿠를 열어서 잘 배포되었는지 확인
heroku open

~~~
#
<img src="{{ site.url }}/images/Heroku/first/스크린샷, 2020-12-03 13-44-50.png">

<img src="{{ site.url }}/images/Heroku/first/스크린샷, 2020-12-03 13-44-04.png">

# 
# 다음에 할일 . 공공데이터 API를 장고 app에다 붙여서 시각화... 및 flutter 앱으로 배포
~~~python

~~~