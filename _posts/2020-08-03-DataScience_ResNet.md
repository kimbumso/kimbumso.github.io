---
layout: post
title: "ResNet"
description: 
headline: 
modified: 2020-08-03
category: DataScience
tags: [jekyll]
imagefeature: 
mathjax: 
chart: 
comments: true
featured: true
---

# ResNet 구축

<p>&nbsp;</p>

[한요섭박사님 강의 링크](https://www.youtube.com/watch?v=eSYoOwk31mM&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys&index=12)

[GIT 링크](https://github.com/hanyoseob/youtube-cnn-003-pytorch-image-regression-framework)

[paper 링크](https://arxiv.org/abs/1609.04802)


<img src="{{ site.url }}/images/study/Resnet/20200803_153106.png">

<p>&nbsp;</p>

## Model 추가하기
### class ResNet(nn.Module)
~~~python

#model.py
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=16):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)

        self.fc = CBR2d(nker, out_channels, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)  # Single Conv Layer .Unet과 동일하게 kernelsize =1

    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x

~~~

<p>&nbsp;</p>

## Train para 추가하기
### -
~~~python

#train.py
# resnet, srresnet 추가
parser.add_argument("--network", default="resnet", choices=["unet", "hourglass", "resnet", "srresnet"], type=str, dest="network") 

~~~

<p>&nbsp;</p>

## network 추가하기
### -
~~~python

#train.py
# resnet, srresnet 추가
elif network == "resnet":
    net = ResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "srresnet":
    net = SRResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)

~~~


<img src="{{ site.url }}/images/study/SRResnet/Screenshot_2020-08-03-12-36-57.png">