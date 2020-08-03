---
layout: post
title: "SRResNet"
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

# SR ResNet 구축

<p>&nbsp;</p>

[강의 링크](https://www.youtube.com/watch?v=drAN7gLA8sU&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys&index=11)

[GIT 링크](https://github.com/hanyoseob/youtube-cnn-003-pytorch-image-regression-framework)

[paper 링크](https://arxiv.org/abs/1609.04802)


<img src="{{ site.url }}/images/study/SRResnet/20200803_141441.png">

<p>&nbsp;</p>

## Model 추가하기
### class SRResNet(nn.Module)
~~~python

#model.py
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=16):  # Nblk(NumberOfBlock) = resblock가 몇번 반복되는지 정의하는 parameter

    super(ResNet, self).__init__()

    self.learning_type = learning_type

    self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0)  # padding은 int(kernel_size/2) = 9/2 =4.5 -> 4

~~~

<p>&nbsp;</p>

## layer 추가하기
### class ResBlock(nn.Module)
~~~python

#layer.py
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        # 1st conv
        layers += [CBR2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=relu)]

        # 2nd conv
        layers += [CBR2d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=None)] # 그림엔 Relu가 없음 None으로 함

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)  # Elementwise Sum이 적용된 output 도출

~~~

<p>&nbsp;</p>

## Model 추가하기
### class SRResNet(nn.Module)

~~~python

#model.py
        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)
        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)
~~~

<p>&nbsp;</p>

## layer 추가하기
### class PixelUnshuffle(nn.Module) 역방향
### class PixelShuffle(nn.Module) 정방향
### https://arxiv.org/pdf/1609.05158.pdf 참조해서 구현
~~~python

#layer.py
class PixelUnshuffle(nn.Module):# channel 방향으로 stack 되어있는 resolution을 
                                # hight-resolution image로 변경해줌
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx
        # Batchsize, ChannelSize, Height, Width
        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)   # B,C는 그대로 두고 downsampling한 axis
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (ry * rx), H // ry, W // rx)
        return x

class PixelShuffle(nn.Module):# High-resolution image -> low resolution image
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        # axis 수정
        x = x.permute(0, 1, 4, 2, 5, 3)
        # 원본의 형태로 다시 만들어줌
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x
~~~

<p>&nbsp;</p>

## Model 추가하기
### class SRResNet(nn.Module)

~~~python

#model.py
        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]  # 64->256 = 4* nker(64)
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.ReLU()]
        self.ps1 = nn.Sequential(*ps1)  # subfixcel이 정의되어있는 첫번째 block

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = CBR2d(nker, out_channels, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=None)  # 마지막 colvolution layer


    def forward(self, x):
        x = self.enc(x)  # 첫번째 block 수행
        x0 = x  # skip connection 으로 건너띄어진 Elementwise Sum output으로 정의

        x = self.res(x)  # B residual blcoks 수행

        x = self.dec(x)  # Conv, BN 수행
        x = x + x0

        x = self.ps1(x)  # subfixcel 1 수행
        x = self.ps2(x)  # subfixcel 1 수행

        x = self.fc(x)  # 마지막 colvolution 수행

        return x
~~~

<p>&nbsp;</p>

## Train para 추가하기
### -
~~~python

#train.py
# resnet, srresnet 추가
parser.add_argument("--network", default="srresnet", choices=["unet", "hourglass", "resnet", "srresnet"], type=str, dest="network") 

# option에 0을 추가해서 keepdim에 값을 할당하여 down dimension을 수행
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')
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

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/SRResnet/20200803_173207.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/SRResnet/20200803_173343.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/SRResnet/20200803_173527.png">

<p>&nbsp;</p>

<img src="{{ site.url }}/images/study/SRResnet/20200803_173446.png">