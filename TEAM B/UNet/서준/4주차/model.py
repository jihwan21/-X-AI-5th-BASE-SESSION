import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 네트워크에서 반복적으로 사용되는 Conv + BatchNorm + Relu를 합쳐서 하나의 함수로 정의
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            
            # 배치 정규화
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            # activation function = ReLU()
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers) # *을 붙이는 이유 : *을 사용하여 list를 unpacking 하려고

            return cbr
        
        # Encoder 구성
        self.encoder1_1 = CBR2d(in_channels=1, out_channels=64)
        self.encoder1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2_1 = CBR2d(in_channels=64, out_channels=128)
        self.encoder2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3_1 = CBR2d(in_channels=128, out_channels=256)
        self.encoder3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder4_1 = CBR2d(in_channels=256, out_channels=512)
        self.encoder4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.encoder5_1 = CBR2d(in_channels=512, out_channels=1024)
        ## Encoder 완성

        # Decoder 구성
        # Expansive path
        self.decoder5_1 = CBR2d(in_channels=1024, out_channels=512)

        # ConvTranspose2d : stride = 2일 때 영상의 크기를 2배로 크게하는 역할 수행
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # in_channels = 512 * 2인 이유 : encoder에서 사용햇던 feature map -> decoder에서도 사용할 수 있도록
        # forward에서 dec와 enc을 concat해주기 때문
        self.decoder4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.decoder4_1 = CBR2d(in_channels=512, out_channels=256)

        # ConvTranspose2d : stride = 2일 때 영상의 크기를 2배로 크게하는 역할 수행
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.decoder3_1 = CBR2d(in_channels=256, out_channels=128)

        # ConvTranspose2d : stride = 2일 때 영상의 크기를 2배로 크게하는 역할 수행
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.decoder2_1 = CBR2d(in_channels=128, out_channels=64)

        # ConvTranspose2d : stride = 2일 때 영상의 크기를 2배로 크게하는 역할 수행
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.decoder1_1 = CBR2d(in_channels=64, out_channels=64)

        # ConvTranspose2d : stride = 2일 때 영상의 크기를 2배로 크게하는 역할 수행
        # segmentation에 필요한 n개의 클래스에 대한 output 정의
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Encoder
        encoder1_1 = self.encoder1_1(x)
        encoder1_2 = self.encoder1_2(encoder1_1)
        pool1 = self.pool1(encoder1_2)

        encoder2_1 = self.encoder2_1(pool1)
        encoder2_2 = self.encoder2_2(encoder2_1)
        pool2 = self.pool2(encoder2_2)

        encoder3_1 = self.encoder3_1(pool2)
        encoder3_2 = self.encoder3_2(encoder3_1)
        pool3 = self.pool3(encoder3_2)

        encoder4_1 = self.encoder4_1(pool3)
        encoder4_2 = self.encoder4_2(encoder4_1)
        pool4 = self.pool4(encoder4_2)

        encoder5_1 = self.encoder5_1(pool4)

        # Decoder
        decoder5_1 = self.decoder5_1(encoder5_1)

        unpool4 = self.unpool4(decoder5_1)

        # encoder에서 사용했던 feature map을 decoder에서도 동일하게 사용하도록 구축
        cat4 = torch.cat((unpool4, encoder4_2), dim=1)
        decoder4_2 = self.decoder4_2(cat4)
        decoder4_1 = self.decoder4_1(decoder4_2)

        unpool3 = self.unpool3(decoder4_1)
        cat3 = torch.cat((unpool3, encoder3_2), dim=1)
        decoder3_2 = self.decoder3_2(cat3)
        decoder3_1 = self.decoder3_1(decoder3_2)

        unpool2 = self.unpool2(decoder3_1)
        cat2 = torch.cat((unpool2, encoder2_2), dim=1)
        decoder2_2 = self.decoder2_2(cat2)
        decoder2_1 = self.decoder2_1(decoder2_2)

        unpool1 = self.unpool1(decoder2_1)
        cat1 = torch.cat((unpool1, encoder1_2), dim=1)
        decoder1_2 = self.decoder1_2(cat1)
        decoder1_1 = self.decoder1_1(decoder1_2)

        x = self.fc(decoder1_1)

        return x