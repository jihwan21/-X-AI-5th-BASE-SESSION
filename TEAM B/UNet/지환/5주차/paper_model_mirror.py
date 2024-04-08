### X:AI 5주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch
import torch.nn as nn

### Network Architecture

## contracting path(left side)
# two 3x3 conv(unpadded) -> ReLU -> 2x2 max pooling(stride 2) 

## expansive path(right side)
# 2x2 conv(up convoluton) -> concat cropped feature map from the contracting path -> two 3x3 conv -> ReLU

## final layer
# 1x1 conv

### total 23 convolutional layers



# Conv Block 정의
def conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0), 
                         nn.ReLU(inplace=True)
    )

# crop 함수 정의
def crop_conv(con_conv1, exp_conv2):
    w_diff_size = con_conv1.size()[2] - exp_conv2.size()[2] 
    h_diff_size = con_conv1.size()[3] - exp_conv2.size()[3] 
    
    w_start = w_diff_size // 2
    h_start = h_diff_size // 2

    w_end = con_conv1.size()[2] - w_start
    h_end = con_conv1.size()[3] - h_start
    
    w_sub = w_end - w_start
    h_sub = h_end - h_start

    if w_sub % 2 != 0:
        w_end -= 1
    
    if h_sub % 2 != 0:
        h_end -= 1

    crop_con_conv = con_conv1[:, :, w_start:w_end, h_start:h_end]

    return crop_con_conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path
        self.con_f11 = conv(1, 64) 
        self.con_f12 = conv(64, 64) 

        self.pool_f1 = nn.MaxPool2d(kernel_size=2) 

        self.con_f21 = conv(64, 128) 
        self.con_f22 = conv(128, 128) 

        self.pool_f2 = nn.MaxPool2d(kernel_size=2) 

        self.con_f31 = conv(128, 256) 
        self.con_f32 = conv(256, 256) 

        self.pool_f3 = nn.MaxPool2d(kernel_size=2) 

        self.con_f41 = conv(256, 512) 
        self.con_f42 = conv(512, 512) 

        self.pool_f4 = nn.MaxPool2d(kernel_size=2) 

        # center path
        self.con_f51 = conv(512, 1024) 
        self.con_f52 = conv(1024, 1024)

        # Expansive path
        self.unpool_f4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2) 
        # crop 
        self.exp_f41 = conv(1024, 512) 
        self.exp_f42 = conv(512, 512) 

        self.unpool_f3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2) 
        # crop
        self.exp_f31 = conv(512, 256) 
        self.exp_f32 = conv(256, 256) 

        self.unpool_f2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2) 
        # crop
        self.exp_f21 = conv(256, 128) 
        self.exp_f22 = conv(128, 128) 

        self.unpool_f1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2) 
        # crop
        self.exp_f11 = conv(128, 64) 
        self.exp_f12 = conv(64, 64) 

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0) # fc layer


    # layer 연결
    def forward(self, x):
        con_f11 = self.con_f11(x)
        con_f12 = self.con_f12(con_f11)
        pool_f1 = self.pool_f1(con_f12)

        con_f21 = self.con_f21(pool_f1)
        con_f22 = self.con_f22(con_f21)
        pool_f2 = self.pool_f2(con_f22)

        con_f31 = self.con_f31(pool_f2)
        con_f32 = self.con_f32(con_f31)
        pool_f3 = self.pool_f3(con_f32)

        con_f41 = self.con_f41(pool_f3)
        con_f42 = self.con_f42(con_f41)
        pool_f4 = self.pool_f4(con_f42)

        con_f51 = self.con_f51(pool_f4)
        con_f52 = self.con_f52(con_f51)

        unpool_f4 = self.unpool_f4(con_f52) # 56 x 56 x 512
        # crop
        crop_con_f42 = crop_conv(con_f42, unpool_f4) # 56 x 56 x 512
        # concat
        cat_f4 = torch.cat((crop_con_f42, unpool_f4), dim=1) # 56 x 56 x 1024
        exp_f41 = self.exp_f41(cat_f4)  # 54 x 54 x 512
        exp_f42 = self.exp_f42(exp_f41) # 52 x 52 x 512

        unpool_f3 = self.unpool_f3(exp_f42) # 104 x 104 x 256
        # crop
        crop_con_f32 = crop_conv(con_f32, unpool_f3) # 104 x 104 x 256
        # concat
        cat_f3 = torch.cat((crop_con_f32, unpool_f3), dim=1) # 104 x 104 x 512
        exp_f31 = self.exp_f31(cat_f3)  # 102 x 102 x 256
        exp_f32 = self.exp_f32(exp_f31) # 100 x 100 x 256

        unpool_f2 = self.unpool_f2(exp_f32) # 200 x 200 x 128
        # crop
        crop_con_f22 = crop_conv(con_f22, unpool_f2) # 200 x 200 x 128
        # concat
        cat_f2 = torch.cat((crop_con_f22, unpool_f2), dim=1) # 200 x 200 x 256
        exp_f21 = self.exp_f21(cat_f2)  # 198 x 198 x 128
        exp_f22 = self.exp_f22(exp_f21) # 196 x 196 x 128

        unpool_f1 = self.unpool_f1(exp_f22) # 392 x 392 x 64
        # crop
        crop_con_f12 = crop_conv(con_f12, unpool_f1) # 392 x 392 x 64
        # concat
        cat_f1 = torch.cat((crop_con_f12, unpool_f1), dim=1) # 392 x 392 x 128
        exp_f11 = self.exp_f11(cat_f1)  # 390 x 390 x 64
        exp_f12 = self.exp_f12(exp_f11) # 388 x 388 x 64


        fc = self.fc(exp_f12) # 388 x 388 x 2

        return fc