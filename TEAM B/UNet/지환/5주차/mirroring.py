### X:AI 5주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
from paper_model_mirror import UNet


class Mirroring:
    def __init__(self, input, model):
        self.input = input
        self.model = model
        self.mirror_size = 188
        self.w_stride = 256
        self.h_stride = 128
        self.height = 512
        self.width = 512

    def extract_patches(self):
        _, _, self.height, self.width = self.input.size()
        patch_width = self.width // 2
        patch_height = self.height // 4
        
        patches = []
        
        padded_x = F.pad(self.input.float(), [self.mirror_size // 2, self.mirror_size // 2, self.mirror_size // 2, self.mirror_size // 2], mode='reflect')
        
        for i in range(0, self.width, self.w_stride):
            for j in range(0, self.height, self.h_stride):
                patch = padded_x[:, :, j:j+patch_height+self.mirror_size, i:i+patch_width+self.mirror_size]
                patches.append(patch)
        
        return torch.stack(patches, dim=0)


    # 미러링을 적용하여 입력 이미지를 패치로 추출
    def process(self):
        patches = self.extract_patches()

        outputs = []
        for patch in patches:
            output = self.model(patch)
            output = output[:, :, 2:130, 2:258] # crop
            outputs.append(output)

        # 출력 패치를 결합하여 최종 output 생성
        concat_output = torch.zeros(1, 1, self.width, self.height, dtype=patches.dtype, device=patches.device)
        idx = 0
        for i in range(0, self.width, self.w_stride):
            for j in range(0, self.height, self.h_stride):
                concat_output[:, :, j:j+self.h_stride, i:i+self.w_stride] = outputs[idx]
                idx += 1
                
        return concat_output
