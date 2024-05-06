### X:AI 6주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1) # feed_forward_dim 차원으로 변경 | 확장
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1) # hidden_dim 차원으로 변경 | 압축, 복원
        init_weight(self.conv1)
        init_weight(self.conv2)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1) # Conv1d에 맞게 x 차원 순서 변경                        
        output = self.dropout(F.relu(self.conv1(x))) # relu 활성화 함수 적용 후 dropout  
        output = self.conv2(output)                   

        # 원채 차원 순서로 복원
        output = output.permute(0, 2, 1)          
        return self.dropout(output)