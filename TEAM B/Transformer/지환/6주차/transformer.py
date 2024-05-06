### X:AI 6주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        encoder_output = self.encoder(source)                            
        output, attn_map = self.decoder(target, source, encoder_output) 
        return output, attn_map

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)