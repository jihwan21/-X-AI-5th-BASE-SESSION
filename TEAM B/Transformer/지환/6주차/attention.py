### X:AI 6주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        # hidden 차원(입력 벡터 차원)이 head 수로 나누어 떨어지는지 확인(나누어 떨어지지 않으면 입력 벡터를 여러개의 헤드로 분할할 수 없기 때문)
        assert params.hidden_dim % params.n_head == 0 
        # 여러개의 self attention layer 생성하여 저장
        self.attentions = nn.ModuleList([SelfAttention(params)
                                         for _ in range(params.n_head)])
        # 출력 선형 레이어
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value, mask=None):
        # 각 attention 결과 모음
        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]

        weighted_vs = [weighted_v[0] for weighted_v in self_attentions] # 각 self.dropout(weighted_v) 모음
        attentions = [weighted_v[1] for weighted_v in self_attentions] # 각 attention_score 모음

        # weighted_vs tensor를 하나의 tensor로 concat
        weighted_v = torch.cat(weighted_vs, dim=-1)

        # 최종 output
        output = self.dropout(self.o_w(weighted_v)) 

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim # params hidden 차원 
        self.attention_dim = params.hidden_dim // params.n_head # attention 차원

        # q, k, v     
        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)
        # attention scaling 계수
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None):
        # q, k, v 행렬 생성
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)

        # attention score 계산 (공식에서 분자 먼저 계산한 것)
        self_attention = torch.bmm(q, k.permute(0, 2, 1)) # k.permute(0, 2, 1) = transpose한 것
        self_attention = self_attention / self.scale_factor 

        # masking
        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        # 정규화
        attention_score = F.softmax(self_attention, dim=-1)
        norm_attention_score = self.dropout(attention_score)

        # 정규화한 attention score(가중치)와 Value matrix 계산
        weighted_v = torch.bmm(norm_attention_score, v)

        return self.dropout(weighted_v), attention_score