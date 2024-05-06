### X:AI 6주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector


class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask):
        # 정규화
        normalized_source = self.layer_norm(source)
        # 잔차 연결 + MultiHead Attention
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]

        # output 정규화
        normalized_output = self.layer_norm(output)
        # output 잔차 연결 + PostionWise FFN
        output = output + self.position_wise_ffn(normalized_output)

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        # 입력 token 임베딩 / 고정 크기 벡터로 변환
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)
        # 임베딩 가중치 초기화
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        # 스케일링(임베딩 벡터 크기 조절)
        self.embedding_scale = params.hidden_dim ** 0.5
        # positional 임베딩, 입력 token 위치 정보 / 학습 x
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        # 인코더 레이어 모음
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, source):
        source_mask = create_source_mask(source) # padding token 'True' 처리한 마스크 생성    
        source_pos = create_position_vector(source) # position vector 생성

        source = self.token_embedding(source) * self.embedding_scale # 각 토큰 임베딩 + 스케일 조절
        source = self.dropout(source + self.pos_embedding(source_pos)) # source 임베딩 벡터에 위치 정보 임베딩 벡터 추가

        # 인코더 레이어 반복 수행
        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)

        return self.layer_norm(source)