### X:AI 6주차 Code 과제
### AI빅데이터융합경영 배지환 

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector


class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.encoder_attention = MultiHeadAttention(params) # 인코더-디코더 어텐션
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # 정규화
        norm_target = self.layer_norm(target)
        # 잔차 연결 + MultiHead Attention
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]

        # 정규화
        norm_output = self.layer_norm(output)
        # 인코더 K, V / 디코더 Q MultiHead Attention
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask)
        # 잔차 연결
        output = output + sub_layer

        # 정규화
        norm_output = self.layer_norm(output)
        # output 잔차 연결 + PostionWise FFN
        output = output + self.position_wise_ffn(norm_output)

        return output, attn_map


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        # 출력 token 임베딩
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)
        # 임베딩 가중치 초기화
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        # 스케일링
        self.embedding_scale = params.hidden_dim ** 0.5
        # positional 임베딩
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)

        # 디코더 레이어 모음
        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self, target, source, encoder_output):
        target_mask, dec_enc_mask = create_target_mask(source, target) # target mask 생성
        target_pos = create_position_vector(target) # position vector 생성

        target = self.token_embedding(target) * self.embedding_scale # 각 토큰 임베딩 + 스케일 조절
        target = self.dropout(target + self.pos_embedding(target_pos))  # target 임베딩 벡터에 위치 정보 임베딩 벡터 추가

        # 디코더 레이어 반복 수행
        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)

        # 정규화
        target = self.layer_norm(target)
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))

        return output, attention_map