import torch
import torch.nn.functional as F
from torch import nn

from our_matcher import build_matcher
#from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                           dice_loss, sigmoid_focal_loss)
from our_transformer import build_transformer

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.tanh(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer, Positional_Encoding, num_aspect, num_target, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.Positional_Encoding = Positional_Encoding
        hidden_dim = transformer.d_model
        self.aspect_embed = nn.Linear(hidden_dim, num_aspect + 1)
        self.target_embed = nn.Linear(hidden_dim, num_target + 1)
        self.sentiment_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.aux_loss = aux_loss

    def forward(self,input_ids, attention_masks):
        """ input_ids와 attention_masks는 저희가 쓰는 dataset인 fiqa dataset을 bert의 tokenizer를 써서 추출한거에요
            
            transformer에 넣어서 나오면 (498,32,768)인 모양의 텐서가 나오는데, 
              498은 저희 training dataset batchsize이고
              32는 object query 갯수
              768은 embedding_dimension입니다.
            *(batch_size, num_object_query, embedding_dimension)

            이 return 값은 transformer의 마지막 output인 hs가 각각 target_embed, aspect_embed, sentiment_embed layer를 통과해서,
            outputs_Target, outputs_aspect, output_sentiment를 내뱉습니다. 아직 sentiment_embed_layer는 다 완성을 못했음.
            outputs_target의 경우 이 코드 밑에 
            'print(targets[aspect].unique)'
            'print(targets[target].unique)' 이거를 실행시켜보시면 dataset에서 aspect는 총 5개만 존재하고, target은 226개가 존재합니다.
            그래서 output_aspect는 각각의 sentence에 대해 총 5가지의 class를 가질수 있는 것이니까 498

            기본적으로 aspect
        """
        hs = self.transformer(input_ids, attention_masks, self.query_embed.weight, self.Positional_Encoding)
        #hs = (498,32,768)모양의 텐서 

        outputs_target = self.target_embed(hs)
        #outputs_target = ()
        outputs_aspect = self.aspect_embed(hs)
        outputs_sentiment = self.sentiment_embed(hs)
        out = {'pred_target': outputs_target[-1], 'pred_aspect': outputs_aspect[-1], 'pred_sentiment': outputs_sentiment[-1]}
        return out

