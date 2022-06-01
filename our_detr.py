import torch
import torch.nn.functional as F
from torch import nn
from positional_encodings import PositionalEncoding1D
import numpy as np
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

        outputs_target = self.target_embed(hs)
        #outputs_target = ()
        outputs_aspect = self.aspect_embed(hs)
        outputs_sentiment = self.sentiment_embed(hs)
        out = {'pred_target': outputs_target[-1], 'pred_aspect': outputs_aspect[-1], 'pred_sentiment': outputs_sentiment[-1]}
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth sand the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_targets, num_aspects, matcher, weight_dict, eos_coef1, eos_coef2, losses):
        """ Create the criterion.
        Parameters:
            num_targets: number of target categories, omitting the special no-object category
            num_aspects: number of aspect categories
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_targets = num_targets
        self.num_aspects = num_aspects
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef1 = eos_coef1
        self.eos_coef2 = eos_coef2
        self.losses = losses
        empty_weight = torch.ones(self.num_targets + 1)
        empty_weight[-1] = self.eos_coef1
        self.register_buffer('empty_weight', empty_weight)

    def loss_targets(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "targets" containing a tensor of dim [nb_targets]
        """
        assert 'pred_target' in outputs
        src_logits = outputs['pred_target']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["target"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_targets,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_target = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_target': loss_target}

        #if log:
            # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
        
    def loss_aspects(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "aspects" containing a tensor of dim [nb_targets]
        """
        assert 'pred_aspect' in outputs
        src_logits = outputs['pred_aspect']

        idx = self._get_src_permutation_idx(indices)
        target_aspect_o = torch.cat([t["aspect"][J] for t, (_, J) in zip(targets, indices)])
        target_aspect = torch.full(src_logits.shape[:2], self.num_aspects,
                                    dtype=torch.int64, device=src_logits.device)
        target_aspect[idx] = target_aspect_o

        loss_aspect = F.cross_entropy(src_logits.transpose(1, 2), target_aspect)
        losses = {'loss_aspect': loss_aspect}

        #if log:
            # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_sentiment(self, outputs, targets, indices):

        assert 'pred_sentiment' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sentiment = outputs['pred_sentiment'][idx]
        target_sentiment = torch.cat([t['sentiment'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_sentiment = F.mse_loss(src_sentiment, target_sentiment, reduction='mean')
        loss_sentiment = 10*loss_sentiment
        losses = {'loss_sentiment': loss_sentiment}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'target': self.loss_targets,
            'aspect': self.loss_aspects,
            'sentiment': self.loss_sentiment,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    n = np.zeros(shape=(436, 32, 768))
    x = torch.tensor(n, dtype=torch.float32)
    p_enc_1d_model = PositionalEncoding1D(768)
    p_enc = p_enc_1d_model(x)

    transformer = build_transformer(1)
    matcher = build_matcher(1)

    model = DETR(
        transformer,
        Positional_Encoding = p_enc,
        num_aspect = 4,
        num_target = 226,
        num_queries= 32,
        aux_loss=False,
    )
    
    weight_dict = {'loss_target': 1, 'loss_aspect': 1, 'loss_sentiment' : 1}
    losses = ['target', 'aspect', 'sentiment']
    criterion = SetCriterion(4, 226, matcher, weight_dict, 0.1 , 0.1, losses)

    return criterion
