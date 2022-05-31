import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_target: float = 1, cost_aspect: float = 1,  cost_sentiment: float = 1):
        """Creates the matcher

        Params:
            cost_target: This is the relative weight of the classification error in the matching cost of target
            cost_aspect: This is the relative weight of the classification error in the matching cost of target
            cost_sentiment: This is the relative weight of the loss of the sentiment in the matching cost
        """
        super().__init__()
        self.cost_target = cost_target
        self.cost_aspect = cost_aspect
        self.cost_sentiment = cost_sentiment
        assert cost_target != 0 or cost_sentiment != 0 or cost_target != 0, "all costs cant be 0"

    @torch.no_grad()
    #여기서 output은 DETR에서 나온 output이고, target은 위에 가공한 dataset인 'targets'을 쓰면 될 것 같다.
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_target": Tensor of dim [batch_size, num_queries, num_targets] with the classification logits
                 "pred_aspects" : Tensor of dim [batch_size, num_queries, num_aspect] with classification logits
                 "pred_sentiment": Tensor of dim [batch_size, num_queries, 1] with predicted sentiment score

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "targets": Tensor of dim [num_targets] 
                 "aspects": Tensor of dim [num_targets] 
                 "sentiment" : tensor of dim [num_targets]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_target"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_target_prob = outputs["pred_target"].flatten(0, 1).softmax(-1)
        out_aspect_prob = outputs["pred_aspect"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_sentiment = outputs["pred_sentiment"].flatten(0, 1)  # [batch_size * num_queries, 1]

        # Also concat the target labels and boxes
        tgt_targets = torch.cat([v["target"] for v in targets])
        tgt_aspects = torch.cat([v["aspect"] for v in targets])
        tgt_sentiment = torch.cat([v["sentiment"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_target = -out_target_prob[:, tgt_targets]
        cost_aspect = -out_aspect_prob[:, tgt_aspects]
        
        #mean squared error loss
        loss = nn.MSELoss()
        # Compute mean squared error loss for sentiment
        cost_sentiment = loss(out_sentiment, tgt_sentiment)

        # Final cost matrix
        C = self.cost_target * cost_target + self.cost_aspect * cost_aspect + self.cost_sentiment * cost_sentiment
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["sentiment"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_target=args.set_cost_target, cost_aspect = args.set_cost_aspect, cost_sentiment=args.set_cost_sentiment)
