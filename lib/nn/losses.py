import torch
from torch import nn

from typing import Text, Dict, List


class MarginRankingLoss:
    def __init__(self, device, margin):
        self.device = device
        self.calculator = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def calculate(self, score_map: Dict[Text, torch.Tensor], comparisons: Dict[Text, int]):
        loss = 0.0
        for pair, preferred in comparisons.items():
            g1, g2 = eval(pair)
            preferred_t = torch.tensor(preferred, device=self.device)
            loss += self.calculator(score_map[g1], score_map[g2], preferred_t)
        return loss


class LogSigmoidRankingLoss:
    def __init__(self, device):
        self.device = device

    def calculate(self, score_map: Dict[Text, torch.Tensor], comparisons: Dict[Text, int]):
        loss = 0.0
        for pair, preferred in comparisons.items():
            g1, g2 = eval(pair)
            preferred_t = torch.tensor(preferred, device=self.device)
            loss += -nn.functional.logsigmoid((score_map[g1] - score_map[g2]) * preferred_t)
        return loss
