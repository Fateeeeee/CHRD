
from torch import nn
import torch.nn.functional as F
import torch

class CHRDistiller(nn.Module):
    def __init__(self, num_classes, lambda_consist=0.01, lambda_relation=0.1, temperature=2.0):
        super().__init__()
        self.lambda_consist = lambda_consist
        self.lambda_relation = lambda_relation
        self.temp = temperature
        self.num_classes = num_classes
        
    def forward(self, s_main, s_aux, t_main, t_aux):
        consist_loss = self.head_consistency(s_main, s_aux, t_main, t_aux)
        relation_loss = self.relation_distillation(s_main, s_aux, t_main, t_aux)
        return  self.lambda_relation * relation_loss, self.lambda_consist * consist_loss
    
    def head_consistency(self, s_main, s_aux, t_main, t_aux):
        with torch.no_grad():
            t_consist = self.compute_consistency(t_main, t_aux)

        s_consist = self.compute_consistency(s_main, s_aux)
        return F.kl_div(
            F.log_softmax(s_consist / self.temp, dim=1),
            F.softmax(t_consist.detach() / self.temp, dim=1),
            reduction='batchmean'
        )
    
    def relation_distillation(self, s_main, s_aux, t_main, t_aux):
        with torch.no_grad():
            t_relation = self.compute_relation_matrix(t_main, t_aux)

        s_relation = self.compute_relation_matrix(s_main, s_aux)
        return F.kl_div(
            F.log_softmax(s_relation / self.temp, dim=-1),
            F.softmax(t_relation.detach() / self.temp, dim=-1),
            reduction='batchmean'
        )
    