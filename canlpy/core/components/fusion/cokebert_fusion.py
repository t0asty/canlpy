import math
import torch.nn as nn
from canlpy.core.components.fusion import Fusion

class DK_fusion(Fusion):
    def __init__(self, k_v_dim, layer_no):
        super().__init__()
        self.k_v_dim = k_v_dim
        self.number = layer_no

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=layer_no+1)
    
    def forward(self, q_i, k, v):
        attention = ((q_i * k).sum(self.number+2)).div(math.sqrt(self.k_v_dim))

        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax(self.leaky_relu(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to


        attention = attention.unsqueeze(self.number+1)

        sentence_entity_reps = attention.matmul(v).squeeze(self.number+1)

        return sentence_entity_reps
