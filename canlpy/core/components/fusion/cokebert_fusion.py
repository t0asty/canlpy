# This file is adapted from the CokeBERT repository at https://github.com/thunlp/CokeBERT
# Copyright by the CokeBERT authors.

import math
import torch.nn as nn
from canlpy.core.components.fusion import Fusion

class DK_fusion(Fusion):
    """ A class for the Text and Knowledge Fusion in a `DKEncoderLayer` from CokeBert.
    """
    def __init__(self, k_v_dim, layer_no):
        """Constructs a `DK_fusion` module
        
        Args:
            k_v_dim (int): size of the k and v vectors (internal Knowledge representation)
            layer_no (int): Number of the layer (assigned in reverse order)
        """
        super().__init__()
        self.k_v_dim = k_v_dim
        self.number = layer_no

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=layer_no+1)
    
    def forward(self, q_i, k, v):
        """
        Args:
            q_i: internal text representation: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]
            k: internal knowledge representation: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]
            v: internal knowledge representation: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]

        Returns:
            sentence_entity_reps: internal entity representations: torch.Tensor of shape [batch_size, sequence_length, entity_size]
        """
        attention = ((q_i * k).sum(self.number+2)).div(math.sqrt(self.k_v_dim))

        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax(self.leaky_relu(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to


        attention = attention.unsqueeze(self.number+1)

        sentence_entity_reps = attention.matmul(v).squeeze(self.number+1)

        return sentence_entity_reps
