import torch.nn as nn
from canlpy.core.models.common.activation_functions import get_activation_function
from canlpy.core.models.bert.model import DenseSkipLayer
from canlpy.core.components.fusion import Fusion


class ErnieFusion(Fusion):
    def __init__(self, hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn):
        super().__init__()

        self.dense_intermediate_tokens = nn.Linear(hidden_size, intermediate_size)
        self.dense_intermediate_ent = nn.Linear(entity_size, intermediate_size)
        self.intermediate_act_fn = get_activation_function(activation_fn) if isinstance(activation_fn, str) else activation_fn

        self.skip_layer_tokens = DenseSkipLayer(intermediate_size,hidden_size,hidden_dropout_prob)
        self.skip_layer_ent = DenseSkipLayer(intermediate_size,entity_size,hidden_dropout_prob)

    def forward(self, attention_tokens, attention_ent):
        
        intermediate_tokens = self.dense_intermediate_tokens(attention_tokens)
        intermediate_ent = self.dense_intermediate_ent(attention_ent)
        intermediate_hidden = self.intermediate_act_fn(intermediate_tokens+intermediate_ent)

        hidden_states = self.skip_layer_tokens(intermediate_hidden,attention_tokens)
        hidden_states_ent = self.skip_layer_ent(intermediate_hidden,attention_ent)

        return hidden_states,hidden_states_ent