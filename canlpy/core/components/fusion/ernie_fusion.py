import torch.nn as nn
from canlpy.core.components.activation_functions import get_activation_function
from canlpy.core.models.bert.model import DenseSkipLayer
from canlpy.core.components.fusion import Fusion


class ErnieFusion(Fusion):
    """ Class for the Fusion of Text and Knowledge representations in ERNIE

    """
    def __init__(self, hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn):
        """ Constructs an `ErnieFusion` module
        
        Args:
            hidden_size (int): size of the Text representations
            entity_size (int): size of the Knowledge representations
            intermediate_size (int): hidden size of the Fusion layer
            hidden_dropout_prob (float): dropout probability
            activation_fn: str or function: activation function of Fusion layer
        """
        super().__init__()

        self.dense_intermediate_tokens = nn.Linear(hidden_size, intermediate_size)
        self.dense_intermediate_ent = nn.Linear(entity_size, intermediate_size)
        self.intermediate_act_fn = get_activation_function(activation_fn) if isinstance(activation_fn, str) else activation_fn

        self.skip_layer_tokens = DenseSkipLayer(intermediate_size,hidden_size,hidden_dropout_prob)
        self.skip_layer_ent = DenseSkipLayer(intermediate_size,entity_size,hidden_dropout_prob)

    def forward(self, attention_tokens, attention_ent):
        """ Forward pass through the model
        
        Args:
            attention_tokens: text representations, a torch.FloatTensor of shape [batch_size, sequence_length, hidden_size]
            attention_ent: entity representations, a torch.FloatTensor of shape [batch_size, sequence_length, entity_size]

        Returns:
            hidden_states: New text representations, a torch.FloatTensor of shape [batch_size, sequence_length, hidden_size]
            hidden_states_ent: New entity representations, a torch.FloatTensor of shape [batch_size, sequence_length, entity_size]
        """
        intermediate_tokens = self.dense_intermediate_tokens(attention_tokens)
        intermediate_ent = self.dense_intermediate_ent(attention_ent)
        intermediate_hidden = self.intermediate_act_fn(intermediate_tokens+intermediate_ent)

        hidden_states = self.skip_layer_tokens(intermediate_hidden,attention_tokens)
        hidden_states_ent = self.skip_layer_ent(intermediate_hidden,attention_ent)

        return hidden_states,hidden_states_ent