# This file is adapted from the ERNIE repository at https://github.com/thunlp/ERNIE
# Copyright by the ERNIE authors.

import copy
import torch
from torch import nn
from canlpy.core.models.bert.model import BertLayer, DenseSkipLayer, BertAttention
from canlpy.core.components.fusion import ErnieFusion

class ErnieLayer(nn.Module):
    """
    An Ernie Layer, takes as input tokens/entity embeddings and masks and outputs tokens/entity embeddings

    Parameters:
        hidden_size: Size of the encoder layers and the pooler layer.
        entity_size: Size of the entity embeddings,
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder for the tokens.
        num_attention_heads_ent: Number of attention heads for each attention layer in
            the Transformer encoder for the entities.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        activation_fn: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the tokens embeddings 
        attention_mask: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
        hidden_states_ent: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the entity embeddings 
        attention_mask_ent: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
        ent_mask: a torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1] used to indicate which position correspond to entities or not
    """
    def __init__(self, hidden_size,entity_size,intermediate_size,num_attention_heads,num_attention_heads_ent,attention_probs_dropout_prob,hidden_dropout_prob,activation_fn):
        super().__init__()

        self.attention_tokens = BertAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob)
        self.attention_ent = BertAttention(entity_size,num_attention_heads_ent,attention_probs_dropout_prob,hidden_dropout_prob)

        self.fusion = ErnieFusion(hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn)
        

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):

        attention_tokens = self.attention_tokens(hidden_states,attention_mask)
        attention_ent = self.attention_ent(hidden_states_ent,attention_mask_ent) * ent_mask
        
        hidden_states,hidden_states_ent = self.fusion(attention_tokens,attention_ent)

        return hidden_states,hidden_states_ent

class ErnieLayerMix(nn.Module):
    """
    An Ernie Layer, takes as input tokens/entity embeddings and masks and outputs tokens/entity embeddings. 
    Differs from ErnieLayer by not applying any multi-head attention and dense layer on the entities before fusion.  

    Parameters:
        hidden_size: Size of the encoder layers and the pooler layer.
        entity_size: Size of the entity embeddings,
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder for the tokens.
        num_attention_heads_ent: Number of attention heads for each attention layer in
            the Transformer encoder for the entities.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        activation_fn: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the tokens embeddings 
        attention_mask: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
        hidden_states_ent: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the entity embeddings 
        attention_mask_ent: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
        ent_mask: a torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1] used to indicate which position correspond to entities or not
    
    Returns:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the new tokens embeddings 
        hidden_states_ent: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the new entity embeddings 
    """
    def __init__(self, hidden_size,entity_size,intermediate_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob,activation_fn):
        super().__init__()
        self.attention_tokens = BertAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob)

        self.fusion = ErnieFusion(hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn)
        

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_tokens = self.attention_tokens(hidden_states,attention_mask)
        attention_ent = hidden_states_ent * ent_mask
        
        hidden_states,hidden_states_ent = self.fusion(attention_tokens,attention_ent)

        return hidden_states,hidden_states_ent

class ErnieEncoder(nn.Module):
    """
    An ErnieEncoder takes as input tokens/entity embeddings and masks and outputs tokens/entity embeddings

    Parameters:
        config: an `ErnieConfig` file

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the tokens embeddings 
        attention_mask: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
        hidden_states_ent: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the entity embeddings 
        attention_mask_ent: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
        ent_mask: a torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1] used to indicate which position correspond to entities or not
        output_all_encoded_layers: whether to output all encoder layers or not
        
    Returns:
        if output_all_encoded_layers:
            the tokens embeddings at each layer
        if not output_all_encoded_layers
            the tokens embeddings at the last layer
    """
    def __init__(self, config):
        super().__init__()
        bert_layer = BertLayer(config.hidden_size,config.intermediate_size,config.num_attention_heads,config.attention_probs_dropout_prob,config.hidden_dropout_prob,config.hidden_act)
        ernie_layer = ErnieLayer(config.hidden_size,config.entity_size,config.intermediate_size,config.num_attention_heads,config.num_attention_heads_ent,config.attention_probs_dropout_prob,config.hidden_dropout_prob,config.hidden_act)
        ernie_layer_mix = ErnieLayerMix(config.hidden_size,config.entity_size,config.intermediate_size,config.num_attention_heads,config.attention_probs_dropout_prob,config.hidden_dropout_prob,config.hidden_act)
        layers = []
        for t in config.layer_types:
            if t == "sim":
                layers.append(copy.deepcopy(bert_layer))
            if t == "norm":
                layers.append(copy.deepcopy(ernie_layer))
            if t == "mix":
                layers.append(copy.deepcopy(ernie_layer_mix))
        for _ in range(config.num_hidden_layers-len(layers)):
            layers.append(copy.deepcopy(bert_layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        ent_mask = ent_mask.unsqueeze(-1)#.to(dtype=torch.float).unsqueeze(-1)

        for layer_module in self.layer:
            if(isinstance(layer_module, BertLayer)):
                hidden_states = layer_module(hidden_states, attention_mask)
            else:
                hidden_states, hidden_states_ent = layer_module(hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers    