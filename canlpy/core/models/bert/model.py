import torch
import torch.nn as nn
import typing
from typing import List,Dict,Tuple
import os
import json
import copy
import math
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import logging

from torch import layer_norm

from canlpy.core.components.activation_functions import get_activation_function

#Standard Linear BERT Self-Attention with no feed_forward network (hidden_size->hidden_size)
#NOTE: Later improvement would be to use PyTorch multi-head attention
#BertSelfAttention
class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size,num_attention_heads,attention_probs_dropout_prob):
        super().__init__()
        if (hidden_size % num_attention_heads != 0):
            raise ValueError(f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                "heads ({num_attention_heads})")

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size // num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    #Used to go from 1 headed attention to multi-headed attention
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
#BertAttention_simple
class BertAttention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob):
        super().__init__()
        ##BertSelfAttention
        self.multi_head_attention = MultiHeadAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob)#BertSelfAttention
        ##BertSelfOutput
        self.skip_layer = DenseSkipLayer(hidden_size,hidden_size,hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask):
        attention = self.multi_head_attention(hidden_states, attention_mask)
        hidden_states = self.skip_layer(attention,hidden_states)

        return hidden_states

class BertEmbeddings(nn.Module):
    """Construct the BERT embeddings of the token_ids from word, position and token_type embeddings."""
    def __init__(self, vocab_size,hidden_size,max_position_embeddings,hidden_dropout_prob,type_vocab_size):
        '''max_position_embeddings: The maximum sequence length that this model might ever be used with
        type_vocab_size: The vocabulary size of the `token_type_ids`'''
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

##Linear+ LayerNorm+ Dropout
class DenseSkipLayer(nn.Module):
    def __init__(self, input_size,output_size,dropout_prob): 
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.LayerNorm = LayerNorm(output_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, skip_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #Normalize after adding skip connection
        hidden_states = self.LayerNorm(hidden_states + skip_tensor)

        return hidden_states

#BertLayer_simple
class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob,activation_fn):
        super().__init__()
        #BertAttention_simple
        self.attention = BertAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob)

        #BertIntermediate_simple
        self.dense_intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation_function(activation_fn) if isinstance(activation_fn, str) else activation_fn

        #BertOutput_simple
        self.skip_layer_out = DenseSkipLayer(intermediate_size,hidden_size,hidden_dropout_prob)



    def forward(self, hidden_states, attention_mask):

        #BertAttention_simple
        hidden_states = self.attention(hidden_states, attention_mask)

        #BertIntermediate_simple
        intermediate_hidden_states = self.dense_intermediate(hidden_states)
        intermediate_hidden_states = self.intermediate_act_fn(intermediate_hidden_states)

        #BertOutput_simple
        hidden_states = self.skip_layer_out(intermediate_hidden_states,hidden_states)

        return hidden_states

class BertPooler(nn.Module):
    '''Does the classification of the CLS token'''
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token (CLS).
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        res = self.weight * x + self.bias

        return res


def init_weights(module, initializer_range):
    """Recursively initialize all weights """
    def _do_init(m):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        else:
            for submodule in module.modules():
                _do_init(submodule)