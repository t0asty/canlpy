import typing
from typing import List,Dict,Tuple
import os
import json
import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import logging

from torch import layer_norm

from canlpy.core.components.activation_functions import get_activation_function

#Standard Linear BERT Self-Attention with no feed_forward network (hidden_size->hidden_size)
#NOTE: Later improvement would be to use PyTorch multi-head attention
# => must verify that it produces the same result with pre-trained ERNIE
class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer

    Parameters:
        hidden_size: dimension of the token embeddings.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder for the tokens.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the tokens embeddings 
        attention_mask: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
    
    Returns:
        context_layer: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size], the embeddings 
        after the attention.
    """
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


class BertAttention(nn.Module):

    """
    A BERT attention layer

    Parameters:
        hidden_size: dimension of the token embeddings.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder for the tokens.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        hidden_dropout_prob: The dropout probability for the fully connected
            layer.

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size]
            containing the tokens embeddings.
        attention_mask: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].
    
    Returns:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,tokens_embedding_size] the produced embeddings.
    """
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
    """
    Construct the BERT embeddings of the token_ids from word, position and token_type embeddings.

    Parameters:
        vocab_size: the number of tokens in the vocabulary.
        hidden_size: dimension of the token embeddings.
        max_position_embeddings: the maximum sequence length that this model might ever be used with.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        hidden_dropout_prob: The dropout probability for the embeddings.
        type_vocab_size: the vocabulary size of the `token_type_ids`.

    Args:
        input_ids: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary
        token_type_ids: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).

    Returns:
        embeddings: a torch.FloatTensor of shape [batch_size, sequence_length,embedding_dimension], 
        the embedding corresponding to the provided ids.

    """
    def __init__(self, vocab_size,hidden_size,max_position_embeddings,hidden_dropout_prob,type_vocab_size):
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

class DenseSkipLayer(nn.Module):

    """
    Performs Linear + Dropout + SkipLayer + LayerNorm

    Parameters:
        input_size: the size of the input embeddings.
        output_size: the size of the output embeddings.
        dropout_prob: the dropout ratio.

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,`input_size`]
            containing the token embeddings
        skip_tensor: a torch.FloatTensor of shape [batch_size, sequence_length,`output_size`] that is added
            to the tensor after the dense layer
    Returns:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,`output_size`]
    """

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

class BertLayer(nn.Module):
    """
    Correspond to a standard encoder BERT layer

    Parameters:
        hidden_size: Size of the encoder layers and the pooler layer.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder for the tokens.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected
            layers.
        activation_fn: The non-linear activation function (function or string). 
            If string, "gelu", "relu" and "swish" are supported.

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,`input_size`]
            containing the token embeddings
        attention_mask: a torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1].

    Returns:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,`hidden_size`]

    """

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
    """
    Does the classification of the CLS token (first token)
    
    Parameters:
        hidden_size: dimension of the CLS token

    Args:
        hidden_states: a torch.FloatTensor of shape [batch_size, sequence_length,`hidden_size`]
            containing the token embeddings
        
    Returns:
        pooled_output: a torch.FloatTensor of shape [batch_size,`hidden_size`]
    """
    
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

#NOTE: does the produce the exact same results as PyTorch LayerNorm
#=> could still replace by it
class LayerNorm(nn.Module):
    """
    Performs a layer nornlizationon the tensor
    
    Parameters:
        hidden_size: dimension of the token embeddings.

    Args:
        x: a torch.FloatTensor to perform layer norm on.
    Returns:
        res: the normalized tensor.
        
    Returns:
        pooled_output: a torch.FloatTensor of shape [batch_size,`hidden_size`]
    """
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
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
    """
    Recursively initialize all weights 

    Args:
        module: the nn.Module to recursively initialize.
        initializer_range: the std_dev of the normal initializer for
                    initializing all weight matrices.
    
    """
    def _do_init(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(m, LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

        else:
            for _, submodule in m.named_children():
                _do_init(submodule)
                    
    _do_init(module)