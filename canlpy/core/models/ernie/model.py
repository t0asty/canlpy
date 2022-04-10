import torch
import typing
from typing import List,Dict,Tuple
import os
import json
import copy
import math
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import logging

logger = logging.getLogger(__name__)

CONFIG_NAME = 'ernie_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
MAPPING_FILE = 'mapping.json'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class ErnieConfig(object):
    """Configuration class to store the configuration of an `ErnieModel`.
    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 entity_size=100,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 num_attention_heads_ent=4,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_types=None):
        """Constructs ErnieConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `ErnieModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `ErnieModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_types: list() of ERNIE encoders which can be 'sim' (Bert encoder), 
            'mix' (Ernie encoder but no multihead attention for entites) or 'norm' (standard Ernie encoder)
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.entity_size = entity_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_ent = num_attention_heads_ent
        self.hidden_act = hidden_act #Stores a string
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_types = layer_types
       
    @classmethod
    def load_from_json(cls,path):
        config = cls(0)#Create default config
        with open(path, "r", encoding='utf-8') as reader:
            json_config = json.loads(reader.read())
            for key, value in json_config.items():
                config.__dict__[key] = value
        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

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
        return self.weight * x + self.bias

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
        self.intermediate_act_fn = ACT2FN[activation_fn] if isinstance(activation_fn, str) else activation_fn

        #BertOutput_simple
        self.skip_layer_out = DenseSkipLayer(intermediate_size,hidden_size,hidden_dropout_prob)


    #Have same interface for forward for ErnieBlock and BertBlock => act as identity for entities
    #TODO: see if change
    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):

        #BertAttention_simple
        hidden_states = self.attention(hidden_states, attention_mask)

        #BertIntermediate_simple
        intermediate_hidden_states = self.dense_intermediate(hidden_states)
        intermediate_hidden_states = self.intermediate_act_fn(intermediate_hidden_states)

        #BertOutput_simple
        hidden_states = self.skip_layer_out(intermediate_hidden_states,hidden_states)

        return hidden_states,hidden_states_ent

class ErnieFusion(nn.Module):
    def __init__(self, hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn):
        super().__init__()

        #BertIntermediate
        self.dense_intermediate_tokens = nn.Linear(hidden_size, intermediate_size)
        self.dense_intermediate_ent = nn.Linear(entity_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[activation_fn] if isinstance(activation_fn, str) else activation_fn

        #BertOutput
        self.skip_layer_tokens = DenseSkipLayer(intermediate_size,hidden_size,hidden_dropout_prob)
        self.skip_layer_ent = DenseSkipLayer(intermediate_size,entity_size,hidden_dropout_prob)

    def forward(self, attention_tokens, attention_ent):
        
        #BertIntermediate
        intermediate_tokens = self.dense_intermediate_tokens(attention_tokens)
        intermediate_ent = self.dense_intermediate_ent(attention_ent)
        intermediate_hidden = self.intermediate_act_fn(intermediate_tokens+intermediate_ent)

        #BertOutput
        hidden_states = self.skip_layer_tokens(intermediate_hidden,attention_tokens)
        hidden_states_ent = self.skip_layer_ent(intermediate_hidden,attention_ent)

        return hidden_states,hidden_states_ent

class ErnieLayer(nn.Module):
    def __init__(self, hidden_size,entity_size,intermediate_size,num_attention_heads,num_attention_heads_ent,attention_probs_dropout_prob,hidden_dropout_prob,activation_fn):
        super().__init__()

        #BertAttention
        self.attention_tokens = BertAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob)
        self.attention_ent = BertAttention(entity_size,num_attention_heads_ent,attention_probs_dropout_prob,hidden_dropout_prob)

        self.fusion = ErnieFusion(hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn)
        

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):

        #BertAttention
        attention_tokens = self.attention_tokens(hidden_states,attention_mask)
        attention_ent = self.attention_ent(hidden_states_ent,attention_mask_ent) * ent_mask
        
        hidden_states,hidden_states_ent = self.fusion(attention_tokens,attention_ent)

        return hidden_states,hidden_states_ent

class ErnieLayerMix(nn.Module):
    #No multi-head attention + dense for entities
    def __init__(self, hidden_size,entity_size,intermediate_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob,activation_fn):
        super().__init__()
        self.attention_tokens = BertAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob,hidden_dropout_prob)

        self.fusion = ErnieFusion(hidden_size,entity_size,intermediate_size,hidden_dropout_prob,activation_fn)
        

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_tokens = self.attention_tokens(hidden_states,attention_mask)
        attention_ent = hidden_states_ent * ent_mask
        
        hidden_states,hidden_states_ent = self.fusion(attention_tokens,attention_ent)

        return hidden_states,hidden_states_ent

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

class ErnieEncoder(nn.Module):
    def __init__(self, config:ErnieConfig):
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
        ent_mask = ent_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)
        # if self.training:
        #     ent_mask = ent_mask.half().unsqueeze(-1)
        # else:
        #     ent_mask = ent_mask.float().unsqueeze(-1)
        # ent_mask = ent_mask.float().unsqueeze(-1)
        for layer_module in self.layer:
            hidden_states, hidden_states_ent = layer_module(hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers    
    
class PreTrainedErnieModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, ErnieConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `ErnieConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, dir_path, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        
        # Load config
        config_file = os.path.join(dir_path, CONFIG_NAME)
        config = ErnieConfig.load_from_json(config_file)

        model = cls(config, *inputs, **kwargs)

        if(state_dict==None):
            weights_path = os.path.join(dir_path, WEIGHTS_NAME)
            if(torch.cuda.is_available()):
                state_dict = torch.load(weights_path)
            else:
                state_dict = torch.load(weights_path,map_location='cpu')

        missing_keys = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        mapping_file = os.path.join(dir_path, MAPPING_FILE)

        if(os.path.exists(mapping_file)):
            with open(mapping_file, 'rb') as file:
                mapping = json.load(file)

            for old_key,new_key in mapping.items():
                if(new_key in model.state_dict()):
                    model_shape = model.state_dict()[new_key].shape
                    if(model_shape!=state_dict[old_key].shape):
                        logger.error(f'model.state_dict() {new_key}:{model_shape} != state_dict {old_key}:{state_dict[old_key].shape}')
                        
                state_dict[new_key] = state_dict.pop(old_key)

        
        missing_keys,unexpected_keys =  model.load_state_dict(state_dict,strict=False)
        logger.info(f"Missing keys are: \n {missing_keys}")
        logger.info(f"Unexpected keys are: \n {unexpected_keys}")

        return model, missing_keys

class ErnieModel(PreTrainedErnieModel):
    """ERNIE model

    Params:
        config: a ErnieConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        # `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
        #     classifier pretrained on top of the hidden state associated to the first character of the
        #     input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config.vocab_size,config.hidden_size, config.max_position_embeddings,config.hidden_dropout_prob,config.type_vocab_size)
        self.encoder = ErnieEncoder(config)
        #No need in every cases => might want to remove it
        self.pooler = BertPooler(config.hidden_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_ent_mask = ent_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_ent_mask = extended_ent_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_ent_mask = (1.0 - extended_ent_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      input_ent,
                                      extended_ent_mask,
                                      ent_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class ErnieForMaskedLM(PreTrainedErnieModel):
    """Ernie model with the masked language modeling head.
    This module comprises the Ernie model followed by the masked language modeling head.

    Params:
        config: a ErnieConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ErnieConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = ErnieForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        self.cls = BertOnlyMLMHead(config, self.model.embeddings.word_embeddings.weight)
        #Recursively initalize all the weights
        self.apply(self.init_weights)

    def forward(self, input_ids, input_ents, ent_mask=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.model(input_ids, token_type_ids, attention_mask, input_ents, ent_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
        
class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

#HEADS: Copy paste from Ernie github & UNTESTED 

class BertForPreTraining(PreTrainedErnieModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        self.cls = BertPreTrainingHeads(config, self.model.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, 
        input_ent=None, ent_mask=None, next_sentence_label=None, candidate=None, ent_labels=None):
        # the id in ent_labels should be consistent with the order of candidate.
        sequence_output, pooled_output = self.model(input_ids, token_type_ids, attention_mask, input_ent, ent_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score, prediction_scores_ent = self.cls(sequence_output, pooled_output, candidate)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            ent_ae_loss = loss_fct(prediction_scores_ent.view(-1, candidate.size()[0]), ent_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + ent_ae_loss
            original_loss = masked_lm_loss + next_sentence_loss
            return total_loss, original_loss
        else:
            return prediction_scores, seq_relationship_score, prediction_scores_ent

class BertEntPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        #TODO: might want to change
        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = 100
        self.transform = BertPredictionHeadTransform(config_ent)

    def forward(self, hidden_states, candidate):
        hidden_states = self.transform(hidden_states)
        candidate = torch.squeeze(candidate, 0)
        # hidden_states [batch_size, max_seq, dim]
        # candidate [entity_num_in_the_batch, dim]
        # return [batch_size, max_seq, entity_num_in_the_batch]
        return torch.matmul(hidden_states, candidate.t())

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.predictions_ent = BertEntPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, candidate):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        prediction_scores_ent = self.predictions_ent(sequence_output, candidate)
        return prediction_scores, seq_relationship_score, prediction_scores_ent

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BertForNextSentencePrediction(PreTrainedErnieModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.model = ErnieModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.model(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score

class BertForEntityTyping(PreTrainedErnieModel):
    def __init__(self, config, num_labels=2):
        super(BertForEntityTyping, self).__init__(config)
        self.num_labels = num_labels
        self.bert = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, num_labels, False)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.typing(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

class BertForSTSB(PreTrainedErnieModel):
    def __init__(self, config, num_labels=2):
        super(BertForSTSB, self).__init__(config)
        self.num_labels = 2
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

        self.m = torch.nn.LogSoftmax(-1)
        self.mm = torch.nn.Softmax(-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        _, pooled_output = self.model(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        probs = self.m(logits)

        if labels is not None:
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            per_example_loss = -torch.sum(labels * probs, -1)
            loss = torch.mean(per_example_loss)
            return loss
        else:
            return self.mm(logits)

class BertForSequenceClassification(PreTrainedErnieModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        _, pooled_output = self.model(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForNQ(PreTrainedErnieModel):

    def __init__(self, config, num_choices=2):
        super().__init__(config)
        self.num_choices = num_choices
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, choice_mask=None, labels=None):
        #if choice_mask==None:
        #    _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        #    pooled_output = self.dropout(pooled_output)
        #    logits = self.classifier(pooled_output)
        #    return logits
            
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_input_ent = input_ent.view(-1, input_ent.size(-2), input_ent.size(-1))
        flat_ent_mask = ent_mask.view(-1, ent_mask.size(-1))
        _, pooled_output = self.model(flat_input_ids, flat_token_type_ids, flat_attention_mask, flat_input_ent, flat_ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        null_socre = torch.zeros([labels.shape[0],1]).cuda()
        reshaped_logits = torch.cat([null_socre, reshaped_logits], -1) + choice_mask

        if labels is not None:
            weight = torch.FloatTensor([0.3]+[1]*16).cuda()
            loss_fct = CrossEntropyLoss(weight)
            loss = loss_fct(reshaped_logits, labels+1)
            return loss
        else:
            return reshaped_logits

class BertForMultipleChoice(PreTrainedErnieModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices=2):
        super().__init__(config)
        self.num_choices = num_choices
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.model(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits

class BertForTokenClassification(PreTrainedErnieModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForQuestionAnswering(PreTrainedErnieModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a BertConfig class instance with the configuration to build a new model, or
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-base-multilingual`
                . `bert-base-chinese`
                The pre-trained model will be downloaded and cached if needed.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.model(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
