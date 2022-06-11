# -*- coding: utf-8 -*-
"""Re-Implementation of the CokeBert Model (Su et al., 2020)

This module contains several versions of CokeBert for different 
fine-tune tasks. 

"""


import copy
import json
import math
import os
import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from canlpy.core.models.bert.model import BertEmbeddings, BertPooler, init_weights, LayerNorm
from canlpy.core.models.ernie.components import ErnieEncoder
from canlpy.core.components.heads import BertOnlyMLMHead
from canlpy.core.components.fusion.cokebert_fusion import DK_fusion

logger = logging.getLogger(__name__)

CONFIG_NAME = 'cokebert_config.json'
"""str: name of the config file in the checkpoint"""
WEIGHTS_NAME = 'pytorch_model.bin'
"""str: name of the file containing weights in the checkpoint"""
MAPPING_FILE = 'mapping.json'
"""str: name of the file containing the mapping in the checkpoint"""

class CokeBertConfig():
    """Configuration class to store the configuration of a `CokeBertModel`.
    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 entity_size=200,
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
                 layer_types=[],
                 k_v_dim=100,
                 q_dim=768,
                 dk_layers=2):
        """Constructs CokeBertConfig.

        Args:
            vocab_size (int): Vocabulary size of `inputs_ids` in `CokeBertModel`.
            hidden_size (int): Size of the encoder layers and the pooler layer.
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            num_attention_heads (int): Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size (int): The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob (int): The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (float): The dropout ratio for the attention
                probabilities.
            max_position_embeddings (int): The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size (int): The vocabulary size of the `token_type_ids` passed into
                `CokeBertModel`.
            initializer_range (float): The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_types (list): list of `ErnieLayer`s which can be 'sim' (Bert encoder), 
                'mix' (Ernie encoder but no multihead attention for entites) or 'norm' (standard Ernie encoder)
            k_v_dim (int): Size of the hidden knowledge representation in the dynamic knowledge encoder
            q_dim (int): Size of the hidden text representation in the dynamic knowledge encoder
            dk_layers (int): Number of layers in the dynamic knowledge encoder
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
        self.k_v_dim = k_v_dim
        self.q_dim = q_dim
        self.dk_layers = dk_layers
       
    @classmethod
    def load_from_json(cls,path):
        """Loads config from json file
        
        Args:
            cls: current CokeBertConfig Class
            path (str): path to config.json file

        Returns:
            config: Loaded CokeBertConfig
        """
        config = cls(0)#Create default config
        with open(path, "r", encoding='utf-8') as reader:
            json_config = json.loads(reader.read())
            for key, value in json_config.items():
                config.__dict__[key] = value
        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string.
        
        Returns:
            JSON-String of Config
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        
    def to_dict(self):
        """Serializes this instance to a Python dictionary.
        
        Returns:
            output: Python dictionary of Config
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_text_encoder_config(self):
        """Splits Config to create `ErnieEncoder` as TextEncoder
        
        Returns:
            CokeBertConfig for TextEncoder
        """
        params = self.to_dict()
        params['layer_types'] = [x for x in params['layer_types'] if x == 'sim']
        params['num_hidden_layers'] = len(params['layer_types'])
        return CokeBertConfig(**params)

    def to_knowl_encoder_config(self):
        """Splits Config to create `ErnieEncoder` as KnowledgeEncoder
        
        Returns:
            CokeBertConfig for KnowledgeEncoder
        """
        params = self.to_dict()
        params['layer_types'] = [x for x in params['layer_types'] if x != 'sim']
        params['num_hidden_layers'] = len(params['layer_types'])
        return CokeBertConfig(**params)

class PreTrainedCokeBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        """"""
        super().__init__()
        if not isinstance(config, CokeBertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `CokeBertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.

        Args:
            module: one of `nn.Linear`, `nn.Embedding`, `LayerNorm`, module to initialize weights of
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

        Args:
            dir_path (str): a path or url to a pretrained model archive containing:

                    . `bert_config.json` a configuration file for the model

                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance

            state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of Google pre-trained models
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        
        # Load config
        config_file = os.path.join(dir_path, CONFIG_NAME)
        config = CokeBertConfig.load_from_json(config_file)

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
                if(new_key in model.state_dict() and old_key in state_dict.keys()):
                    model_shape = model.state_dict()[new_key].shape
                    if(model_shape!=state_dict[old_key].shape):
                        logger.error(f'model.state_dict() {new_key}:{model_shape} != state_dict {old_key}:{state_dict[old_key].shape}')
                        
                    state_dict[new_key] = state_dict.pop(old_key)

        
        missing_keys,unexpected_keys =  model.load_state_dict(state_dict,strict=False)
        logger.info(f"Missing keys are: \n {missing_keys}")
        logger.info(f"Unexpected keys are: \n {unexpected_keys}")

        return model, missing_keys

class CokeBertModel(PreTrainedCokeBertModel):
    """ A class to handle the Transformer Model (without fine-tuning head)"""
    def __init__(self, config):
        """ Constructs a CokeBertModel

        Args:
            config (`CokeBertConfig`): The config that sets the model's hyperparameters
        """
        super(CokeBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.max_position_embeddings, config.hidden_dropout_prob, config.type_vocab_size)
        self.text_encoder = ErnieEncoder(config.to_text_encoder_config())
        self.knowledge_encoder = ErnieEncoder(config.to_knowl_encoder_config())
        self.pooler = BertPooler(config.hidden_size)
        self.dk_encoder = DKEncoder(config.k_v_dim, config.q_dim, config.dk_layers)

        self.k_v_dim = config.k_v_dim
        self.entity_size = config.entity_size
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True, k_v_s=None):
        """ Forward pass through the `CokeBertModel`

        Args:
            input_ids: a torch.LongTensor of shape [batch_size, sequence_length]
                with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
                `extract_features.py`, `run_classifier.py` and `run_squad.py`)
            token_type_ids: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
                types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
                a `sentence B` token (see BERT paper for more details).
            attention_mask: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
                input sequence length in the current batch. It's the mask that we typically use for attention when
                a batch has varying length sentences.
            input_ent: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                with the entities embeddings
            ent_mask: a torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]
            output_all_encoded_layers: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
            k_v_s: list of (k_i, v_i) vectors as input for the dynamic knowledge encoder

        Returns:
            sequence_output, pooled_output

            sequence_output: the full sequence of hidden-states corresponding
                    to the last attention block of shape [batch_size, sequence_length, hidden_size]       
            pooled_output: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
                classifier pretrained on top of the hidden state associated to the first character of the
                input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
        """
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
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_ent_mask = extended_ent_mask.to(dtype=torch.float32)
        extended_ent_mask = (1.0 - extended_ent_mask) * -10000.0

        #############################################################

        embedding_output = self.embeddings(input_ids, token_type_ids)

        ##
        all_encoder_layers=list()
        ent_mask = ent_mask.to(dtype=torch.float32)#.unsqueeze(-1)
        ##

        all_encoder_layers = self.text_encoder(embedding_output, extended_attention_mask, input_ent, extended_ent_mask, ent_mask, output_all_encoded_layers=output_all_encoded_layers)

        hidden_states = all_encoder_layers[-1]

        if len(input_ent[input_ent!=0]) == 0:
            # no input entities -> return 0s
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1],self.entity_size)
        else:

            hidden_states_ent = self.dk_encoder(input_ent, hidden_states, k_v_s)

        encoded_layers = self.knowledge_encoder(hidden_states, extended_attention_mask,hidden_states_ent, extended_ent_mask, ent_mask, output_all_encoded_layers=output_all_encoded_layers)
        if output_all_encoded_layers:
            for layer in encoded_layers:
                all_encoder_layers.append(layer)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
            #return encoded_layers, pooled_output
        
        return sequence_output, pooled_output

class DKEncoder(nn.Module):
    """ A class for the Dynamic Knowledge Encoder for a `CokeBertModel`.
    """
    def __init__(self, k_v_dim, q_dim, no_layers):
        """Constructs a Dynamic KnowledgeEncoder

        Args:
            k_v_dim (int): size of the k and v vectors (internal Knowledge representation)
            q_dim (int): size of the q vector (Internal Text Representation)
            no_layers (int): Number of layers in the Dynamic Knowledge Encoder
        """
        super().__init__()

        layers = []
        for i in range(no_layers, 0, -1):
            layers.append(DKEncoder_layer(k_v_dim, q_dim, i))
        self.layers = nn.ModuleList(layers)

        self.tanh = nn.Tanh()

        self.k_v_dim = k_v_dim

    def forward(self, input_ent, q, k_v_s):
        """
        Forward pass through the Dynamic Knowledge Encoder

        Args:
            input_ent: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                with the entities embeddings
            q: the full sequence of hidden-states corresponding
                to the last text-attention block of shape [batch_size, sequence_length, hidden_size] 
            k_v_s: list of (k, v) tuples of length `no_layers`, 
                k, v are of shape [batch_size, sequence_length, k_v_dim]
        
        Returns:
            hidden_states_ent: internal entity representations: torch.Tensor of shape [batch_size, sequence_length, entity_size]
        """
        q = q[:,0,:] #CLS token only
        for i in range(len(self.layers)):
            k, v = k_v_s[-(i+1)]
            if i != 0:
                v = torch.cat([v, combined], -1)
            layer = self.layers[i]
            combined = layer(q, k, v)

        hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1], self.k_v_dim*2)#.cuda()
        ent_pos_s = torch.nonzero(input_ent) # id start from 0

        for batch in range(input_ent.shape[0]):
            for i,index in enumerate(ent_pos_s[ent_pos_s[:,0]==batch]):
                hidden_states_ent[batch][int(index[1])] = combined[batch][i]

        return hidden_states_ent

class DKEncoder_layer(nn.Module):
    """ A class for one Dynamic Knowledge Encoder Layer for a `DKEncoder` from CokeBert.
    """
    def __init__(self, k_v_dim, q_dim, layer_no):
        """ Constructs a `DKEncoder_layer`.
        
        Args:
            k_v_dim (int): size of the k and v vectors (internal Knowledge representation)
            q_dim (int): size of the q vector (Internal Text Representation)
            layer_no (int): Number of the layer (assigned in reverse order)
        """
        super().__init__()
        self.text = DK_text(k_v_dim, q_dim, layer_no)
        self.knowledge = DK_knowledge(k_v_dim)
        self.fusion = DK_fusion(k_v_dim, layer_no)

    def forward(self, q, k, v):
        """
        Forward pass through the Dynamic Knowledge Encoder layer

        Args:
            input_ent: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                with the entities embeddings
            q: the representation of hidden-states corresponding
                to the last text-attention block of shape [batch_size, 1, q_dim] 
            k: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]
            v: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]
        
        Returns:
            internal entity representations: torch.Tensor of shape [batch_size, sequence_length, entity_size]
        """
        q_i = self.text(q)
        k = self.knowledge(k)

        return self.fusion(q_i, k, v)

class DK_text(nn.Module):
    """ A class for the Text Processing in a `DKEncoderLayer` from CokeBert.
    """
    def __init__(self, k_v_dim, q_dim, layer_no):
        """ Constructs a `DK_text` module
        
        Args:
            k_v_dim (int): size of the k and v vectors (internal Knowledge representation)
            q_dim (int): size of the q vector (Internal Text Representation)
            layer_no (int): Number of the layer (assigned in reverse order)
        """
        super().__init__()
        self.q_linear = nn.Linear(q_dim, k_v_dim, bias=True)
        self.tanh = nn.Tanh()

        self.number = layer_no

    def forward(self, q):
        """ Forward pass through the Text Processing Module of a Dynamic Knowledge Encoder layer
        
        Args:
            q: the full sequence of hidden-states corresponding
                to the last text-attention block of shape [batch_size, sequence_length, hidden_size] 

        Returns:
            q_i: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]
        """
        q_i = self.q_linear(q)
        q_i = self.tanh(q_i)

        for i in range(1, self.number+2):
            q_i = q_i.unsqueeze(i)

        return q_i

class DK_knowledge(nn.Module):
    """ A class for the Knowledge Processing in a `DKEncoderLayer` from CokeBert."""
    def __init__(self, k_v_dim):
        """ Constructs a `DK_knowledge` module

        Args:
            k_v_dim (int): size of the k and v vectors (internal Knowledge representation)
        """
        super().__init__()
        self.k_v_linear = nn.Linear(k_v_dim, k_v_dim, bias=False)
    
    def forward(self, k):
        """ Forward pass through the Knowledge Processing Module of a Dynamic Knowledge Encoder layer
        
        Args:
            k: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]

        Returns:
            hidden knowledge representation: torch.Tensor of shape [batch_size, sequence_length, k_v_dim]
        """
        return self.k_v_linear(k)

class CokeBertForSequenceClassification(PreTrainedCokeBertModel):
    """
    CokeBert model for sequence classification.  
    This module is composed of the CokeBert model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        """Constructs a `CokeBertForSequenceClassification` model

        Args:
            `config`: a CokeBertConfig class instance with the configuration to build a new model.
            `num_labels`: the number of classes for the classifier. Default = 2.
        """
        super().__init__(config)

        self.num_labels = num_labels
        self.model = CokeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()

        self.classifier = nn.Linear(config.hidden_size*2, num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, k_v_s=None):
        """ Performs a Forward Pass through the model

        Args:
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
            `input_ent`: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                with the entities embeddings
            `ent_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]
            `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
                with indices selected in [0, ..., num_labels].
            k_v_s: list of (k_i, v_i) vectors as input for the dynamic knowledge encoder

        Returns:
            if `labels` is not `None`:  
                Outputs the CrossEntropy classification loss of the output with the labels.  
            if `labels` is `None`:  
                Outputs the classification logits of shape [batch_size, num_labels].  
        """
        seq_out, pooled_output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, input_ent=input_ent, ent_mask=ent_mask, output_all_encoded_layers=False, k_v_s=k_v_s)

        head = seq_out[input_ids==1601]
        tail = seq_out[input_ids==1089]
        try:
            pooled_output = torch.cat([head,tail], -1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
        except:
            print(input_ids)
            print("===")
            print(head.shape)
            print("===")
            print(tail.shape)
            raise ValueError()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            try:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            except:
                print(head.shape)
                print("---")
                print(tail.shape)
                print("---")
                print(pooled_output.shape)
                print("---")
                print(logits.shape)
                print("---")
                print(self.num_labels)
                print("---")
                print(labels.shape)
                raise ValueError()
            return loss
        else:
            return logits

class CokeBertForEntityTyping(PreTrainedCokeBertModel):
    """
    CokeBert model for classification.  
    This module is composed of the CokeBert model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        """Constructs a `CokeBertForEntityTyping` model

        Args:
            `config`: a CokeBertConfig class instance with the configuration to build a new model.
            `num_labels`: the number of classes for the classifier. Default = 2.
        """
        super().__init__(config)

        self.num_labels = num_labels
        self.model = CokeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.apply(init_weights)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, k_v_s=None):
        """ Performs a Forward Pass through the model

        Args:
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
            `input_ent`: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                with the entities embeddings
            `ent_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]
            `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
                with indices selected in [0, ..., num_labels].
            k_v_s: list of (k_i, v_i) vectors as input for the dynamic knowledge encoder

        Returns:
            if `labels` is not `None`:  
                Outputs the CrossEntropy classification loss of the output with the labels.  
            if `labels` is `None`:  
                Outputs the classification logits of shape [batch_size, num_labels].  
        """
        seq_out, pooled_output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, input_ent=input_ent, ent_mask=ent_mask, output_all_encoded_layers=False, k_v_s=k_v_s)

        pooled_output = seq_out[input_ids==1601]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.typing(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

class CokeBertForMaskedLM(PreTrainedCokeBertModel):
    """
    CokeBert model for masked pre-training task.  
    This module is composed of the CokeBert model with a linear layer on top of
    the sequence output.
    """
    def __init__(self, config):
        """Constructs a `CokeBertForMaskedLM` model

        Args:
            `config`: a CokeBertConfig class instance with the configuration to build a new model.
        """
        super().__init__(config)
        self.model = CokeBertModel(config)
        self.cls = BertOnlyMLMHead(config, self.model.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self, input_ids, input_ents, ent_mask=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None, k_v_s=None):
        """ Performs a Forward Pass through the model

        Args:
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
            `input_ent`: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                with the entities embeddings
            `ent_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]
            `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
                with indices selected in [0, ..., num_labels].
            k_v_s: list of (k_i, v_i) vectors as input for the dynamic knowledge encoder

        Returns:
            if `masked_lm_labels` is not `None`:  
                Outputs the CrossEntropy classification loss of the output with the labels.  
            if `masked_lm_labels` is `None`:  
                Outputs the classification logits of shape [batch_size, num_labels].  
        """
        sequence_output, _ = self.model(input_ids, token_type_ids, attention_mask, input_ents, ent_mask,
                                       output_all_encoded_layers=False, k_v_s=k_v_s)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores