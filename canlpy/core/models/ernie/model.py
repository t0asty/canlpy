import torch
import typing
from typing import List,Dict,Tuple
import os
import json
import math
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import logging

import canlpy.core.models.bert.model as bert
from canlpy.core.models.bert.model import BertEmbeddings, BertPooler, LayerNorm
from canlpy.core.models.ernie.components import ErnieLayer, ErnieLayerMix,ErnieEncoder
from canlpy.core.components.heads import BertOnlyMLMHead,BertOnlyNSPHead,ErniePreTrainingHeads

logger = logging.getLogger(__name__)

CONFIG_NAME = 'ernie_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
MAPPING_FILE = 'mapping.json'

class ErnieConfig():
    """
    Configuration class to store the configuration of an `ErnieModel`.
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
        """
        Constructs ErnieConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `ErnieModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            entity_size= Size of the entity embeddings,
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder for the tokens.
            num_attention_heads_ent: Number of attention heads for each attention layer in
                the Transformer encoder for the entities.
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
            layer_types: list() of ErnieEncoders which can be 'sim' (Bert encoder), 
            'mix' (Ernie encoder but no multihead attention for entities) or 'norm' (standard Ernie encoder)
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
        """Loads and returns a config class from a json file located at path"""
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
    
class PreTrainedErnieModel(nn.Module):
    """ 
    An abstract class to handle weights initialization and
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

    def init_weights(self):
        """ 
        Initialize the weights.
        """
        bert.init_weights(self, self.config.initializer_range)

    @classmethod
    def from_pretrained(cls, dir_path, state_dict=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedErnieModel from a pre-trained model file or a pytorch state dict.

        Args:
            dir_path:
                - a path to a pretrained model archive containing:
                    . `ernie_config.json`: a configuration file for the model
                    . `pytorch_model.bin`: a PyTorch dump of a ErnieForPreTraining instance
                    .  `mapping.json`: an Optional file to remap the weights from the pre-trained weights to this implementation 
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of the PyTorch dump
            *inputs, **kwargs: additional input for the specific Ernie class
                (ex: num_labels for ErnieForSequenceClassification)
        Returns:
            The loaded pretrained model
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

        return model

class ErnieModel(PreTrainedErnieModel):
    """
    Ernie model

    Params:
        config: an ErnieConfig class instance with the configuration to build a new model

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
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Returns: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
             classifier pretrained on top of the hidden state associated to the first character of the
             input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    input_ent: shape(1,6,100)
    ent_mask: torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.ErnieConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.ErnieModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask, input_ent, ent_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config.vocab_size,config.hidden_size, config.max_position_embeddings,config.hidden_dropout_prob,config.type_vocab_size)
        self.encoder = ErnieEncoder(config)
        self.pooler = BertPooler(config.hidden_size)
        self.init_weights()

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

#Specific ERNIE models

class ErnieForMaskedLM(PreTrainedErnieModel):
    """
    Ernie model with the masked language modeling head.

    This module comprises the Ernie model followed by the masked language modeling head.

    Params:
        config: a ErnieConfig class instance with the configuration to build a new model.

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
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Returns:
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
    input_ent: shape(1,6,100)
    ent_mask: torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ErnieConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = ErnieForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask,input_ent,ent_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        self.cls = BertOnlyMLMHead(config, self.model.embeddings.word_embeddings.weight)
        #Recursively initalize all the weights
        self.init_weights()

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

class ErnieForPreTraining(PreTrainedErnieModel):
    """
    Ernie model with pre-training heads.
    This module comprises the Ernie model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: an ErnieConfig class instance with the configuration to build a new model.

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
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Returns:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        self.cls = ErniePreTrainingHeads(config, self.model.embeddings.word_embeddings.weight)
        self.init_weights()

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

class ErnieForNextSentencePrediction(PreTrainedErnieModel):
    """
    Ernie model with next sentence prediction head.
    This module comprises the Ernie model followed by the next sentence classification head.

    Params:
        config: a ErnieConfig class instance with the configuration to build a new model.

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
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Returns:
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

    config = ErnieConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = ErnieForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.init_weights()

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

class ErnieForEntityTyping(PreTrainedErnieModel):

    """
    Ernie model with entity typing prediction head.
    This module comprises the Ernie model followed by the entity typing head.

    Params:
        config: a ErnieConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

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
    """

    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, num_labels, False)
        self.init_weights()

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

class ErnieForSTSB(PreTrainedErnieModel):
    """
    Ernie model with STSB prediction head (predict sentence similarity).
    This module comprises the Ernie model followed by the STSB head.

    Params:
        config: a ErnieConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

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
    """
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = 2
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

        self.m = torch.nn.LogSoftmax(-1)
        self.mm = torch.nn.Softmax(-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        _, pooled_output = self.model(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        probs = self.m(logits)

        if labels is not None:
            per_example_loss = -torch.sum(labels * probs, -1)
            loss = torch.mean(per_example_loss)
            return loss
        else:
            return self.mm(logits)

class ErnieForSequenceClassification(PreTrainedErnieModel):
    """
    Ernie model for classification.
    This module is composed of the Ernie model with a linear layer on top of
    the pooled output.

    Params:
        `config`: an ErnieConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

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

    Returns:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

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

class ErnieForNQ(PreTrainedErnieModel):
    """
    Ernie model for NQ.
    This module is composed of the Ernie model with a linear layer on top of
    the pooled output.

    Params:
        `config`: an ErnieConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

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
        `choice_mask`:
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Returns:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """

    def __init__(self, config, num_choices=2):
        super().__init__(config)
        self.num_choices = num_choices
        self.model = ErnieModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, choice_mask=None, labels=None):
            
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_input_ent = input_ent.view(-1, input_ent.size(-2), input_ent.size(-1))
        flat_ent_mask = ent_mask.view(-1, ent_mask.size(-1))
        _, pooled_output = self.model(flat_input_ids, flat_token_type_ids, flat_attention_mask, flat_input_ent, flat_ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        null_score = torch.zeros([labels.shape[0],1]).cuda()
        reshaped_logits = torch.cat([null_score, reshaped_logits], -1) + choice_mask

        if labels is not None:
            weight = torch.FloatTensor([0.3]+[1]*16).cuda()
            loss_fct = CrossEntropyLoss(weight)
            loss = loss_fct(reshaped_logits, labels+1)
            return loss
        else:
            return reshaped_logits

class ErnieForQuestionAnswering(PreTrainedErnieModel):
    """
    Ernie model for Question Answering (span extraction).
    This module is composed of the Ernie model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a ErnieConfig class instance with the configuration to build a new model

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
        `input_ent`: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
            with the entities embeddings
        `ent_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]
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
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()

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

