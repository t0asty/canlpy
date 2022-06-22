# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import torch
import torch.nn as nn
from canlpy.core.models.bert.model import LayerNorm
from canlpy.core.components.activation_functions import get_activation_function

class BertLMPredictionHead(nn.Module):
    """ Bert Head for using Bert as a predictive Language Model 
    """
    def __init__(self, config, bert_model_embedding_weights):
        """ constructs a `BertLMPredictionHead`
        
        Args:
            config: a `BertConfig``
            `bert_model_embedding_weights`:  torch.Tensor of size [hidden_size, vocab_size], the weights of the input embeddings of bert
        """
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
        """Forward pass through the module
        
        Returns:
            hidden_states: torch.Tensor of size [batch_size, seq_length, vocab_size], output representations
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    """ Bert Head for the Masked Language Modeling Pre-training task of Bert
    """
    def __init__(self, config, bert_model_embedding_weights):
        """ constructs a `BertOnlyMLMHead`
        
        Args:
            config: a `BertConfig`
            `bert_model_embedding_weights`:  torch.Tensor of size [hidden_size, vocab_size], the weights of the input embeddings of bert
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        """Forward pass through the module
        
        Returns:
            hidden_states: torch.Tensor of size [batch_size, seq_length, vocab_size], output representations
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertPredictionHeadTransform(nn.Module):
    """ Bert Head for the Masked Language Modeling Pre-training task of Bert
    """
    def __init__(self, config):
        """Constructs a `BertPredictionHeadTransform`
        
        Args:
            config: a `BertConfig`
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation_function(config.hidden_act) if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """ Forward pass through the module
        
        Returns:
            hidden_states: a torch.Tensor of size [batch_size, seq_length, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertOnlyNSPHead(nn.Module):
    """ Bert Head for the Next Sentence Prediction task of Bert
    """
    def __init__(self, config):
        """ constructs a `BertOnlyNSPHead`
        
        Args:
            config: a `BertConfig`
        """
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        """ Forward pass through the module
        
        Returns:
            seq_relationship_score: a torch.Tensor of size [batch_size, 2], score for prediction of next sentence
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class ErnieEntPredictionHead(nn.Module):
    """ Ernie Head for predicting entities
    """
    def __init__(self, config):
        """ constructs a `ErnieEntPredictionHead`
        
        Args:
            config: a `BertConfig`
        """
        super().__init__()
        config_ent = copy.deepcopy(config)
        #Replace the hidden_size by the entity size for the Bert head to have the correct dimension
        config_ent.hidden_size = config.ent.entity_size
        self.transform = BertPredictionHeadTransform(config_ent)

    def forward(self, hidden_states, candidate):
        """ Forward pass through the module
        
        Returns:
            seq_relationship_score: a torch.Tensor of size [batch_size, seq_length, no_of_entity_in_batch], predicted entities
        """
        hidden_states = self.transform(hidden_states)
        candidate = torch.squeeze(candidate, 0)
        # hidden_states [batch_size, max_seq, dim]
        # candidate [entity_num_in_the_batch, dim]
        # return [batch_size, max_seq, entity_num_in_the_batch]
        return torch.matmul(hidden_states, candidate.t())

class ErniePreTrainingHeads(nn.Module):
    """Heads for Ernie Pretraining

    """
    def __init__(self, config, bert_model_embedding_weights):
        """ constructs `ErniePreTrainingHeads`
        
        Args:
            config: a `BertConfig`
            `bert_model_embedding_weights`:  torch.Tensor of size [hidden_size, vocab_size], the weights of the input embeddings of bert/ernie
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.predictions_ent = BertEntPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, candidate):
        """forward pass through the module
        
        Returns:
            prediction_scores: token prediction score 
            seq_relationship_score: next sentence prediction score
            prediction_scores_ent: entity prediction score
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        prediction_scores_ent = self.predictions_ent(sequence_output, candidate)
        return prediction_scores, seq_relationship_score, prediction_scores_ent