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
from canlpy.core.components.fusion.cokebert_fusion import DK_fusion

logger = logging.getLogger(__name__)

CONFIG_NAME = 'cokebert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
MAPPING_FILE = 'mapping.json'

class CokeBertConfig(object):
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
        self.k_v_dim = k_v_dim
        self.q_dim = q_dim
        self.dk_layers = dk_layers
       
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

    def to_text_encoder_config(self):
        params = self.to_dict()
        params['layer_types'] = [x for x in params['layer_types'] if x == 'sim']
        params['num_hidden_layers'] = len(params['layer_types'])
        return CokeBertConfig(**params)

    def to_knowl_encoder_config(self):
        params = self.to_dict()
        params['layer_types'] = [x for x in params['layer_types'] if x != 'sim']
        params['num_hidden_layers'] = len(params['layer_types'])
        return CokeBertConfig(**params)


class PreTrainedCokeBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
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
        
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # elif isinstance(module, LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

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
    def __init__(self, config):
        super(CokeBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config.vocab_size, config.hidden_size, config.max_position_embeddings, config.hidden_dropout_prob, config.type_vocab_size)
        self.text_encoder = ErnieEncoder(config.to_text_encoder_config())
        self.knowledge_encoder = ErnieEncoder(config.to_knowl_encoder_config())
        self.pooler = BertPooler(config.hidden_size)
        self.dk_encoder = DKEncoder(config.k_v_dim, config.q_dim, config.dk_layers)

        self.k_v_dim = config.k_v_dim
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True, k_v_s=None):
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
            # TODO: get shape from config
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1],200).cuda()#.half()
        else:

            hidden_states_ent = self.dk_encoder(input_ent, hidden_states, k_v_s) #[(k_1, v_1), (k_2, v_2)]
            #hidden_states_ent = self.word_graph_attention(input_ent, hidden_states[:,0,:], k, v, "entity")

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
    def __init__(self, k_v_dim, q_dim, no_layers):
        super(DKEncoder, self).__init__()

        #self.k_v_linear_1 = nn.Linear(k_v_dim, k_v_dim, bias=False)
        #self.k_v_linear_2 = nn.Linear(k_v_dim, k_v_dim, bias=False)
        #self.v_linear_1 = nn.Linear(k_v_dim, k_v_dim, bias=False)
        #self.v_linear_2 = nn.Linear(k_v_dim, k_v_dim, bias=False)
        #self.q_linear_1 = nn.Linear(q_dim, k_v_dim, bias=True)
        #self.q_linear_2 = nn.Linear(q_dim, k_v_dim, bias=True)
        layers = []
        for i in range(no_layers, 0, -1):
            layers.append(DKEncoder_layer(k_v_dim, q_dim, i))
        self.layers = nn.ModuleList(layers)

        self.tanh = nn.Tanh()

        self.k_v_dim = k_v_dim

    """
    def self_attention(self, q, k_1, v_1, k_2, v_2):
        q_2 = self.q_linear_2(q)
        q_2 = self.tanh(q_2)

        q_2 = q_2.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        k = self.k_v_linear_2(k_2)

        attention = ((q_2 * k).sum(4)).div(math.sqrt(self.k_v_dim))

        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_3(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to


        attention = attention.unsqueeze(3)

        sentence_entity_reps = attention.matmul(v_2).squeeze(3)

        v_1 = torch.cat([v_1, sentence_entity_reps],-1)

        q_1 = self.q_linear_1(q)
        q_1 = self.Tanh(q_1)

        q_1 = q_1.unsqueeze(1).unsqueeze(2)


        k = self.k_v_linear_1(k_1)

        attention = ((q_1*k).sum(3)).div(math.sqrt(self.k_v_dim))


        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_2(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to
        attention = attention.unsqueeze(2)

        sentence_entity_reps = attention.matmul(v_1).squeeze(2)

        return sentence_entity_reps
    """


    def forward(self, input_ent, q, k_v_s):#k_1, v_1, k_2, v_2):
        q = q[:,0,:] #all input: 0, !=0
        #combined = self.self_attention(q, k_1, v_1, k_2, v_2)
        for i in range(len(self.layers)):
            k, v = k_v_s[-(i+1)]
            if i != 0:
                v = torch.cat([v, combined], -1)
            layer = self.layers[i]
            combined = layer(q, k, v)

        hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1], self.k_v_dim*2).cuda()
        ent_pos_s = torch.nonzero(input_ent) # id start from 0

        for batch in range(input_ent.shape[0]):
            for i,index in enumerate(ent_pos_s[ent_pos_s[:,0]==batch]):
                hidden_states_ent[batch][int(index[1])] = combined[batch][i]

        return hidden_states_ent


class DKEncoder_layer(nn.Module):
    def __init__(self, k_v_dim, q_dim, layer_no):
        """
        self.number = layer_no
        self.k_v_dim = k_v_dim

        self.k_v_linear = nn.Linear(k_v_dim, k_v_dim, bias=False)
        self.v_linear = nn.Linear(k_v_dim, k_v_dim, bias=False)
        self.q_linear = nn.Linear(q_dim, k_v_dim, bias=True)

        self.softmax = nn.Softmax(dim=layer_no+1)
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        """
        super().__init__()
        self.text = DK_text(k_v_dim, q_dim, layer_no)
        self.knowledge = DK_knowledge(k_v_dim)
        self.fusion = DK_fusion(k_v_dim, layer_no)

    """
    def forward(self, q, k, v):
        q_i = self.q_linear(q)
        q_i = self.tanh(q_i)

        for i in range(1, self.number+2):
            q_i = q_i.unsqueeze(i)

        k = self.k_v_linear(k)

        attention = ((q_i * k).sum(self.number+2)).div(math.sqrt(self.k_v_dim))

        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to


        attention = attention.unsqueeze(self.number+1)

        sentence_entity_reps = attention.matmul(v).squeeze(self.number+1)

        return sentence_entity_reps
    """
    def forward(self, q, k, v):
        q_i = self.text(q)
        k = self.knowledge(k)

        return self.fusion(q_i, k, v)


class DK_text(nn.Module):
    def __init__(self, k_v_dim, q_dim, layer_no):
        super().__init__()
        self.q_linear = nn.Linear(q_dim, k_v_dim, bias=True)
        self.tanh = nn.Tanh()

        self.number = layer_no

    def forward(self, q):
        q_i = self.q_linear(q)
        q_i = self.tanh(q_i)

        for i in range(1, self.number+2):
            q_i = q_i.unsqueeze(i)

        return q_i

class DK_knowledge(nn.Module):
    def __init__(self, k_v_dim):
        super().__init__()
        self.k_v_linear = nn.Linear(k_v_dim, k_v_dim, bias=False)
    
    def forward(self, k):
        return self.k_v_linear(k)

class CokeBertForSequenceClassification(PreTrainedCokeBertModel):
    def __init__(self, config, num_labels=2):
        super().__init__(config)

        self.num_labels = num_labels
        self.model = CokeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()

        self.classifier = nn.Linear(config.hidden_size*2, num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, k_v_s=None):
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
            exit()

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
                exit()
            return loss
        else:
            return logits

class CokeBertForEntityTyping(PreTrainedCokeBertModel):
    def __init__(self, config, num_labels=2):
        super().__init__(config)

        self.num_labels = num_labels
        self.model = CokeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.apply(init_weights)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, k_v_s=None):
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