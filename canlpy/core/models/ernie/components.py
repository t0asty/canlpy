import copy
from torch import nn
from canlpy.core.models.bert.model import BertLayer, DenseSkipLayer, BertAttention
from canlpy.core.components.fusion import ErnieFusion

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

class ErnieEncoder(nn.Module):
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
        ent_mask = ent_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)

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