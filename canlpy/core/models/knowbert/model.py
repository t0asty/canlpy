from typing import Dict, List

import math
import torch
import torch.nn as nn
import numpy as np
import yaml
import tarfile
import os

from canlpy.core.util.util import get_dtype_for_module, extend_attention_mask_for_bert, find_value
from canlpy.core.components.fusion.knowbert_fusion import SolderedKG
from canlpy.core.util.file_utils import cached_path
from canlpy.core.util.knowbert_tokenizer.vocabulary import Vocabulary

from transformers import BertConfig, BertForPreTraining

class KnowBert(nn.Module):
    """
    KnowBert
    Combines BERT with one or more SolderedKG

    each SolderedKG is inserted at a particular level, given by an index,
    such that we run Bert to the index, then the SolderedKG, then the rest
    of bert.  Indices such that index 0 means run the first contextual layer,
    then add KG, and index 11 means run to the top of BERT, then the KG
    (for bert base with 12 layers).


    Parameters:
        soldered_kgs: a dictionnary of named SolderedKG
        soldered_layers: a dictionnary indicating at which index the SolderedKG is inserted
        bert_model_name: the name of the bert model, eg: `bert_base_uncased`.
        mode: `None` or `entity_linking`, `entity_linking` then only unfreeze the SolderedKB layer, 
            else, unfreeze the entire model.
        state_dict_file: a file containing the state dictionnary.
        strict_load_archive: whether to allow unmapped weights to the loaded model.
        remap_segment_embeddings: determines how many segment embeddings BERT can have (if None, the default of 2 is kept).
        state_dict_map: a dictionnary used to remap embedding weights.
    
    """
    def __init__(self,
                 soldered_kgs: Dict[str, SolderedKG],
                 soldered_layers: Dict[str, int],
                 bert_model_name: str,
                 mode: str = None,
                 state_dict_file: str = None,
                 strict_load_archive: bool = True,
                 remap_segment_embeddings: int = None,
                 state_dict_map:Dict[str,str] = None):

        super().__init__()

        self.remap_segment_embeddings = remap_segment_embeddings

        # get the LM + NSP parameters from BERT
        pretrained_bert = BertForPreTraining.from_pretrained(bert_model_name)
        self.pretrained_bert = pretrained_bert
        self.pretraining_heads = pretrained_bert.cls
        self.pooler = pretrained_bert.bert.pooler

        #NOTE: add the soldered layers as layer of this module
        self.soldered_kgs = soldered_kgs
        for key, skg in soldered_kgs.items():
            self.add_module(key + "_soldered_kg", skg)

        # list of (layer_number, soldered key) sorted in ascending order
        #eg: [(9, 'wordnet')]
        self.layer_to_soldered_kg = sorted(
                [(layer, key) for key, layer in soldered_layers.items()]
        )

        # the last layer
        # eg: 12
        num_bert_layers = len(self.pretrained_bert.bert.encoder.layer)

        # the first element of the list is the index
        #eg:[(9, 'wordnet'), [11, None]]
        self.layer_to_soldered_kg.append([num_bert_layers - 1, None])

        #Load the model's weights
        if state_dict_file is not None:
            if(torch.cuda.is_available()):
                state_dict = torch.load(state_dict_file)
            else:
                state_dict = torch.load(state_dict_file,map_location='cpu')
            
            #Does remapping
            if(state_dict_map is not None):
                state_dict = state_dict.copy()
                metadata = getattr(state_dict, '_metadata', None)
                if metadata is not None:
                    state_dict._metadata = metadata
                for old_key,new_key in state_dict_map.items():
                    state_dict[new_key] = state_dict.pop(old_key)
                        
            self.load_state_dict(state_dict, strict=strict_load_archive)

        #Token type embeddigns in bert <=> segment embeddings => originally: to which out of 2 sentence the token belongs to
        #Remapping allows to have more than 2 segment embeddings type
        if remap_segment_embeddings is not None:
            # will redefine the segment embeddings
            new_embeddings = self._remap_embeddings(self.pretrained_bert.bert.embeddings.token_type_embeddings.weight)
            if new_embeddings is not None:
                del self.pretrained_bert.bert.embeddings.token_type_embeddings
                self.pretrained_bert.bert.embeddings.token_type_embeddings = new_embeddings

        #entity_linking mode indicates that we are only training the entity linker and freezing the other parameters
        assert mode in (None, 'entity_linking')
        self.mode = mode
        #if mode = entity_linking,freeze all and then only unfreeze the SolderedKB layer, else, unfreeze the entire model
        self.unfreeze()
    
    @classmethod
    def from_pretrained(cls,file_or_url,strict_load_archive=True):
        """
            Loads a pretrained knowbert model from an archive file
            Args:
                file_or_url: a file path or url to the archive
                strict_load_archive: whether to allow unmapped weights to the loaded model.
        """
        model_file = cached_path(file_or_url)
        tempdir = model_file+"_extracted"
        if(not os.path.isdir(tempdir)):
            os.makedirs(tempdir,exist_ok=True)

            print(f"Extracting model archive in {tempdir}")
            with tarfile.open(model_file, 'r:gz') as archive:
                archive.extractall(tempdir)

        weights_file = tempdir+'/weights.th'
        config_file = tempdir+'/config.json'

        #Use yaml to open json due to trailing comma
        with open(config_file,'r') as f:
            config = yaml.safe_load(f)

        vocabulary_path = find_value(config,"vocabulary")["directory_path"]
        entity_vocabulary = Vocabulary.from_files(vocabulary_path)
        model_config = config["model"]

        soldered_kgs = {}
        for soldered_kg_name,soldered_kg_config in find_value(model_config,"soldered_kgs").items():
            soldered_kgs[soldered_kg_name] = SolderedKG.from_config(soldered_kg_config,entity_vocabulary)

        state_dict_remap = {}

        for soldered_kg_name in soldered_kgs.keys():
            before_key = f"{soldered_kg_name}_soldered_kg.entity_linker.disambiguator.span_extractor._global_attention._module.weight"
            after_key = before_key.replace("._module","")
            state_dict_remap[before_key]=after_key

            before_key = f"{soldered_kg_name}_soldered_kg.entity_linker.disambiguator.span_extractor._global_attention._module.bias"
            after_key = before_key.replace("._module","")
            state_dict_remap[before_key]=after_key

        layer_indices = find_value(model_config,"soldered_layers")

        model = cls(soldered_kgs = soldered_kgs,
                                        soldered_layers = layer_indices,
                                        bert_model_name = model_config["bert_model_name"],
                                        state_dict_file=weights_file,
                                        strict_load_archive=strict_load_archive,
                                        state_dict_map = state_dict_remap)
        
        return model

    def _remap_embeddings(self, token_type_embeddings):
        embed_dim = token_type_embeddings.shape[1]
        if list(token_type_embeddings.shape) == [self.remap_segment_embeddings, embed_dim]:
            # already remapped!
            return None
        new_embeddings = torch.nn.Embedding(self.remap_segment_embeddings, embed_dim)
        new_embeddings.weight.data.copy_(token_type_embeddings.data[0, :])
        return new_embeddings

    def load_state_dict(self, state_dict, strict=True):
        """
        Initialize the model's weights with the provided state_dict

        Args:
            state_dict: the PyTorch state dict.
            strict: whether to allow unmapped weights to the loaded model.
        """
        if self.remap_segment_embeddings:
            # hack the embeddings!
            new_embeddings = self._remap_embeddings(state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'])
            if new_embeddings is not None:
                state_dict['pretrained_bert.bert.embeddings.token_type_embeddings.weight'] = new_embeddings.weight
        super().load_state_dict(state_dict, strict=strict)

    def unfreeze(self):
        """
        Unfreezes the weights depending of `self.mode`, if `entity_linking`only unfreezes SolderedKGs
        else, unfreezes all weights
        """
        if self.mode == 'entity_linking':
            # all parameters in BERT are fixed, just training the linker
            # linker specific params set below when calling soldered_kg.unfreeze
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for p in self.parameters():
                p.requires_grad_(True)

        for key in self.soldered_kgs.keys():
            module = getattr(self, key + "_soldered_kg")
            module.unfreeze(self.mode)

    def forward(self, tokens=None, segment_ids=None, candidates=None, **kwargs):

        """
        
        Args: 
            tokens['tokens']: a torch.LongTensor of shape , shape: (batch_size, max_seq_len), tokens indices (used to index an embedding).
            Because a batch contains multiple sentences with varying # of tokens, all tokens tensors are padded with zeros.

            segment_ids: a torch.LongTensor of shape (batch_size,max_seq_len) indicating the segments_ids for each token (0 for first segment and 1 for second)

            candidates, for each SolderedKB contains:
            Example:
                candidates['wordnet']['candidate_entity_priors']: torch.FloatTensor of shape (batch_size, max # detected entities, max # KB candidate entities)
                Correctness probabilities estimated by the entity extractor (sums to 1 (or 0 if padding) on axis 2)
                Adds 0 padding to axis 1 when there is less detected entities in the sentence than in the max sentence
                Adds 0 padding to axis 2 when there is less detected KB entities for an entity in the sentence than in the max candidate KB entities entity

                candidates['wordnet']['ids']: torch.LongTensor of shape (batch_size, max # detected entities, max # KB candidate entities)
                ids of the KB candidate entities + 0 padding on axis 1 or 2 if necessary.

                candidates['wordnet']['candidate_spans']: torch.LongTensorshape of shape (batch_size, max # detected entities, 2)
                Spans of which sequence of tokens correspond to an entity in the sentence, eg: [1,2] for Michael Jackson (both bounds are included)
                Padding with [-1,-1] when no more detected entities

                candidates['wordnet']['candidate_segment_ids']: a torch.LongTensorshape of shape (batch_size, max # detected entities)
                indicates the segments_ids for each entity
        
        kwargs:
            gold_entities: id of the correct corresponding entity, used by the entity linker to train

        """

        assert candidates.keys() == self.soldered_kgs.keys()

        #Mask correspond to token = -1
        mask = tokens['tokens'] > 0
        #0 for non masked tokens and -10000.0 for masked tokens
        attention_mask = extend_attention_mask_for_bert(mask, get_dtype_for_module(self))

        #Token embeddings extracted from their indices
        contextual_embeddings = self.pretrained_bert.bert.embeddings(tokens['tokens'], segment_ids)

        output = {}
        start_layer_index = 0
        loss = 0.0

        #dictionnary that for each soldered kg layer contains a list of the ids of the correct entity in text => can be usd to train the entity linker
        gold_entities = kwargs.pop('gold_entities', None)

        for layer_num, soldered_kg_key in self.layer_to_soldered_kg:
            end_layer_index = layer_num + 1
            if end_layer_index > start_layer_index:
                # run bert layer in between previous and current SolderedKG 
                for layer in self.pretrained_bert.bert.encoder.layer[
                                start_layer_index:end_layer_index]:
                    #Only take first index due to the new transformers BertLayer that outputs a tuple
                    contextual_embeddings = layer(hidden_states = contextual_embeddings, attention_mask = attention_mask)[0]
            start_layer_index = end_layer_index

            # run the SolderedKG component
            if soldered_kg_key is not None:
                #Get soldered_kg module
                soldered_kg = getattr(self, soldered_kg_key + "_soldered_kg")
                #Gives for this soldered kg, the span, prior, ids and the segment_ids of the detected entities
                soldered_kwargs = candidates[soldered_kg_key]
                soldered_kwargs.update(kwargs)
                if gold_entities is not None and soldered_kg_key in gold_entities:
                    soldered_kwargs['gold_entities'] = gold_entities[soldered_kg_key]
                kg_output = soldered_kg(
                        contextual_embeddings=contextual_embeddings,
                        tokens_mask=mask,
                        **soldered_kwargs)

                #Add the soldered KG loss (entity linker loss) to the sum of loss of other soldered KG
                if 'loss' in kg_output:
                    loss = loss + kg_output['loss']
                    output[soldered_kg_key+'loss'] = kg_output['loss']

                contextual_embeddings = kg_output['contextual_embeddings']

                #Add the output of the soldered kg to the output of KnowBert
                output[soldered_kg_key] = {}
                for key in kg_output.keys():
                    if key != 'loss' and key != 'contextual_embeddings':
                        output[soldered_kg_key][key] = kg_output[key]

        # get the pooled CLS output
        pooled_output = self.pooler(contextual_embeddings)

        output['loss'] = loss
        output['pooled_output'] = pooled_output
        output['contextual_embeddings'] = contextual_embeddings

        return output