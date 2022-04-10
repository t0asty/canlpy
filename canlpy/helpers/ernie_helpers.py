from canlpy.core.components.tokenization import BertTokenizer
from canlpy.helpers.tokens import TAGME_TOKEN
from typing import List,Dict,Tuple
import re 
import torch
from torch import FloatTensor, Tensor
from torch.nn import Embedding


import tagme
# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = TAGME_TOKEN

NAME_TO_QID = "../kg_embed/entity_map.txt"
QID_TO_EID = "../kg_embed/entity2id.txt"
EID_TO_VEC = "../kg_embed/entity2vec.vec"

def create_model_mapping_ERNIE(model_old)->Dict[str,str]:
    '''Returns a dictionnary mapping the original ERNIE model weights to custom ERNIE model weights'''
    model_mapping = dict()
    for old_name, param in model_old.named_parameters():
        match = re.search(r'\d+', old_name)
        new_name = old_name.replace('bert','model')
        if(match!=None):#Layer
            layer=int(match.group())
        
            #BERT Layers
            if(layer<5):
                new_name = new_name.replace('.attention.self.','.attention.multi_head_attention.')
                new_name = new_name.replace('.attention.output.','.attention.skip_layer.')
                new_name = new_name.replace('.intermediate.dense.','.dense_intermediate.')
                new_name = new_name.replace(f'.{layer}.output.',f'.{layer}.skip_layer_out.')

            #ONLY for layer>=5, ERNIE Layers
            if(layer>=5):
                new_name = new_name.replace('.attention.self.','.attention_tokens.multi_head_attention.')
                new_name = new_name.replace('.attention.self_ent.','.attention_ent.multi_head_attention.')
                new_name = new_name.replace('.attention.output.','.attention_tokens.skip_layer.')
                new_name = new_name.replace('.attention.output_ent.','.attention_ent.skip_layer.')

                new_name = new_name.replace('.intermediate.dense.','.fusion.dense_intermediate_tokens.')
                new_name = new_name.replace('.intermediate.dense_ent.','.fusion.dense_intermediate_ent.')

                new_name = new_name.replace('.output.dense.','.fusion.skip_layer_tokens.dense.')
                new_name = new_name.replace('.output.dense_ent.','.fusion.skip_layer_ent.dense.')
                new_name = new_name.replace('.output.LayerNorm.','.fusion.skip_layer_tokens.LayerNorm.')
                new_name = new_name.replace('.output.LayerNorm_ent.','.fusion.skip_layer_ent.LayerNorm.')
                
                
            model_mapping[old_name] = new_name

        else:
            model_mapping[old_name] = new_name

    return model_mapping

def get_ents(text:str,name_to_QID:Dict[str,str],keep_proba:float = 0.3)->List[Tuple[str,int,int,float]]:
    '''Returns a list of [entity_QID,character_idx_start,character_idx_end,entity_score] for each entity detect in the text
    Input:
        text: string, the text to extract entities from
        name_to_QID: a dictionary mapping entity names to their QID'''
    annotations = tagme.annotate(text)
    entities = []
    # Keep annotations with a score higher than 0.3
    for ann in annotations.get_annotations(keep_proba):
        if ann.entity_title in name_to_QID:
            entities.append([name_to_QID[ann.entity_title], ann.begin, ann.end, ann.score])
        
    return entities

def load_name_to_QID(filename:str)->Dict[str,str]:
    '''Loads the dictionnary mapping entity names to their QID'''
    name_to_QID = {}
    with open(filename,'r') as f:
        for line in f:
            name, qid = line.strip().split("\t")
            name_to_QID[name] = qid

    return name_to_QID

def load_QID_to_eid(filename:str)->Dict[str,int]:
    '''Loads the dictionnary mapping entity QID to their eid (idx in an embedding vector)'''
    QID_to_eid = {}
    with open(filename,'r') as f:
        f.readline()
        for line in f:
            qid, eid = line.strip().split('\t')
            QID_to_eid[qid] = int(eid)
    return QID_to_eid

def load_eid_to_vec(filename:str)->FloatTensor:
    '''Loads the pytorch embedding mapping entity idx to their pre-trained vector representation'''
    vecs = []
    vecs.append([0]*100)
    with open(filename, 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    return embed

def concatenate_tokens_entities(tokens_a:List[str],tokens_b:List[str],entities_a:List[str],entities_b:List[str]) -> Tuple[List[str],List[str],List[int],List[int]]:
    '''Returns the merged tokens and entities as well as a list representing their segments_ids and the input mask
        Input:
            tokens_a: a list of tokens
            tokens_b: a list of tokens
            entities_a: a list of entities QID 
            entities_b: a list of entities QID 
    '''
    tokens =  ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    segments_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
    #UNK for CLS and SEP 
    entities = ["UNK"] + entities_a + ["UNK"] + entities_b + ["UNK"]
    input_mask = [1] * len(tokens) #Assume we consider the whole input
    return tokens,entities,segments_ids,input_mask

def get_entities_embeddings_and_mask(entities:List[str],QID_to_eid:Dict[str,int], eid_to_embeddings: Embedding,device:str)->Tuple[Tensor,Tensor]:
    '''Returns the merged tokens and entities as well as a list representing their segments_ids and the input mask
        Input:
            entities: a list of tokens
            QID_to_eid: a list of tokens
            eid_to_embeddings: a list of entities QID 
    '''
    #Convert QID to entity indices
    indexed_entities = []
    entities_mask = []
    for qid in entities:
        if qid != "UNK" and qid in QID_to_eid:
            indexed_entities.append(QID_to_eid[qid])
            entities_mask.append(1)
        else:
            indexed_entities.append(-1)
            entities_mask.append(0)
    entities_mask[0] = 1

    eid_tensor = torch.tensor([indexed_entities],device=device)
    entities_tensor = eid_to_embeddings(eid_tensor+1)
    entities_mask = torch.tensor([entities_mask],device=device)

    return entities_tensor,entities_mask

def process_sentences(text_a:str,text_b:str,masked_indices:List[int],name_to_QID:Dict[str,str],QID_to_eid:Dict[str,int],eid_to_embeddings:Embedding,tokenizer:BertTokenizer,device:str) -> Tuple[Tensor,Tensor,Tensor,Tensor]:

    # Tokenized input
    #entities_b:[['Q191037', 0, 10, 0.8473327159881592], ['Q2629392', 17, 26, 0.48991236090660095]]
    entities_a = get_ents(text_a,name_to_QID)
    entities_b = get_ents(text_b,name_to_QID)

    #tokens_a: ['who', 'was', 'jim', 'henson', '?']
    #entities_a: ['UNK', 'UNK', 'Q191037', 'UNK', 'UNK'] where the QID is stored for the first appearance of the term
    tokens_a, entities_a = tokenizer.tokenize(text_a, entities_a)
    tokens_b, entities_b = tokenizer.tokenize(text_b, entities_b)

    #tokens: ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '.', '[SEP]']
    #entities: ['UNK', 'UNK', 'UNK', 'Q191037', 'UNK', 'UNK', 'UNK', 'Q191037', 'UNK', 'UNK', 'UNK', 'Q2629392', 'UNK', 'UNK', 'UNK']
    #segments_ids: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    #input_mask:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    tokens,entities,segments_ids,input_mask = concatenate_tokens_entities(tokens_a,tokens_b,entities_a,entities_b)

    # Mask tokens that we will try to predict back with `BertForMaskedLM`
    for idx in masked_indices:
        tokens[idx] = '[MASK]'

    # Convert tokens to BPE indices: [101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 103, 2001, 1037, 13997, 11510, 1012, 102]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

    #ent_tensor: shape(1,15,100)
    #ent_mask: [[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]
    ents_tensor,ent_mask = get_entities_embeddings_and_mask(entities,QID_to_eid, eid_to_embeddings,device=device)

    # Convert inputs to PyTorch tensors, can be put on cuda
    tokens_tensor = torch.tensor([indexed_tokens],device=device)
    segments_tensors = torch.tensor([segments_ids],device=device)

    return tokens_tensor,ents_tensor,ent_mask,segments_tensors