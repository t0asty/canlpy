import torch
from torch.nn import Embedding
import tagme
from canlpy.helpers.ernie_helpers import load_name_to_QID,load_QID_to_eid,process_sentences
import pickle

from canlpy.core.components.tokenization import BertTokenizer
from canlpy.core.models.cokebert.model import CokeBertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#import logging
#logging.basicConfig(level=logging.INFO)

device = 'cpu'

KNOWLEDGE_DIR = '../experiments/CokeBert_evaluation/data/kg_embed/'
PRE_TRAINED_DIR = '../canlpy/pretrained_models/cokebert'

NAME_TO_QID_FILE = KNOWLEDGE_DIR+ 'entity_map.txt'
QID_TO_EID_FILE = KNOWLEDGE_DIR+ 'entity2id.txt'
EID_TO_VEC_FILE = KNOWLEDGE_DIR + 'entity2vec.vec'
REL_TO_VEC_FILE = KNOWLEDGE_DIR + 'relation2vec.vec'

model, _ = CokeBertForMaskedLM.from_pretrained(PRE_TRAINED_DIR)
model.eval()

def load_ent_emb_static():

    with open('../experiments/CokeBert_evaluation/data/load_data_n/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
        ent_to_neighbors = pickle.load(f)

    with open('../experiments/CokeBert_evaluation/data/load_data_n/e1_r_list_2D_Tensor.pkl', 'rb') as f:
        ent_to_relations = pickle.load(f)

    with open('../experiments/CokeBert_evaluation/data/load_data_n/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
        ent_to_outORin = pickle.load(f)

    return ent_to_neighbors, ent_to_relations, ent_to_outORin


def load_k_v_queryR(input_ent):
        input_ent = input_ent.cpu()

        ent_pos_s = torch.nonzero(input_ent)

        max_entity=0
        value=0
        idx_1 = 0
        last_part = 0
        for idx_2,x in enumerate(ent_pos_s):
            if int(x[0]) != value:
                max_entity = max(idx_2-idx_1,max_entity)
                idx_1 = idx_2
                value = int(x[0])
                last_part = 1
            else:
                last_part+=1
        max_entity = max(last_part,max_entity)

        new_input_ent = list()
        for i_th, ten in enumerate(input_ent):
            ten_ent = ten[ten!=0]
            new_input_ent.append( torch.cat( (ten_ent,( torch.LongTensor( [0]*(max_entity-ten_ent.shape[0]) ) ) ) ) )

        input_ent = torch.stack(new_input_ent)

        #Neighbor
        input_ent_neighbor = torch.index_select(ent_to_neighbors,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()

        #create input_ent_neighbor_1

        input_ent_neighbor_emb_1 = torch.index_select(embed_ent,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])) #
        input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_neighbors.shape[1],embed_ent.shape[-1])

        #create input_ent_r_1:
        input_ent_r_emb_1 = torch.index_select(ent_to_relations,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
        input_ent_r_emb_1 = torch.index_select(r_embed,0,input_ent_r_emb_1.reshape(input_ent_r_emb_1.shape[0]*input_ent_r_emb_1.shape[1])) #
        input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_relations.shape[1],r_embed.shape[-1])

        #create outORin_1:
        input_ent_outORin_emb_1 = torch.index_select(ent_to_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
        input_ent_outORin_emb_1 = input_ent_outORin_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb_1.shape[1])
        input_ent_outORin_emb_1 = input_ent_outORin_emb_1.unsqueeze(3)


        #create input_ent_neighbor_2
        input_ent_neighbor_2 = torch.index_select(ent_to_neighbors,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()

        input_ent_neighbor_emb_2 = torch.index_select(embed_ent,0,input_ent_neighbor_2.reshape(input_ent_neighbor_2.shape[0]*input_ent_neighbor_2.shape[1])) #
        input_ent_neighbor_emb_2 = input_ent_neighbor_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_neighbors.shape[1],ent_to_neighbors.shape[1],embed_ent.shape[-1])


        #create input_ent_r_2:
        input_ent_r_2 = torch.index_select(ent_to_relations,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()

        input_ent_r_emb_2 = torch.index_select(r_embed,0,input_ent_r_2.reshape(input_ent_r_2.shape[0]*input_ent_r_2.shape[1])) #
        input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_relations.shape[1],ent_to_neighbors.shape[1],r_embed.shape[-1])

        #create outORin_2: #?
        input_ent_outORin_emb_2 = torch.index_select(ent_to_outORin,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1]))

        input_ent_outORin_emb_2 = input_ent_outORin_emb_2.reshape(input_ent_r_emb_2.shape[0],input_ent_r_emb_2.shape[1],input_ent_r_emb_2.shape[2],input_ent_r_emb_2.shape[3])
        input_ent_outORin_emb_2 = input_ent_outORin_emb_2.unsqueeze(4)

        k_1 = input_ent_outORin_emb_1.to(device=device)*input_ent_r_emb_1.to(device=device)
        v_1 = input_ent_neighbor_emb_1.to(device=device)+k_1
        k_2 = input_ent_outORin_emb_2.to(device=device)*input_ent_r_emb_2.to(device=device)
        v_2 = input_ent_neighbor_emb_2.to(device=device)+k_2

        return k_1,v_1,k_2,v_2

def eval_sentence(text_a,text_b,model,tokenizer,masked_indices):

    tokens_tensor,ents_tensor,ent_mask,segments_tensors = process_sentences(text_a,text_b,masked_indices,name_to_QID,QID_to_eid,ent_embed,tokenizer,device=device)

    # Predict all tokens
    with torch.no_grad():
        #ents_tensor = ents_tensor+1
        k_1, v_1, k_2, v_2 = load_k_v_queryR((ents_tensor + 1).to(torch.long))
        predictions = model(tokens_tensor, ents_tensor, ent_mask, segments_tensors, k_v_s=[(k_1, v_1), (k_2, v_2)])

        # confirm we were able to predict 'henson'
        for masked_index in masked_indices:
            predicted_index = torch.argmax(predictions[0, masked_index]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            print(f"predicted_token for index {masked_index} is {predicted_token}")

#Load pre-trained model tokenizer (vocabulary)
#Special tokenizer for text and entities
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_DIR)

#Eg: 'Northern Ireland': 'Q26'
name_to_QID = load_name_to_QID(NAME_TO_QID_FILE)
#Eg: {'Q11456633': 4525438, 'Q8863973': 1628631}
QID_to_eid = load_QID_to_eid(QID_TO_EID_FILE)

#eid_to_embeddings = torch.load(EID_TO_VEC_FILE)
vecs = []
vecs.append([0]*100)
with open(EID_TO_VEC_FILE, 'r') as fin:
    for line in fin:
        vec = line.strip().split('\t')
        vec = [float(x) for x in vec]
        vecs.append(vec)
embed_ent = torch.FloatTensor(vecs)
ent_embed = torch.nn.Embedding.from_pretrained(embed_ent)
#Creats a dictionnary of entity index->embeddings

ent_to_neighbors, ent_to_relations, ent_to_outORin = load_ent_emb_static()

vecs = []
vecs.append([0]*100) # CLS
with open(REL_TO_VEC_FILE, 'r') as fin:
#with open("kg_embed/relation2vec.del", 'r') as fin:
    for line in fin:
        vec = line.strip().split('\t')
        vec = [float(x) for x in vec]
        vecs.append(vec)
r_embed = torch.FloatTensor(vecs)

text_a = "Who was Jim Henson ? "
text_b = "Jim Henson was a puppeteer ."

#tokens_tensor,ents_tensor,ent_mask,segments_tensors = process_sentences(text_a,text_b,masked_indices,name_to_QID,QID_to_eid,tokenizer)
#tokens: ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '.', '[SEP]']
masked_indices = [8,11,12]#henson, puppet, ##eer
eval_sentence(text_a,text_b,model,tokenizer,masked_indices)