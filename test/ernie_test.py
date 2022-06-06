import unittest
import torch
from torch.nn import Embedding
from canlpy.core.util.tokenization import BertTokenizer
from canlpy.core.models.ernie.model import ErnieForMaskedLM
from canlpy.previous_models.ernie.modeling import BertForMaskedLM
from canlpy.helpers.ernie_helpers import load_name_to_QID,load_QID_to_eid, process_sentences

KNOWLEDGE_DIR = '../canlpy/knowledge/ernie/'
PRE_TRAINED_DIR = '../canlpy/pretrained_models/ernie/ernie_base/'

NAME_TO_QID_FILE = KNOWLEDGE_DIR+ 'entity_map.txt'
QID_TO_EID_FILE = KNOWLEDGE_DIR+ 'entity2id.txt'
EID_TO_VEC_FILE = PRE_TRAINED_DIR + 'entity2vec.pt'

class TestERNIE(unittest.TestCase):
    '''Test class to verify that Custom ERNIE outputs the same predictions as the original ERNIE'''

    @classmethod
    def setUpClass(cls):
        cls.original_model, _= BertForMaskedLM.from_pretrained(PRE_TRAINED_DIR)
        cls.custom_model, _ = ErnieForMaskedLM.from_pretrained(PRE_TRAINED_DIR)       
        cls.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_DIR)
        cls.name_to_QID = load_name_to_QID(NAME_TO_QID_FILE)
        cls.QID_to_eid = load_QID_to_eid(QID_TO_EID_FILE)
        eid_to_embeddings = torch.load(EID_TO_VEC_FILE)
        cls.eid_to_embeddings = Embedding.from_pretrained(eid_to_embeddings)
        cls.device = 'cpu'

    def setUp(self):
        TestERNIE.original_model.eval()
        TestERNIE.custom_model.eval()

    def tearDown(self):
        TestERNIE.original_model.train()
        TestERNIE.custom_model.train()

    def test_custom_equals_original_ERNIE(self):

        text_a = "Who was Jim Henson ? "
        text_b = "Jim Henson was a puppeteer ."
        for masked_idx in range(1,10):
            tokens_tensor,ents_tensor,ent_mask,segments_tensors = process_sentences(text_a,text_b,[masked_idx],TestERNIE.name_to_QID,TestERNIE.QID_to_eid,TestERNIE.eid_to_embeddings,TestERNIE.tokenizer,TestERNIE.device)
        # Predict all tokens
            with torch.no_grad():
                predictions_custom = TestERNIE.custom_model(tokens_tensor, ents_tensor, ent_mask, segments_tensors)
                predictions_original = TestERNIE.original_model(tokens_tensor, ents_tensor, ent_mask, segments_tensors)

                # confirm we were able to predict 'henson'
                predicted_index_custom = torch.argmax(predictions_custom[0, masked_idx]).item()

                predicted_index_original = torch.argmax(predictions_original[0, masked_idx]).item()

                #Given in example.py from github repository
                if(masked_idx==8):
                    predicted_token_original = TestERNIE.tokenizer.convert_ids_to_tokens([predicted_index_custom])[0]
                    self.assertEqual(predicted_token_original,'henson')
                error_msg = "The predictions between custom and original are not equal"
                equal = predictions_custom.eq(predictions_original).all()
                self.assertTrue(equal,error_msg)

if __name__ == '__main__':
    unittest.main()