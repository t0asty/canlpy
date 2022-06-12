from typing import Union, List,Dict
import numpy as np
import torch

from canlpy.core.util.knowbert_tokenizer.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator, truncate_sequence_pair
from canlpy.core.util.knowbert_tokenizer.wiki_linking_util import WikiCandidateMentionGenerator
from canlpy.core.util.knowbert_tokenizer.vocabulary import Vocabulary

WIKI_ENTITY_VOCAB_FILE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/vocabulary_wiki.tar.gz"


start_token = "[CLS]"
sep_token = "[SEP]"

class KnowBertBatchifier:
    """
    Takes a list of sentence strings and returns a tensor dict usable with
    a KnowBert model

    Parameters:
        tokenizer_and_candidate_generator: object that tokenizes a string and finds its corresponding candidate entities
        entity_vocabulary: a Vocaublary object corresponding to the entities
        batch_size: the batch size
        masking_strategy: the masking strategy: None or full_mask, if full_mask, when a token is masked, mask the correspobding entity
    """
    def __init__(self, tokenizer_and_candidate_generator, entity_vocabulary, batch_size=32,
                       masking_strategy=None):

        #Given 
        self.tokenizer_and_candidate_generator = tokenizer_and_candidate_generator 
        self.tokenizer_and_candidate_generator.whitespace_tokenize = False

        assert masking_strategy is None or masking_strategy == 'full_mask'
        self.masking_strategy = masking_strategy

        #map entity -> embedding id
        self.entity_vocabulary = entity_vocabulary
        self.batch_size = batch_size

    def _replace_mask(self, s):
        return s.replace('[MASK]', ' [MASK] ')
    
    def iter_batches(self, sentences_or_sentence_pairs: Union[List[str], List[List[str]]], verbose=True):
        """
        Takes in as input sentences_or_sentence_pairs and yields batches for a KnowBert model

        Args:
            sentences_or_sentence_pairs: sentence of pair of sentences
            verbose: whether to print the processed tokens 

        Returns:
            yield KnowBert batches, see `KnowBert` for more details
        """
        #Contains all the field of processing one sentence
        instances = []
        for sentence_or_sentence_pair in sentences_or_sentence_pairs:
            if isinstance(sentence_or_sentence_pair, list):
                assert len(sentence_or_sentence_pair) == 2
                tokens_candidates = self.tokenizer_and_candidate_generator.\
                        tokenize_and_generate_candidates(
                                self._replace_mask(sentence_or_sentence_pair[0]),
                                self._replace_mask(sentence_or_sentence_pair[1]))
            else:
                tokens_candidates = self.tokenizer_and_candidate_generator.\
                        tokenize_and_generate_candidates(self._replace_mask(sentence_or_sentence_pair))

            if verbose:
                print(tokens_candidates['tokens'])

            # now modify the masking if needed
            if self.masking_strategy == 'full_mask':
                # replace the mask span with a @@mask@@ span
                masked_indices = [index for index, token in enumerate(tokens_candidates['tokens']) if token == '[MASK]']

                spans_to_mask = set([(i, i) for i in masked_indices])
                _replace_candidates_with_mask_entity(
                        tokens_candidates['candidates'], spans_to_mask
                )

                # now make sure the spans are actually masked
                for key in tokens_candidates['candidates'].keys():
                    for span_to_mask in spans_to_mask:
                        found = False
                        for span in tokens_candidates['candidates'][key]['candidate_spans']:
                            if tuple(span) == tuple(span_to_mask):
                                found = True
                        if not found:
                            tokens_candidates['candidates'][key]['candidate_spans'].append(list(span_to_mask))
                            tokens_candidates['candidates'][key]['candidate_entities'].append(['@@MASK@@'])
                            tokens_candidates['candidates'][key]['candidate_entity_priors'].append([1.0])
                            tokens_candidates['candidates'][key]['candidate_segment_ids'].append(0)
                            # hack, assume only one sentence
                            assert not isinstance(sentence_or_sentence_pair, list)
            
            #Converts the tokens candidates list to numpy arrays
            arrays = self.tokenizer_and_candidate_generator.convert_tokens_candidates_to_array(tokens_candidates,self.entity_vocabulary)
            instances.append(arrays)

        for batch in self.batchify(instances):
            _convert_to_tensor(batch)
            yield batch

    def batchify(self,instances):
        """
        Takes a list of arguments (numpy arrays) corresponding to the processing 
        of each sentence and outputs dictionnary of tensor batches of those model input's. 
        Adds padding to numpy arrays if input arrays in same batch are of different length
        """
        
        for i in range(0, len(instances), self.batch_size):
            instance_batch = instances[i:i+self.batch_size]
            max_tokens = max(len(instance["tokens"]) for instance in instance_batch)
            batch = {}
            batch['tokens'] = {}
            batch['tokens']['tokens'] = []
            batch['segment_ids'] = []
            for instance in instance_batch:
                padded_tokens= instance["tokens"]
                padded_segment_ids = instance["segment_ids"]
                padded_tokens = _pad_to_shape(padded_tokens,(max_tokens),0)
                padded_segment_ids = _pad_to_shape(padded_segment_ids,(max_tokens),0)

                batch['tokens']['tokens'].append(padded_tokens)
                batch['segment_ids'].append(padded_segment_ids)

            batch['tokens']['tokens']= np.array(batch['tokens']['tokens'],np.int64)
            batch['segment_ids']= np.array(batch['segment_ids'],np.int64)

            #IMPORTANT: assume the number of candidate entities is already padded inside on instance(done by )
            #=> must padd across multiple instances
            kb_keys = instance_batch[0]['candidates'].keys()
            max_detected_entities = {}
            max_candidate_entities = {}

            for key in kb_keys:
                max_detected_entities[key] = max(instance['candidates'][key]['candidate_entity_ids'].shape[0] for instance in instance_batch)
                max_candidate_entities[key] = max(instance['candidates'][key]['candidate_entity_ids'].shape[1] for instance in instance_batch)

            batch['candidates'] = {}
            for key in kb_keys:
                batch['candidates'][key] = {}
                batch['candidates'][key]['candidate_spans'] = []
                batch['candidates'][key]['candidate_entities'] = {}
                batch['candidates'][key]['candidate_entities']['ids']=[]
                batch['candidates'][key]['candidate_entity_priors'] = []
                batch['candidates'][key]['candidate_segment_ids'] = []
                
            for instance in instance_batch:
                for key, entity_candidates in instance['candidates'].items():
                    max_entity = max_detected_entities[key]
                    max_candidate = max_candidate_entities[key]

                    padded_entity_ids = entity_candidates['candidate_entity_ids']
                    padded_spans = entity_candidates['candidate_spans']
                    padded_entity_priors = entity_candidates['candidate_entity_priors']
                    padded_entity_segment_ids = entity_candidates['candidate_segment_ids']

                    #Same for all here
                    # len_diff = max_entity-padded_entity_ids.shape[0]
                    
                    padded_spans = _pad_to_shape(padded_spans,(max_entity,2),-1)
                    padded_entity_ids = _pad_to_shape(padded_entity_ids,(max_entity,max_candidate),0)
                    padded_entity_priors = _pad_to_shape(padded_entity_priors,(max_entity,max_candidate),0)
                    padded_entity_segment_ids = _pad_to_shape(padded_entity_segment_ids,(max_entity),0)

                    batch['candidates'][key]['candidate_spans'].append(padded_spans)
                    batch['candidates'][key]['candidate_entities']['ids'].append(padded_entity_ids)
                    batch['candidates'][key]['candidate_entity_priors'].append(padded_entity_priors)
                    batch['candidates'][key]['candidate_segment_ids'].append(padded_entity_segment_ids)
            
            batch['candidates'][key]['candidate_spans'] = np.array(batch['candidates'][key]['candidate_spans'],np.int64)
            batch['candidates'][key]['candidate_entities']['ids'] = np.array(batch['candidates'][key]['candidate_entities']['ids'],np.int64)
            batch['candidates'][key]['candidate_entity_priors'] = np.array(batch['candidates'][key]['candidate_entity_priors'],np.float32)
            batch['candidates'][key]['candidate_segment_ids']= np.array(batch['candidates'][key]['candidate_segment_ids'],np.int64)

            yield(batch)

    @classmethod
    def get_wiki_batchifier(cls,entity_vocab=None,max_word_piece_sequence_length=512):
        """
        Returns a wiki batchifier

        Args:
            entity_vocab: optional, a Vocabulary class for the entities
            max_word_piece_sequence_length: the maximum allowed sequence length, the rest is truncated

        Returns:
            a KnowBert batchifier for wiki
        """
        if(entity_vocab is None):
            entity_vocab = Vocabulary.from_files(WIKI_ENTITY_VOCAB_FILE)

        candidate_mention_generator = WikiCandidateMentionGenerator()#use default params: None None True False None
        entity_candidate_generators = {'wiki':candidate_mention_generator}

        tokenizer_and_candidate_generator = BertTokenizerAndCandidateGenerator(entity_candidate_generators,bert_model_type= 'bert-base-uncased',\
            do_lower_case=True,whitespace_tokenize=True,max_word_piece_sequence_length=max_word_piece_sequence_length)

        
        return cls(tokenizer_and_candidate_generator,entity_vocab)


def _pad_to_shape(arr,out_shape,value):
    out = np.ones(out_shape, dtype=arr.dtype)*value
    #1D case
    if(len(arr.shape)==1):
        out[:arr.shape[0]]=arr
    #2D case
    elif(len(arr.shape)==2):
        out[:arr.shape[0],:arr.shape[1]]=arr
    else:
        raise NotImplementedError

    return out

def _convert_to_tensor(dict):
    """Converts in place all nested numpy arrays to tensors while conserving original dtype"""
    for key,value in dict.items():
        if(isinstance(value,np.ndarray)):
            dict[key] = torch.from_numpy(value)
        elif(isinstance(value,Dict)):
            _convert_to_tensor(value)

def _replace_candidates_with_mask_entity(candidates, spans_to_mask):
    """
    candidates = key -> {'candidate_spans': ...}
    """
    for candidate_key in candidates.keys():
        indices_to_mask = []
        for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
            if tuple(candidate_span) in spans_to_mask:
                indices_to_mask.append(k)
        for ind in indices_to_mask:
            candidates[candidate_key]['candidate_entities'][ind] = ['@@MASK@@']
            candidates[candidate_key]['candidate_entity_priors'][ind] = [1.0]


            
