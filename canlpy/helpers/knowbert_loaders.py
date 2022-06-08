import torch
import torch.nn as nn
import logging
import os
import tarfile
import shutil
import gzip
import numpy as np
from tqdm import tqdm

from canlpy.core.util.file_utils import cached_path
from canlpy.core.models.knowbert.model import KnowBert
from canlpy.core.components.fusion.knowbert_fusion import SolderedKG
from canlpy.core.components.fusion.knowbert_fusion.soldered_kg import EntityLinkingWithCandidateMentions
from canlpy.core.models.knowbert.knowledge import WordNetAllEmbedding
from canlpy.core.util.knowbert_tokenizer.vocabulary import Vocabulary
from canlpy.core.util.knowbert_tokenizer.tokenizer import KnowBertBatchifier

WORDNET_ARCHIVE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wordnet_model.tar.gz"
WORDNET_ENTITY_FILE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/entities.jsonl"
WORDNET_EMBEDDING_FILE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5"
WORDNET_VOCAB_FILE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/wordnet_synsets_mask_null_vocab.txt"

WIKI_ARCHIVE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_model.tar.gz"
WIKI_VOCAB_FILE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/vocabulary_wiki.tar.gz"
WIKI_EMBEDDING_FILE = "https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/entities_glove_format.gz"


def read_embeddings_from_text_file(gzip_filename: str,
                                    embedding_dim: int,
                                    vocab: Vocabulary,
                                    namespace: str = "tokens") -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than ``embedding_dim`` raise a warning and are skipped.

    The remainder of the docstring is identical to ``_read_pretrained_embeddings_file``.
    """
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logging.info("Reading pretrained embeddings from file")

    with gzip.open(gzip_filename) as embeddings_file:
        for line in tqdm(embeddings_file):
            #bytes to str
            line = line.decode()
            token = line.split(' ', 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(' ')
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logging.warning("Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                                   embedding_dim, len(fields) - 1, line)
                    continue

                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

    if not embeddings:
        raise Exception("No embeddings of correct dimension found; you probably "
                                 "misspecified your embedding_dim parameter, or didn't "
                                 "pre-populate your Vocabulary")

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    embedding_matrix =  torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,embeddings_std)

    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logging.info(f"Token {token} was not found in the embedding file. Initialising randomly.")

    logging.info(f"Pretrained embeddings were found for {num_tokens_found} out of {vocab_size} tokens")
    embedding = nn.Embedding.from_pretrained(embedding_matrix)
    return embedding

def load_wordnet_model():

    embedding_file = cached_path(WORDNET_EMBEDDING_FILE)
    entity_file = cached_path(WORDNET_ENTITY_FILE)
    vocab_file = cached_path(WORDNET_VOCAB_FILE)
    state_dict_file = cached_path(WORDNET_ARCHIVE)

    span_attention_config = {'hidden_size': 200, 'intermediate_size': 1024, 'num_attention_heads': 4, 'num_hidden_layers': 1}
    span_encoder_config = {'hidden_size': 200, 'intermediate_size': 1024, 'num_attention_heads': 4, 'num_hidden_layers': 1}

    null_entity_id = 117662 # obtained for model.vocab.get_token_index('@@NULL@@', "entity")
    entity_dim = 200

    model_entity_embedder = WordNetAllEmbedding(
                    embedding_file = embedding_file,
                    entity_dim = entity_dim,
                    entity_file = entity_file,
                    vocab_file= vocab_file,
                    entity_h5_key = "tucker_gensen",
                    dropout = 0.1,
                    pos_embedding_dim = 25,
                    include_null_embedding = False)

    entity_embeddings = model_entity_embedder.entity_embeddings
    null_embedding = torch.zeros(entity_dim) #From wordnet code

    entity_linker = EntityLinkingWithCandidateMentions(
                    null_entity_id=null_entity_id,
                    entity_embedding = model_entity_embedder,
                    contextual_embedding_dim =768,
                    span_encoder_config = span_encoder_config,
                    margin = 0.2,
                    decode_threshold = 0.0,
                    loss_type = 'softmax',
                    max_sequence_length = 512,
                    dropout = 0.1,
                    output_feed_forward_hidden_dim = 100,
                    initializer_range = 0.02)

    wordnet_kg = SolderedKG(entity_linker = entity_linker, 
                                span_attention_config = span_attention_config,
                                should_init_kg_to_bert_inverse = False,
                                freeze = False)

    soldered_kgs = {'wordnet':wordnet_kg}

    span_extractor_global_attention_old_name = "wordnet_soldered_kg.entity_linker.disambiguator.span_extractor._global_attention._module.weight"
    span_extractor_global_attention_bias_old_name = "wordnet_soldered_kg.entity_linker.disambiguator.span_extractor._global_attention._module.bias"
    state_dict_map = {span_extractor_global_attention_old_name:span_extractor_global_attention_old_name.replace("._module",""),
                    span_extractor_global_attention_bias_old_name: span_extractor_global_attention_bias_old_name.replace("._module","")}

    tempdir = state_dict_file+"_extracted"
    os.makedirs(tempdir,exist_ok=True)

    print("Extracting model archives")
    with tarfile.open(state_dict_file, 'r:gz') as archive:
        archive.extractall(tempdir)

    weights_file = tempdir+'/weights.th'

    model = KnowBert(soldered_kgs = soldered_kgs,
                                    soldered_layers ={"wordnet": 9},
                                    bert_model_name = "bert-base-uncased",
                                    mode=None,state_dict_file=weights_file,
                                    strict_load_archive=True,
                                    remap_segment_embeddings = None,
                                    state_dict_map = state_dict_map)

    #Remove temporary directory
    shutil.rmtree(tempdir)

    return model

def load_wiki_model_and_batchifier():
    entity_vocabulary = Vocabulary.from_files(WIKI_VOCAB_FILE)
    compressed_embedding_file = cached_path(WIKI_EMBEDDING_FILE)
    wiki_state_dict_file = cached_path(WIKI_ARCHIVE)

    entity_dim = 300

    entity_embeddings = read_embeddings_from_text_file(compressed_embedding_file,embedding_dim=entity_dim,vocab=entity_vocabulary,namespace="entity")
    null_entity_id = entity_vocabulary.get_token_index('@@NULL@@', "entity")
    span_encoder_config = {'hidden_size': entity_dim, 'intermediate_size': 1024, 'num_attention_heads': 4, 'num_hidden_layers': 1}

    entity_linker = EntityLinkingWithCandidateMentions(
                    null_entity_id=null_entity_id,
                    entity_embedding = entity_embeddings,
                    contextual_embedding_dim =768,
                    span_encoder_config = span_encoder_config)

    span_attention_config = {'hidden_size': entity_dim, 'intermediate_size': 1024, 'num_attention_heads': 4, 'num_hidden_layers': 1}

    wiki_kg = SolderedKG(entity_linker = entity_linker, 
                                span_attention_config = span_attention_config,
                                should_init_kg_to_bert_inverse = False,
                                freeze = False)

    soldered_kgs = {'wiki':wiki_kg}

    span_extractor_global_attention_old_name = "wiki_soldered_kg.entity_linker.disambiguator.span_extractor._global_attention._module.weight"
    span_extractor_global_attention_bias_old_name = "wiki_soldered_kg.entity_linker.disambiguator.span_extractor._global_attention._module.bias"
    state_dict_map = {span_extractor_global_attention_old_name:span_extractor_global_attention_old_name.replace("._module",""),
                    span_extractor_global_attention_bias_old_name: span_extractor_global_attention_bias_old_name.replace("._module","")}

    tempdir = wiki_state_dict_file+"_extracted"
    os.makedirs(tempdir,exist_ok=True)

    print("Extracting model archives")
    with tarfile.open(wiki_state_dict_file, 'r:gz') as archive:
        archive.extractall(tempdir)

    weights_file = tempdir+'/weights.th'


    wiki_model = KnowBert(soldered_kgs = soldered_kgs,
                                    soldered_layers ={"wiki": 9},
                                    bert_model_name = "bert-base-uncased",
                                    mode=None,state_dict_file=weights_file,
                                    strict_load_archive=True,
                                    remap_segment_embeddings = None,
                                    state_dict_map = state_dict_map)

    wiki_batchifier = KnowBertBatchifier.get_wiki_batchifier(entity_vocab=entity_vocabulary)
                                    
    return wiki_model,wiki_batchifier

    
