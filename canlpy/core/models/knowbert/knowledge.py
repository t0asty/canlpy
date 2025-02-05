# This file is adapted from the AllenAI library at https://github.com/allenai/kb
# Copyright by the AllenAI authors.

import torch
import torch.nn as nn
import json
import h5py
import gzip
import logging
import numpy as np
from tqdm import tqdm

from canlpy.core.models.bert.model import init_weights
from canlpy.core.util.knowbert_tokenizer.vocabulary import Vocabulary

class EntityEmbedder():
    """An extension to nn.Embedding but allow for more functionalities"""
    pass

def read_embeddings_from_text_file(gzip_filename: str,
                                    embedding_dim: int,
                                    vocab: Vocabulary,
                                    namespace: str = "tokens") -> nn.Embedding:
    """
    Read pre-trained word vectors from a compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than `embedding_dim` raise a warning and are skipped.

    Args:
        gzip_filename: the file name containing the embeddings
        embedding_dim: the dimension of the embeddings
        vocab: a vocabulary object containing the index to token mapping
        namespace: the namespace the embeddings correpond to, eg: entity

    Returns:
        embeddings: nn.Embedding, the extracted embeddings
    """
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    print("Reading pretrained embeddings from file")

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

class WordNetAllEmbedding(nn.Module, EntityEmbedder):
    """
    Combines pretrained fixed embeddings with learned POS embeddings.

    Given entity candidate list:
        - get list of unique entity ids
        - look up
        - concat POS embedding
        - linear project to candidate embedding shape

    Parameters:
        embedding_file: a file containing the embeddings
        entity_dim: the dimension of the entities
        entity_file: the file containing the entities
        vocab_file: the file containing the vocabulary
        entity_h5_key: unknown
        dropout: dropout to apply on the embeddings
        pos_embedding_dim: the dimension of the position embedding
        include_null_embedding: whether to include null embedding
    """
    POS_MAP = {
        '@@PADDING@@': 0,
        'n': 1,
        'v': 2,
        'a': 3,
        'r': 4,
        's': 5,
        # have special POS embeddings for mask / null, so model can learn
        # it's own representation for them
        '@@MASK@@': 6,
        '@@NULL@@': 7,
        '@@UNKNOWN@@': 8
    }

    def __init__(self,
                 embedding_file: str,
                 entity_dim: int,
                 entity_file: str = None,
                 vocab_file: str = None,
                 entity_h5_key: str = 'conve_tucker_infersent_bert',
                 dropout: float = 0.1,
                 pos_embedding_dim: int = 25,
                 include_null_embedding: bool = False):
        """
        pass pos_emedding_dim = None to skip POS embeddings and all the
            entity stuff, using this as a pretrained embedding file
            with feedforward
        """

        super().__init__()

        if pos_embedding_dim is not None:
            # entity_id -> pos abbreviation, e.g.
            # 'cat.n.01' -> 'n'
            # includes special, e.g. '@@PADDING@@' -> '@@PADDING@@'
            entity_to_pos = {}
            with open(entity_file, 'r') as fin:
                for node in fin:
                    node = json.loads(node)
                    if node['type'] == 'synset':
                        entity_to_pos[node['id']] = node['pos']
            for special in ['@@PADDING@@', '@@MASK@@', '@@NULL@@', '@@UNKNOWN@@']:
                entity_to_pos[special] = special
    
            # list of entity ids
            entities = ['@@PADDING@@']
            with open(vocab_file, 'r') as fin:
                for line in fin:
                    entities.append(line.strip())
    
            # the map from entity index id -> pos embedding id,
            # will use for POS embedding lookup
            entity_id_to_pos_index = [
                 self.POS_MAP[entity_to_pos[ent]] for ent in entities
            ]
            self.register_buffer('entity_id_to_pos_index', torch.tensor(entity_id_to_pos_index))
    
            self.pos_embeddings = torch.nn.Embedding(len(entities), pos_embedding_dim)
            init_weights(self.pos_embeddings, 0.02)

            self.use_pos = True
        else:
            self.use_pos = False

        # load the embeddings
        with h5py.File(embedding_file, 'r') as fin:
            entity_embeddings = fin[entity_h5_key][...]
        self.entity_embeddings = torch.nn.Embedding(
                entity_embeddings.shape[0], entity_embeddings.shape[1],
                padding_idx=0
        )
        self.entity_embeddings.weight.data.copy_(torch.tensor(entity_embeddings).contiguous())

        if pos_embedding_dim is not None:
            assert entity_embeddings.shape[0] == len(entities)
            concat_dim = entity_embeddings.shape[1] + pos_embedding_dim
        else:
            concat_dim = entity_embeddings.shape[1]

        self.proj_feed_forward = torch.nn.Linear(concat_dim, entity_dim)
        init_weights(self.proj_feed_forward, 0.02)

        self.dropout = torch.nn.Dropout(dropout)

        self.embedding_dim = entity_dim

        self.include_null_embedding = include_null_embedding
        self.null_embedding=None
        if include_null_embedding:
            # a special embedding for null
            entities = ['@@PADDING@@']
            with open(vocab_file, 'r') as fin:
                for line in fin:
                    entities.append(line.strip())
            self.null_id = entities.index("@@NULL@@")
            self.null_embedding = torch.nn.Parameter(torch.zeros(entity_dim))
            self.null_embedding.data.normal_(mean=0.0, std=0.02)

    def get_output_dim(self):
        return self.embedding_dim

    def get_null_embedding(self):
        return self.null_embedding

    def forward(self, entity_ids):
        """
        Args:
            entity_ids: torch.LongTensor (batch_size, num_candidates, num_entities) of entity ids

        Returns:
            entity_embeddings:  torch.LongTensor (batch_size, num_candidates, num_entities, embed_dim)
        """
        # get list of unique entity ids
        unique_ids, unique_ids_to_entity_ids = torch.unique(entity_ids, return_inverse=True)
        # unique_ids[unique_ids_to_entity_ids].reshape(entity_ids.shape)

        # look up (num_unique_embeddings, full_entity_dim)
        unique_entity_embeddings = self.entity_embeddings(unique_ids.contiguous()).contiguous()

        # get POS tags from entity ids (form entity id -> pos id embedding)
        # (num_unique_embeddings)
        if self.use_pos:

            #Reshape to simulate embedding of size 1
            unique_pos_ids = torch.nn.functional.embedding(unique_ids, self.entity_id_to_pos_index.reshape(-1,1)).flatten().contiguous()
            # (num_unique_embeddings, pos_dim)
            unique_pos_embeddings = self.pos_embeddings(unique_pos_ids).contiguous()
            # concat
            entity_and_pos = torch.cat([unique_entity_embeddings, unique_pos_embeddings], dim=-1)
        else:
            entity_and_pos = unique_entity_embeddings

        # run the ff
        # (num_embeddings, entity_dim)
        projected_entity_and_pos = self.dropout(self.proj_feed_forward(entity_and_pos.contiguous()))

        # replace null if needed
        if self.include_null_embedding:
            null_mask = unique_ids == self.null_id
            projected_entity_and_pos[null_mask] = self.null_embedding

        # remap to candidate embedding shape
        return projected_entity_and_pos[unique_ids_to_entity_ids].contiguous()