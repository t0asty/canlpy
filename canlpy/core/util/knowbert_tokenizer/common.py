import spacy
from spacy.tokens import Doc

def get_empty_candidates():
    """
    Returns the values corresponding to an empty candidate
    """
    return {
        "candidate_spans": [[-1, -1]],
        "candidate_entities": [["@@PADDING@@"]],
        "candidate_entity_priors": [[1.0]]
    }


# from https://spacy.io/usage/linguistic-features#custom-tokenizer-example
class WhitespaceTokenizer(object):
    """
    Simple whitespace tokenizer useable by spacy
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)