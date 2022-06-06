import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,token_embeddings,entity_embeddings):
        """
        Performs the information fusion between the token and entity emmbeddings
        Args:
            token_embeddings: the tokens embeddings
            entity_embeddings: the entity embeddings
        Returns:
            token_embeddings: the enhanced embeddings of the tokens 
            entity embeddings(Optional): the enhanced embeddings of the entities
        """
        raise NotImplementedError()