import torch.nn as nn

class Fusion(nn.Module):
    """Abstract class for the fusion of text and knowledge representations
    
    """
    def __init__(self):
        """ TO BE OVERWRITTEN  
        constructs Fusion module
        """
        super().__init__()
    
    def forward(self,args):
        """ TO BE OVERWRITTEN
        Performs the information fusion between the token and entity embeddings
        Args:
            May contain:
            token_embeddings: the tokens embeddings
            entity_embeddings: the entity embeddings
            token/entity masks
        Returns:
            token_embeddings: the enhanced embeddings of the tokens 
            entity embeddings(Optional): the enhanced embeddings of the entities
        """
        raise NotImplementedError()