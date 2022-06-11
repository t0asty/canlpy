
import torch
import torch.nn as nn
from canlpy.core.models.knowbert.metrics import Average, CategoricalAccuracy, WeightedAverage, ExponentialMovingAverage, MeanReciprocalRank
from pytorch_pretrained_bert.modeling import BertForPreTraining

#Knowbert with masked LM and NSP
class KnowBertForPreTraining(nn.Module):
    """
    A knowbert model with Masked LM and NSP loss used for pretraining 
    Parameters:
        knowbert_model: the underlying knowbert model
    """
    def __init__(self,knowbert_model):

        #The model should be initialized with the corresponding training mode (freeze or not the entity linker)
        self.knowbert_model = knowbert_model
        self.nsp_loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        self.lm_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    
    def _compute_loss(self,
                      contextual_embeddings,
                      pooled_output,
                      lm_label_ids,
                      next_sentence_label):

        """
        Computes the losses of the model

        Args:
            contextual_embeddings: the predicted embeddings, used for MLM
            pooled_output: the pooled CLS token used for NSP
            lm_label_ids: the expected label ids
            next_sentence_label: the next sentence label

        Returns:
            loss: the sum of the internal entity linker loss and the MLM and NSP loss
        """


        # (batch_size, timesteps, vocab_size), (batch_size, 2)
        prediction_scores, seq_relationship_score = knowbert_model.pretraining_heads(
                contextual_embeddings, pooled_output
        )

        if lm_label_ids is not None:
            # Loss
            vocab_size = prediction_scores.shape[-1]
            masked_lm_loss = self.lm_loss_function(
                prediction_scores.view(-1, vocab_size), lm_label_ids["lm_labels"].view(-1)
            )
        else:
            masked_lm_loss = 0.0

        if next_sentence_label is not None:
            next_sentence_loss = self.nsp_loss_function(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )

        else:
            next_sentence_loss = 0.0
        
        return masked_lm_loss, next_sentence_loss
        

    def forward(self, tokens=None, segment_ids=None, candidates=None,
                lm_label_ids=None, next_sentence_label=None, **kwargs):

        output = self.knowbert_model(tokens, segment_ids, candidates,kwargs)

        #Can also do something about the loss obtained for the respective KG_Soldered
        loss = output['loss']
        contextual_embeddings = output['contextual_embeddings']
        pooled_output = output['pooled_output']
    
        if lm_label_ids is not None or next_sentence_label is not None:
                # compute MLM and NSP loss
                masked_lm_loss, next_sentence_loss = self._compute_loss(
                        contextual_embeddings,
                        pooled_output,
                        lm_label_ids,
                        next_sentence_label)

                loss = loss + masked_lm_loss + next_sentence_loss

        # if 'mask_indicator' in kwargs:
        #     self._compute_mrr(contextual_embeddings,
        #                     pooled_output,
        #                     lm_label_ids['lm_labels'],
        #                     kwargs['mask_indicator'])

        return loss
