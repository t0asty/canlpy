import torch
import json

from pytorch_pretrained_bert.modeling import \
        BertLayer, BertAttention, BertSelfAttention, BertSelfOutput, \
        BertOutput, BertIntermediate, BertEncoder, BertLayerNorm


def get_dtype_for_module(module):
    # gets dtype for module parameters, for fp16 support when casting
    # we unfortunately can't set this during module construction as module
    # will be moved to GPU or cast to half after construction.
    return next(module.parameters()).dtype

def extend_attention_mask_for_bert(mask, dtype):
    # mask = (batch_size, timesteps)
    # returns an attention_mask useable with BERT
    # see: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L696
    extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# def init_bert_weights(module, initializer_range, extra_modules_without_weights=()):
#     # these modules don't have any weights, other then ones in submodules,
#     # so don't have to worry about init
#     modules_without_weights = (
#         BertEncoder, torch.nn.ModuleList, torch.nn.Dropout, BertLayer,
#         BertAttention, BertSelfAttention, BertSelfOutput,
#         BertOutput, BertIntermediate
#     ) + extra_modules_without_weights


#     # modified from pytorch_pretrained_bert
#     def _do_init(m):
#         if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             m.weight.data.normal_(mean=0.0, std=initializer_range)
#         elif isinstance(m, BertLayerNorm):
#             m.bias.data.zero_()
#             m.weight.data.fill_(1.0)
#         elif isinstance(m, modules_without_weights):
#             pass
#         else:
#             raise ValueError(str(m))

#         if isinstance(m, torch.nn.Linear) and m.bias is not None:
#             m.bias.data.zero_()

#     for mm in module.modules():
#         _do_init(mm)