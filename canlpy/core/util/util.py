
def get_dtype_for_module(module):
    """
    Returns the datatype of the parameters of a module
    """
    # gets dtype for module parameters, for fp16 support when casting
    # we unfortunately can't set this during module construction as module
    # will be moved to GPU or cast to half after construction.
    return next(module.parameters()).dtype

def extend_attention_mask_for_bert(mask, dtype):
    """
    Returns an attention_mask useable with BERT, 

    Args:
        mask: tensor of positions to be masked (0,1)
        dtype: the dtype of the returned mask
    """
    # mask = (batch_size, timesteps)
    # returns an attention_mask useable with BERT
    # see: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L696
    extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

def find_value(config,key):
    """
    Finds a specific value in a configuration object of nested dictionnaries

    Args:
        config: the configuration object
        key: the key the value corresponds to
    Returns:
        the corresponding value or None if it is not found 
    """
    if key in config:
        return config[key]
    else:
        for value in config.values():
            if(isinstance(value,dict)):
                temp = find_value(value,key)
                if temp:
                    return temp