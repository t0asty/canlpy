import torch
import math
import torch.nn as nn

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": nn.functional.relu, "swish": nn.functional.silu}

def get_activation_function(name:str):
    return ACT2FN[name]