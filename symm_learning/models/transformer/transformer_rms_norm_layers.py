import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import RMSNorm


class RMSNormTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, **kwargs):
        super().__init__(d_model, **kwargs)
        # override the two LayerNorms
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

class RMSNormTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, **kwargs):
        super().__init__(d_model, **kwargs)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)