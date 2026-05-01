from typing import Literal

import torch
import torch.nn as nn

from torch.nn import RMSNorm, LayerNorm


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer with configurable normalization.

    Extends :class:`torch.nn.TransformerEncoderLayer` by allowing the two
    internal ``LayerNorm`` modules to be replaced with ``RMSNorm``.

    Args:
        d_model: Dimensionality of the model embeddings.
        norm_module: Normalization type. One of ``'layernorm'`` or ``'rmsnorm'``.
            Defaults to ``'rmsnorm'``.
        **kwargs: Remaining arguments forwarded to
            :class:`torch.nn.TransformerEncoderLayer`.
    """
    def __init__(
            self,
            d_model,
            norm_module: Literal["layernorm", "rmsnorm"] = "rmsnorm",
            **kwargs
    ):
        super().__init__(d_model, **kwargs)
        norm_module_cls = (
            RMSNorm if norm_module == 'rmsnorm' else
            LayerNorm if norm_module == 'layernorm' else
            None
        )
        if norm_module_cls is None:
            raise ValueError(f"Unsupported norm_module '{norm_module}'. Expected 'layernorm' or 'rmsnorm'.")
        # override the two LayerNorms
        self.norm1 = norm_module_cls(d_model)
        self.norm2 = norm_module_cls(d_model)

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Transformer decoder layer with configurable normalization.

    Extends :class:`torch.nn.TransformerDecoderLayer` by allowing the three
    internal ``LayerNorm`` modules to be replaced with ``RMSNorm``.

    Args:
        d_model: Dimensionality of the model embeddings.
        norm_module: Normalization type. One of ``'layernorm'`` or ``'rmsnorm'``.
            Defaults to ``'rmsnorm'``.
        **kwargs: Remaining arguments forwarded to
            :class:`torch.nn.TransformerDecoderLayer`.
    """
    def __init__(
            self,
            d_model,
            norm_module: Literal["layernorm", "rmsnorm"] = "rmsnorm",
            **kwargs
    ):
        super().__init__(d_model, **kwargs)
        norm_module_cls = (
            RMSNorm if norm_module == 'rmsnorm' else
            LayerNorm if norm_module == 'layernorm' else
            None
        )
        if norm_module_cls is None:
            raise ValueError(f"Unsupported norm_module '{norm_module}'. Expected 'layernorm' or 'rmsnorm'.")
        self.norm1 = norm_module_cls(d_model)
        self.norm2 = norm_module_cls(d_model)
        self.norm3 = norm_module_cls(d_model)