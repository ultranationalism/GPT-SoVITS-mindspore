# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/transformer.py
import copy
import math
import numbers
from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import mindspore as ms
from AR.modules.activation import MultiheadAttention
from AR.modules.scaling import BalancedDoubleSwish
from mindspore import nn,Tensor,ops,Parameter
from mindspore.common.initializer import initializer,One,Zero
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out

_shape_t = Union[int, List[int]]


class LayerNorm(nn.Cell):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = { "dtype": dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        """
        if self.elementwise_affine:
            self.weight = Parameter(
                ops.ones(self.normalized_shape, **factory_kwargs)
            )
            self.bias = Parameter(
                ops.ones(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.weight=Parameter(Tensor())
            self.bias=Parameter(Tensor())
        """
        #self.reset_parameters()
        self.layer_norm=nn.LayerNorm(normalized_shape,epsilon =self.eps,gamma_init ='ones',beta_init='zeros')


    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            #nn.init.ones_(self.weight)
            self.weight.set_data(initializer(One(), self.weight.shape, self.weight.dtype))
            #nn.init.zeros_(self.bias)
            self.bias.set_data(initializer(One(), self.bias.shape, self.bias.dtype))

    def construct(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                self.layer_norm(
                    input
                ),
                embedding,
            )

        assert embedding is None
        return self.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class IdentityNorm(nn.Cell):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super(IdentityNorm, self).__init__()

    def construct(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


class TransformerEncoder(nn.Cell):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
        cache=None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    cache=cache,
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                cache=cache,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Cell):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = ops.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=ms.float32,
        linear1_self_attention_cls: nn.Cell = nn.Dense,
        linear2_self_attention_cls: nn.Cell = nn.Dense,
        linear1_feedforward_cls: nn.Cell = nn.Dense,
        linear2_feedforward_cls: nn.Cell = nn.Dense,
        layer_norm_cls: nn.Cell = nn.LayerNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        # print(233333333333,d_model,nhead)
        # import os
        # os._exit(2333333)
        self.self_attn = MultiheadAttention(
            d_model,  # 512 16
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        fan_in, _ = _calculate_fan_in_and_fan_out((dim_feedforward, d_model))
        bound = 1 / math.sqrt(fan_in)
        self.dense1 = nn.Dense(d_model, dim_feedforward, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        fan_in1, _ = _calculate_fan_in_and_fan_out((d_model, dim_feedforward))
        bound1 = 1 / math.sqrt(fan_in1)
        self.dense2 = nn.Dense(dim_feedforward, d_model, dtype=dtype)


        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        # # We can't test self.activation in forward() in TorchScript,
        # # so stash some information about it instead.
        # if activation is F.relu or isinstance(activation, torch.nn.ReLU):
        #     self.activation_relu_or_gelu = 1
        # elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
        #     self.activation_relu_or_gelu = 2
        # else:
        #     self.activation_relu_or_gelu = 0
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = ops.relu

    def construct(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        cache=None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != ms.bool_ and not ops.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
                cache=cache,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask,cache=cache))
            x = self.norm2(x + self._ff_block(x))

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        cache=None,
    ) -> Tensor:
        # print(x.shape,attn_mask.shape,key_padding_mask)
        # ms.Tensor.shape([1, 188, 512]) ms.Tensor.shape([188, 188]) None
        # import os
        # os._exit(23333)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            cache=cache,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.dense2(self.dropout(self.activation(self.dense1(x))))
        return self.dropout2(x)


class AdaptiveLayerNorm(nn.Cell):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Dense(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def construct(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = ops.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                axis =-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = ops.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            axis =-1,
        )
        return weight * self.norm(input) + bias


def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str):
    if activation == "relu":
        return ops.relu
    if activation == "gelu":
        return ops.gelu

    raise ValueError(f"The activation must be relu/gelu, but get {activation}")
