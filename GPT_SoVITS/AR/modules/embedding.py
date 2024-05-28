# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/embedding.py
import math

import mindspore as ms
from mindspore import nn,ops,Parameter


class TokenEmbedding(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self) -> ms.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> ms.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def construct(self, x: ms.Tensor):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x


class SinePositionalEmbedding(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = Parameter(ops.ones(1), requires_grad=alpha)
        self.dropout = nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(ms.Tensor(0.0).broadcast_to((1, 4000)))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.shape[1] >= x.shape[1]:
                if self.pe.dtype != x.dtype:
                    self.pe = self.pe.to(dtype=x.dtype)
                return
        pe = ops.zeros(x.shape[1], self.embedding_dim)
        if self.reverse:
            position = ops.arange(
                x.shape[1] - 1, -1, -1.0, dtype=ms.float32
            ).unsqueeze(1)
        else:
            position = ops.arange(0, x.shape[1], dtype=ms.float32).unsqueeze(1)
        div_term = ops.exp(
            ops.arange(0, self.embedding_dim, 2, dtype=ms.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = ops.stop_gradient(pe.to( dtype=x.dtype))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.shape[1]]
        return self.dropout(output)
