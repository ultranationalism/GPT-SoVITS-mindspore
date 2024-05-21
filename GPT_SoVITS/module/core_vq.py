# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""
import typing as tp

#from einops import rearrange, repeat
#import torch
import mindspore as ms
from mindspore import nn,ops,Parameter
from mindspore.common.initializer import Uniform
from tqdm import tqdm


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    #moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    ops.assign(moving_avg,ops.mul(moving_avg, decay))
    ops.assign(moving_avg, ops.add(ops.mul(new, (1 - decay)), moving_avg))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = ms.Tensor(shape = shape, dtype=ms.float32, init=Uniform())
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = ops.randperm(num_samples)[:num]
    else:
        indices = ops.randint(0, num_samples, (num,))

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype
    max_kmeans_samples = 500
    samples = samples[:max_kmeans_samples, :]
    means = sample_vectors(samples, num_clusters)

    print("kmeans start ... ")
    for _ in tqdm(range(num_iters)):
        #diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        samples_expanded = ops.expand_dims(samples, 1)
        means_expanded = ops.expand_dims(means, 0)
        diffs=samples_expanded-means_expanded
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(axis=-1).indices
        bins = ops.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        #new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = ops.tensor_scatter_add(new_means, buckets.repeat(repeats=(1, dim)), samples) #可能有Bug
        new_means = new_means / bins_min_clamped[..., None]

        means = ops.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Cell):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., ms.Tensor], tp.Any] = (
            uniform_init if not kmeans_init else ops.zeros
        )
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.inited = ms.Tensor([not kmeans_init])
        self.cluster_size = ops.zeros(codebook_size)
        self.embed = embed
        self.embed_avg = embed.clone()

    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        ops.assign(self.embed, embed)
        ops.assign(self.embed_avg, embed.clone())
        ops.assign(self.cluster_size, cluster_size)
        ops.assign(self.inited, ms.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        # broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = ops.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        #self.embed.data.copy_(modified_codebook)
        ops.assign(self.embed,modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not ops.any(expired_codes):
            return

        #batch_samples = rearrange(batch_samples, "... d -> (...) d")
        new_shape = (-1, batch_samples.shape[-1])
        batch_samples.reshape(new_shape)
        self.replace_(batch_samples, mask=expired_codes)
        # broadcast_tensors(self.buffers())

    def preprocess(self, x):
        new_shape = (-1, x.shape[-1])
        x.reshape(new_shape)
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        #quantize = ops.embedding(embed_ind, self.embed)
        embedding_layer = nn.Embedding(vocab_size=self.embed.shape[0],embedding_size=self.embed.shape[1])
        embedding_layer.weight=Parameter(self.embed,requires_grad=False)
        quantize=embedding_layer(embed_ind)#TODO :Please remove code and enable comments after Mindshare 2.3. x version
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def construct(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = ops.one_hot(embed_ind, self.codebook_size).astype(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            #self.embed.data.copy_(embed_normalized)
            ops.assign(self.embed,embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Cell):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Dense(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Dense(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        #x = rearrange(x, "b d n -> b n d")
        x = x.transpose((0, 2, 1))
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        #quantize = rearrange(quantize, "b n d -> b d n")
        quantize = quantize.transpose((0, 2, 1))
        return quantize

    def construct(self, x):
        #device = x.device
        #x = rearrange(x, "b d n -> b n d")
        x = x.transpose((0, 2, 1))
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = ops.stop_gradient(x + (quantize - x))

        loss = Parameter([0.0], requires_grads=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = ops.mse_loss(ops.stop_gradient(quantize), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        #quantize = rearrange(quantize, "b n d -> b d n")
        quantize = quantize.transpose((0, 2, 1))
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Cell):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.CellList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def construct(
        self, x, n_q: tp.Optional[int] = None, layers: tp.Optional[list] = None
    ):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        out_quantized = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            if layers and i in layers:
                out_quantized.append(quantized)

        out_losses, out_indices = map(ops.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses, out_quantized

    def encode(
        self, x: ms.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None
    ) -> ms.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = ops.stack(all_indices)
        return out_indices

    def decode(self, q_indices: ms.Tensor, st: int = 0) -> ms.Tensor:
        quantized_out = ms.Tensor(0.0)
        for i, indices in enumerate(q_indices):
            layer = self.layers[st + i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
