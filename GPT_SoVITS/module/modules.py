import math
import numpy as np
import mindspore as ms
from mindspore import ops,nn,Parameter
from mindspore.common.initializer import Normal

from mindspore.nn import Conv1d
from mindnlp.modules.weight_norm import remove_weight_norm, weight_norm

from module import commons
from module.commons import init_weights, get_padding
from module.transforms import piecewise_rational_quadratic_transform
from .spectral_norm import spectral_norm as s_n
#import torch.distributions as D
import mindspore.nn.probability.distribution as D


LRELU_SLOPE = 0.1


class LayerNorm(nn.Cell):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.layer_norm = nn.LayerNorm(normalized_shape=(self.channels,),epsilon=self.eps,gamma_init='ones',beta_init='zeros')


    def construct(self, x):
        x = x.swapaxes(1, -1)
        x = self.layer_norm(x)
        return x.swapaxes(1, -1)


class ConvReluNorm(nn.Cell):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.CellList()
        self.norm_layers = nn.CellList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p=p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def construct(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Cell):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p=p_dropout)
        self.convs_sep = nn.CellList()
        self.convs_1x1 = nn.CellList()
        self.norms_1 = nn.CellList()
        self.norms_2 = nn.CellList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    group=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def construct(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = ops.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = ops.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(nn.Cell):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.CellList()
        self.res_skip_layers = nn.CellList()
        self.drop = nn.Dropout(p=p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def construct(self, x, x_mask, g=None, **kwargs):
        output = ops.zeros_like(x)
        n_channels_tensor = ms.Tensor([self.hidden_channels],ms.int32)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = ops.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


class ResBlock1(nn.Cell):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.convs1 = nn.CellList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                    ),
                    dim=0
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                    ),
                    dim=0
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                    ),
                    dim=0
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.CellList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    )
                    ,dim=0
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                    dim=0
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                    dim=0
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def construct(self, x, x_mask=None):
        n=0
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = ops.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt=ops.pad(xt,((0, 0), (get_padding(self.kernel_size, self.dilation[n]), get_padding(self.kernel_size, self.dilation[n])), (0, 0)))
            xt = c1(xt)
            xt = ops.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt=ops.pad(xt,((0, 0), (get_padding(self.kernel_size, 1), get_padding(self.kernel_size, 1)), (0, 0)))
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        n+=1
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Cell):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.convs = nn.CellList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                    ),
                    dim=0
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                    ),
                    dim=0
                ),
            ]
        )
        self.convs.apply(init_weights)

    def construct(self, x, x_mask=None):
        n=0
        for c in self.convs:
            xt = ops.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt=ops.pad(xt,((0, 0), (get_padding(self.kernel_size, self.dilation[n]), get_padding(self.kernel_size, self.dilation[n])), (0, 0)))
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        n+=1
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Log(nn.Cell):
    def construct(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = ops.log(ops.clamp_min(x, 1e-5)) * x_mask
            logdet = ops.sum(-y, [1, 2])
            return y, logdet
        else:
            x = ops.exp(x) * x_mask
            return x


class Flip(nn.Cell):
    def construct(self, x, *args, reverse=False, **kwargs):
        x = ops.flip(x, [1])
        if not reverse:
            logdet = ops.zeros(x.shape(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Cell):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = Parameter(ops.zeros([channels, 1]))
        self.logs = Parameter(ops.zeros([channels, 1]))

    def construct(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + ops.exp(self.logs) * x
            y = y * x_mask
            logdet = ops.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * ops.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Cell):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def construct(self, x, x_mask, g=None, reverse=False):
        x0, x1 = ops.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = ops.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = ops.zeros_like(m)

        if not reverse:
            x1 = m + x1 * ops.exp(logs) * x_mask
            x = ops.cat([x0, x1], 1)
            logdet = ops.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * ops.exp(-logs) * x_mask
            x = ops.cat([x0, x1], 1)
            return x


class ConvFlow(nn.Cell):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def construct(self, x, x_mask, g=None, reverse=False):
        x0, x1 = ops.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = ops.cat([x0, x1], 1) * x_mask
        logdet = ops.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class LinearNorm(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        spectral_norm=False,
    ):
        super(LinearNorm, self).__init__()
        self.fc = nn.Dense(in_channels, out_channels, bias)

        if spectral_norm:
            self.fc = s_n(self.fc)

    def construct(self, input):
        out = self.fc(input)
        return out


class Mish(nn.Cell):
    def __init__(self):
        super(Mish, self).__init__()

    def construct(self, x):
        return x * ops.tanh(ops.softplus(x))


class Conv1dGLU(nn.Cell):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2 * out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = ops.split(x, split_size_or_sections=self.out_channels, axis=1)
        x = x1 * ops.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class ConvNorm(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        spectral_norm=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            has_bias=bias,
        )

        if spectral_norm:
            self.conv = s_n(self.conv)

    def construct(self, input):
        out = self.conv(input)
        return out


class MultiHeadAttention(nn.Cell):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0, spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Dense(d_model, n_head * d_k)
        self.w_ks = nn.Dense(d_model, n_head * d_k)
        self.w_vs = nn.Dense(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_model, 0.5), dropout=dropout
        )

        self.fc = nn.Dense(n_head * d_v, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if spectral_norm:
            self.w_qs = s_n(self.w_qs)
            self.w_ks = s_n(self.w_ks)
            self.w_vs = s_n(self.w_vs)
            self.fc = s_n(self.fc)

    def construct(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.shape

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)
        )  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn


class ScaledDotProductAttention(nn.Cell):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(axis=2)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, q, k, v, mask=None):
        attn = ops.bmm(q, k.swapaxes(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = ops.bmm(p_attn, v)
        return output, attn


class MelStyleEncoder(nn.Cell):
    """MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(p=self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(p=self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(
            self.n_head,
            self.hidden_dim,
            self.hidden_dim // self.n_head,
            self.hidden_dim // self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = ops.mean(x, axis=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = ops.div(x, len_)
        return out

    def construct(self, x, mask=None):
        x = x.swapaxes(1, 2)
        if mask is not None:
            mask = (mask.int() == 0).squeeze(1)
        max_len = x.shape[1]
        slf_attn_mask = (
            mask.unsqueeze(1).broadcast_to((-1, max_len, -1)) if mask is not None else None
        )

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.swapaxes(1, 2)
        x = self.temporal(x)
        x = x.swapaxes(1, 2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w.unsqueeze(-1)


class MelStyleEncoderVAE(nn.Cell):
    def __init__(self, spec_channels, z_latent_dim, emb_dim):
        super().__init__()
        self.ref_encoder = MelStyleEncoder(spec_channels, style_vector_dim=emb_dim)
        self.fc1 = nn.Dense(emb_dim, z_latent_dim)
        self.fc2 = nn.Dense(emb_dim, z_latent_dim)
        self.fc3 = nn.Dense(z_latent_dim, emb_dim)
        self.z_latent_dim = z_latent_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = ops.exp(0.5 * logvar)
            eps = ops.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def construct(self, inputs, mask=None):
        enc_out = self.ref_encoder(inputs.squeeze(-1), mask).squeeze(-1)
        mu = self.fc1(enc_out)
        logvar = self.fc2(enc_out)
        posterior = D.Normal(mu, ops.exp(logvar))
        tmp=D.Normal(ops.zeros_like(mu), ops.ones_like(logvar))
        loss_kl = posterior.kl_loss('Normal',[tmp.mean,tmp.sd]).mean() #可能会出错，如果输出直接是mean后的结果
        #loss_kl = kl_divergence.mean()

        z = posterior.rsample()
        style_embed = self.fc3(z)

        return style_embed.unsqueeze(-1), loss_kl

    def infer(self, inputs=None, random_sample=False, manual_latent=None):
        if manual_latent is None:
            if random_sample:
                posterior = D.Normal(
                    ops.zeros([1, self.z_latent_dim]),
                    ops.ones([1, self.z_latent_dim]),
                )
                z = posterior.rsample()
            else:
                enc_out = self.ref_encoder(inputs.swapaxes(1, 2))
                mu = self.fc1(enc_out)
                z = mu
        else:
            z = manual_latent
        style_embed = self.fc3(z)
        return style_embed.unsqueeze(-1), z


class ActNorm(nn.Cell):
    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = Parameter(ops.zeros([1, channels, 1]))
        self.bias = Parameter(ops.zeros([1, channels, 1]))

    def construct(self, x, x_mask=None, g=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = ops.ones([x.shape(0), 1, x.shape(2)]).to(
                 dtype=x.dtype
            )
        x_len = ops.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * ops.exp(-self.logs) * x_mask
            logdet = None
            return z
        else:
            z = (self.bias + ops.exp(self.logs) * x) * x_mask
            logdet = ops.sum(self.logs) * x_len  # [b]
            return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        denom = ops.stop_gradient(ops.sum(x_mask, [0, 2]))
        m = ops.stop_gradient(ops.sum(x * x_mask, [0, 2]) / denom)
        m_sq = ops.stop_gradient(ops.sum(x * x * x_mask, [0, 2]) / denom)
        v = ops.stop_gradient(m_sq - (m**2))
        logs = ops.stop_gradient(0.5 * ops.log(ops.clamp(v, min=1e-6)))

        bias_init = ops.stop_gradient(
            (-m * ops.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
        )
        logs_init = ops.stop_gradient((-logs).view(*self.logs.shape).to(dtype=self.logs.dtype))

        self.bias.data=ops.stop_gradient(bias_init)
        self.logs.data=ops.stop_gradient(logs_init)


class InvConvNear(nn.Cell):
    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian

        w_init = ops.geqrf(
            ms.Tensor(self.n_split, self.n_split,init=Normal()).init_data()
        )[0]
        if ops.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = Parameter(w_init)

    def construct(self, x, x_mask=None, g=None, reverse=False, **kwargs):
        b, c, t = x.shape
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = ops.ones((b,), dtype=x.dtype) * t
        else:
            x_len = ops.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = (
            x.permute(0, 1, 3, 2, 4)
            .view(b, self.n_split, c // self.n_split, t)
        )

        if reverse:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = ops.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = ops.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = ops.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).view(b, c, t) * x_mask
        if reverse:
            return z
        else:
            return z, logdet

    def store_inverse(self):
        self.weight_inv = ops.inverse(self.weight.float()).to(dtype=self.weight.dtype)
