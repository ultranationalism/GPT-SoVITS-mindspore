from typing import Any, Optional, TypeVar

import mindspore as ms
from mindspore import nn,ops,Parameter
from mindspore.common.initializer import initializer

#TODO:The following is a temporary spectrally_norm method that may be deleted in the future.Parallel is not currently supported
"""
TODO:_register_state_dict_hook and _register_load_state_dict_pre_hook not currently supported in ms2.2
    So when loading and saving the model, you need to manually use these two functions
"""
class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.normalize=ops.L2Normalize()

    def reshape_weight_to_matrix(self, weight: ms.Tensor) -> ms.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.ndimension()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: nn.Cell, do_power_iteration: bool) -> ms.Tensor:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = ops.stop_gradient(self.normalize(ops.mv(weight_mat.t(), u), axis=0, epsilon =self.eps))
                u = ops.stop_gradient(self.normalize(ops.mv(weight_mat, v), axis=0, epsilon =self.eps))
            if self.n_power_iterations > 0:
                # See above on why we need to clone
                u = ops.stop_gradient(u.clone())
                v = ops.stop_gradient(v.clone())

        sigma = ops.dot(u, ops.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    """
    def remove(self, module: nn.Cell) -> None:
        weight = ops.stop_gradient(self.compute_weight(module, do_power_iteration=False))
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))
    """

    def __call__(self, module: nn.Cell, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        #v = torch.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
        weight_mat_pinv = ops.pinv(weight_mat.t().mm(weight_mat))
        v=ops.matmul((ops.matmul(weight_mat_pinv, weight_mat.t())),u.unsqueeze(1)).squeeze(1)
        return ops.mul(v,(target_sigma / ops.tensor_dot(u, ops.mv(weight_mat, v),axes=1)))

    @staticmethod
    def apply(module: nn.Cell, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module.parameters_dict()[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        
        normalize=ops.L2Normalize()
        weight_mat = fn.reshape_weight_to_matrix(weight)

        h, w = weight_mat.size()
        # randomly initialize `u` and `v`
        u = ops.stop_gradient(normalize(initializer('normal', shape=h, dtype=weight.dtype), axis =0, epsilon=fn.eps))
        v = ops.stop_gradient(normalize(initializer('normal', shape=w, dtype=weight.dtype), axis =0, epsilon=fn.eps))

        delattr(module, fn.name)
        new=Parameter(weight,name=fn.name + "_orig")
        setattr(module,fn.name + "_orig",new)
        #module.register_parameter(fn.name + "_orig", weight)
        
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.

        #module.register_buffer(fn.name + "_u", u)
        new_u=Parameter(u,name=fn.name + "_u")
        setattr(module,fn.name + "_u",new_u)
        #module.register_buffer(fn.name + "_v", v)
        new_v=Parameter(v,name=fn.name + "_v")
        setattr(module,fn.name + "_v",new_v)

        module.register_forward_pre_hook(fn)
        #module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        #module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

def SpectralNormLoadStateDictPre(fn,state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    weight_key = prefix + fn.name
    for suffix in ('_orig', '', '_u'):
        key = weight_key + suffix
    weight_orig = state_dict[weight_key + '_orig']
    weight = state_dict.pop(weight_key)
    sigma = (weight_orig / weight).mean()
    weight_mat = fn.reshape_weight_to_matrix(weight_orig)
    u = state_dict[weight_key + '_u']
    v = fn._solve_v_and_rescale(weight_mat, u, sigma)
    state_dict[weight_key + '_v'] = v

T_module = TypeVar('T_module', bound=nn.Cell)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (nn.Conv1dTranspose,
                               nn.Conv2dTranspose,
                               nn.Conv3dTranspose)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module
