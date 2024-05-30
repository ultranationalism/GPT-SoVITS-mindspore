# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/utils.py
# reference: https://github.com/lifeiteng/vall-e
import mindspore as ms
from mindspore import nn,ops
from typing import Tuple
from mindspore.nn.probability.distribution import Exponential

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = ops.arange(max_length, dtype=length.dtype)
    return x.unsqueeze(0) < length.unsqueeze(1)


def make_pad_mask(lengths: ms.Tensor, max_len: int = 0) -> ms.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    #>>> lengths = ms.Tensor([1, 3, 2, 5])
    #>>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.shape[0]
    seq_range = ops.arange(0, max_len)
    expaned_lengths = seq_range.unsqueeze(0).broadcast_to((n, max_len))

    return expaned_lengths >= lengths.unsqueeze(-1)


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < ops.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = ops.sort(logits, descending=True)
        cumulative_probs = ops.cumsum(ops.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = ms.Tensor(sorted_indices_to_remove[..., :-1])
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = ops.multinomial(ops.softmax(logits, axis=-1), num_samples=1)
    return token


from typing import Optional, Tuple


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    exponential_distribution = Exponential(rate=1.0)
    q = exponential_distribution.sample(probs_sort.shape)
    return ops.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=ms.int32)


def logits_to_probs(
    logits,
    previous_tokens: Optional[ms.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    if previous_tokens is not None:
        previous_tokens = previous_tokens.squeeze()
    # print(logits.shape,previous_tokens.shape)
    # pdb.set_trace()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = ops.gather_elements(logits, dim=0, index=previous_tokens)
        score = ops.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits=logits.scatter(axis=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = ops.sort(logits, descending=True)
        cum_probs = ops.cumsum(
            ops.softmax(sorted_logits, axis=-1), axis=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = ops.topk(logits, min(top_k, logits.shape[-1]))
        dim_size = v.shape[-1]
        index = ops.arange(dim_size-1, dim_size)

        pivot = ops.index_select(v, -1, index).unsqueeze(-1)
        logits = ops.where(logits < pivot, -float("Inf"), logits)

    probs = ops.softmax(logits, axis=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[ms.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[ms.Tensor, ms.Tensor]:
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def dpo_loss(policy_chosen_logps: ms.Tensor,
             policy_rejected_logps: ms.Tensor,
             reference_chosen_logps: ms.Tensor,
             reference_rejected_logps: ms.Tensor,
             beta: float,
             reference_free: bool = False) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -ops.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards

def get_batch_logps(logits_target: ms.Tensor, logits_reject: ms.Tensor, labels_target: ms.Tensor, labels_reject: ms.Tensor, average_log_prob: bool = False) -> Tuple[ms.Tensor, ms.Tensor]:

    # dummy token; we'll ignore the losses on these tokens later

    per_token_logps_target = ops.gather_elements(logits_target.log_softmax(-1), dim=2, index=labels_target.unsqueeze(2)).squeeze(2)
    per_token_logps_reject = ops.gather_elements(logits_reject.log_softmax(-1), dim=2, index=labels_reject.unsqueeze(2)).squeeze(2)

    return per_token_logps_target.sum(-1), per_token_logps_reject.sum(-1)

def make_reject_y(y_o, y_lens):
    def repeat_P(y):
        range_idx, _ = ops.randint(0, len(y), size=(2,)).sort()
        pre = y[:range_idx[0]]
        shf = y[range_idx[1]:]
        range_text = y[range_idx[0]:range_idx[1]]
        new_y = ops.cat([pre, range_text, range_text, shf])
        return new_y
    def lost_P(y):
        range_idx, _ = ops.randint(0, len(y), size=(2,)).sort()
        pre = y[:range_idx[0]]
        shf = y[range_idx[1]:]
        range_text = y[range_idx[0]:range_idx[1]]
        new_y = ops.cat([pre, shf])
        return new_y
    bs = len(y_lens)
    reject_y = []
    reject_y_lens = []
    for b in range(bs):
        process_item_idx = ops.randint(0, 1, size=(1, ))[0]
        if process_item_idx == 0:
            new_y = repeat_P(y_o[b])
            reject_y.append(new_y)
            reject_y_lens.append(len(new_y))
        elif process_item_idx==1:
            new_y = lost_P(y_o[b])
            reject_y.append(new_y)
            reject_y_lens.append(len(new_y))
    max_length = max(reject_y_lens)
    for b in range(bs):
        pad_length = max_length - reject_y_lens[b]
        reject_y[b] = ops.cat([reject_y[b], ops.zeros(pad_length, dtype=y_o.dtype)], axis=0)

    reject_y = ops.stack(reject_y, axis = 0)
    reject_y_lens = ms.Tensor(reject_y_lens)

    return reject_y, reject_y_lens
