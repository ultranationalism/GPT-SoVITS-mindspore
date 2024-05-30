import mindspore as ms
import mindaudio
from mindspore import ops
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return ops.log(ops.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return ops.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if ops.min(y)[0] < -1.0:
        print("min value is ", ops.min(y)[0])
    if ops.max(y)[0] > 1.0:
        print("max value is ", ops.max(y)[0])

    y = ops.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    ynum=y.asnumpy()
    spec = ms.Tensor(mindaudio.stft(
        ynum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hamming",
        center=center,
        pad_mode="reflect",
        return_complex=False,
    ),dtype=y.dtype)

    spec = ops.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" 
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = ms.Tensor(mel).to(
            dtype=spec.dtype
        )
    spec = ops.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if ops.min(y) < -1.0:
        print("min value is ", ops.min(y))
    if ops.max(y) > 1.0:
        print("max value is ", ops.max(y))

    global mel_basis
    dtype_device = str(y.dtype) + "_" 
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = ms.Tensor(mel).to(
            dtype=y.dtype
        )

    y = ops.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    ynum=y.asnumpy()

    spec = ms.Tensor(mindaudio.stft(
        ynum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hamming",
        center=center,
        pad_mode="reflect",
        return_complex=False,
    ),dtype=y.dtype)

    spec = ops.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = ops.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
