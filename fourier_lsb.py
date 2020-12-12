import numpy as np
from scipy.fft import fftn, ifftn
from itertools import product

from utils import str_to_binary, binary_to_str
from utils import int_to_binary, binary_to_int


def encode_lsb_fourier(image, message, nfreq, channel_first=False,
                       lowest_first=False, dim=2):
    """
    Encodes a message into the frequency domain of an image.

    :param image: (ndarray) cover image (supports arbitrary number
        of channels)
    :param message: (str) message
    :param nfreq: (int) number of frequencies to embed bits in
    :param channel_first: (bool) if true, then the message is first
        embedded into one channel before moving onto the next;
        otherwise, the message is embedded into each channel before
        moving to the next row/column
    :param lowest_first: (bool) if true, then bits are encoded into
        the lowest nfreq frequencies
    :param dim: (int) dimension of DFT (either 2 or 3)
    :return: (ndarray) stego image
    """

    message += '<EOS>'

    bits = str_to_binary(message)
    nbits = len(bits)

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    nrows, ncols, nchannels = image.shape

    ys = np.arange(0, nrows - nrows % 2, 2)
    xs = np.arange(0, ncols - ncols % 2, 2)
    cs = np.arange(nchannels)

    if dim == 3:
        cs = cs[:-1:2]

    if channel_first:
        image = image.reshape((nchannels, nrows, ncols))
        indices = product(cs, ys, xs)
    else:
        indices = product(ys, xs, cs)

    pos = 0
    for i, j, k in indices:
        dft = fftn(image[i:i+2, j:j+2, k:k+2])
        shape = dft.shape

        if dft.size == 8:       # sort by Hamming distance
            ls = np.array([0, 1, 2, 4, 3, 5, 6, 7])
        else:                   # dft.size == 4
            ls = np.arange(4)
            dft = dft.reshape((1, 2, 2))

        if lowest_first:
            ls = ls[:nfreq]
        else:
            ls = ls[-nfreq:]

        for l in ls:
            x, y, z = np.array(list(int_to_binary(l, nbits=3).bin)) \
                .astype(int)

            if pos < nbits:
                b = int_to_binary(
                    int(dft[x, y, z].real),
                    nbits=64,
                    signed=True
                )
                b.set(bits[pos], -1)
                dft[x, y, z] = complex(
                    binary_to_int(b, signed=True),
                    dft[x, y, z].imag
                )
                pos += 1

        if dft.shape != shape:
            dft = dft.reshape((2, 2, 1))

        image[i:i+2, j:j+2, k:k+2] = ifftn(dft).real

        if pos >= nbits:
            if channel_first:
                image = image.reshape((nrows, ncols, nchannels))
            return image

    if channel_first:
        image = image.reshape((nrows, ncols, nchannels))
    return image


def decode_lsb_fourier(image, nfreq, channel_first=False,
                       lowest_first=False, dim=2):
    """
    Decodes message from the frequency domain of an image.

    :param image: (ndarray) stego image
    :param nfreq: (int) number of frequencies to embed bits in
    :param channel_first: (bool) if true, then finishes embedding
        in current change before moving onto the next batch
    :param lowest_first: (bool) if true, then embed bits in the
        nfreq lowest frequencies
    :param dim: (int) dimension of DFT (either 2 or 3)
    :return: (str) hidden message
    """

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    nrows, ncols, nchannels = image.shape

    ys = np.arange(0, nrows - nrows % 2, 2)
    xs = np.arange(0, ncols - ncols % 2, 2)
    cs = np.arange(nchannels)

    if dim == 3:
        cs = cs[:-1:2]

    if channel_first:
        image = image.reshape((nchannels, nrows, ncols))
        indices = product(cs, ys, xs)
    else:
        indices = product(ys, xs, cs)

    pos = 0
    bits, message = [], ''
    for i, j, k in indices:
        dft = fftn(image[i:i+2, j:j+2, k:k+2])

        if dft.size == 8:       # sort by Hamming distance
            ls = np.array([0, 1, 2, 4, 3, 5, 6, 7])
        else:                   # dft.size == 4
            ls = np.arange(4)
            dft = dft.reshape((1, 2, 2))

        if lowest_first:
            ls = ls[:nfreq]
        else:
            ls = ls[-nfreq:]

        for l in ls:
            x, y, z = np.array(list(int_to_binary(l, nbits=3).bin)) \
                .astype(int)

            bits.append(int_to_binary(
                int(dft[x, y, z].real),
                nbits=64,
                signed=True
            )[-1])

            if pos % 8 == 7:
                message += binary_to_str(bits)
                bits = []

            pos += 1

            if message[-5:] == '<EOS>':
                return message[:-5]

    return message[:-5]

