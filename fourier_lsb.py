import numpy as np
from scipy.fft import fftn, ifftn
from utils import str_to_binary, binary_to_str
from utils import float_to_binary, binary_to_float


# not working

def encode_lsb_fourier(image, message):
    """
    Encodes a message into the frequency domain of an image.

    :param image: (ndarray) cover image
    :param message: (str) message
    :return: (ndarray) stego image
    """

    message += '<EOS>'

    bits = str_to_binary(message)
    nbits = len(bits)

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    nrows, ncols, nchannels = image.shape

    dft = fftn(image)

    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchannels):
                pos = 2 * (ncols * i + nchannels * j) + c
                if pos < nbits:
                    b_real = float_to_binary(dft[i, j, c].real)
                    b_real.set(bits[pos], -1)

                    b_imag = float_to_binary(dft[i, j, c].imag)
                    if pos + 1 < nbits:
                        b_imag.set(bits[pos+1], -1)

                    dft[i, j, c] = complex(
                        binary_to_float(b_real),
                        binary_to_float(b_imag)
                    )
                else:
                    return np.real(ifftn(dft))

    return np.real(ifftn(dft))


def decode_lsb_fourier(image):
    """
    Decodes message from the frequency domain of an image.

    :param image: (ndarray) stego image
    :return: (str) hidden message
    """

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    nrows, ncols, nchannels = image.shape

    dft = fftn(image)

    bits, message = [], ""
    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchannels):
                parts = [dft[i, j, c].real, dft[i, j, c].imag]

                for p, part in enumerate(parts):
                    bits.append(float_to_binary(part)[-1])

                    pos = 2 * (ncols * i + nchannels * j + c) + p
                    if pos % 8 == 7:
                        message += binary_to_str(bits)
                        print(message)
                        bits = []

                    if message[-5:] == '<EOS>':
                        return message[:-5]

    return message[:-5]

