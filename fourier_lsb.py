import numpy as np
from scipy.fft import fftn, ifftn
from utils import str_to_binary, binary_to_str
from utils import int_to_binary, binary_to_int


# TODO: get rid of padding

def encode_lsb_fourier(image, message):
    """
    Encodes a message into the frequency domain of an image.

    :param image: (ndarray) cover image (currently only
        supports images with a single channel)
    :param message: (str) message
    :return: (ndarray) stego image
    """

    message += '<EOS>'

    bits = str_to_binary(message)
    nbits = len(bits)

    image = np.pad(image,
                   ((0, image.shape[0] % 2), (0, image.shape[1] % 2)),
                   'constant', constant_values=0)

    nrows, ncols = image.shape

    pos = 0
    for i in range(0, nrows, 2):
        for j in range(0, ncols, 2):
            dft = fftn(image[i:i+2, j:j+2])

            for (x, y) in [(0, 1), (1, 0), (1, 1)]:
                if pos < nbits:
                    b = int_to_binary(int(dft[x, y].real), nbits=64, signed=True)
                    b.set(bits[pos], -1)
                    dft[x, y] = complex(
                        float(binary_to_int(b, signed=True)),
                        dft[x, y].imag
                    )
                    pos += 1

            image[i:i+2, j:j+2] = ifftn(dft).real

            if pos >= nbits:
                return image

    return image


def decode_lsb_fourier(image):
    """
    Decodes message from the frequency domain of an image.

    :param image: (ndarray) stego image
    :return: (str) hidden message
    """

    nrows, ncols = image.shape

    pos = 0
    bits, message = [], ""
    for i in range(0, nrows, 2):
        for j in range(0, ncols, 2):
            dft = fftn(image[i:i+2, j:j+2])

            for (x, y) in [(0, 1), (1, 0), (1, 1)]:
                bits.append(int_to_binary(int(dft[x, y].real), nbits=64, signed=True)[-1])

                if pos % 8 == 7:
                    message += binary_to_str(bits)
                    bits = []

                pos += 1

                if message[-5:] == '<EOS>':
                    return message[:-5]

    return message[:-5]

