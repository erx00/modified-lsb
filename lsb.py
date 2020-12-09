from utils import str_to_binary, binary_to_str
from utils import int_to_binary, binary_to_int


def encode_lsb(image, message):
    """
    Converts the input message into binary and encodes it
    into the input image using the least significant bit
    algorithm.

    :param image: (ndarray) cover image (supports grayscale
        and RGB)
    :param message: (str) message
    :return: (ndarray) stego image
    """

    message += '<EOS>'

    bits = str_to_binary(message)
    nbits = len(bits)

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    nrows, ncols, nchannels = image.shape

    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchannels):
                pos = ncols * i + nchannels * j + c
                if pos < nbits:
                    b = int_to_binary(image[i, j, c])
                    b[-1] = bits[pos]
                    image[i, j, c] = binary_to_int(b)
                else:
                    return image

    return image


def decode_lsb(image):
    """
    Decodes message from input image using the least
    significant bit algorithm.

    :param image: (ndarray) stego image (supports grayscale
        and RGB)
    :return: (str) message
    """

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    nrows, ncols, nchannels = image.shape

    bits, message = [], ""
    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchannels):
                bits.append(int_to_binary(image[i, j, c])[-1])

                pos = ncols * i + nchannels * j + c
                if pos % 8 == 7:
                    message += binary_to_str(bits)
                    bits = []

                if message[-5:] == '<EOS>':
                    return message[:-5]

    return message
