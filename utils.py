from bitarray import bitarray


def str_to_binary(s):
    """
    Converts input string into binary.

    :param s: (str) input
    :return: (List[int]) list of bits
    """

    bits = bitarray()
    bits.frombytes(s.encode('utf-8'))

    return bits.tolist(True)


def binary_to_str(bits):
    """
    Converts a list of bits

    :param bits: (List[int])
    :return: (str)
    """

    return bitarray(bits).tobytes().decode('utf-8')


def int_to_binary(z):
    """
    Converts an 8-bit integer into a list of bits.

    :param z: (int) 8-bit integer
    :return: (List[int])
    """

    return [int(b) for b in '{:08b}'.format(z)]


def binary_to_int(bits):
    """
    Converts a list of bits into an integer.

    :param bits: (List[int])
    :return: (int)
    """

    z = 0
    for bit in bits:
        z = (z << 1) | bit

    return z
