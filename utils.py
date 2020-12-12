from bitarray import bitarray
import bitstring


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


def int_to_binary(z, nbits=8, signed=False):
    """
    Converts an integer into its binary representation.

    :param z: (int) integer
    :param nbits: (int) number of bits needed
    :param signed: (bool) whether to use signed binary
        representation of integers
    :return: (bitstring.BitArray)
    """

    if signed:
        return bitstring.BitArray(int=z, length=nbits)
    else:
        return bitstring.BitArray(uint=z, length=nbits)


def binary_to_int(b, signed=False):
    """
    Converts a binary representation into an integer.

    :param b: (bitstring.BitArray)
    :param signed: (bool) whether binary
        representation is signed
    :return: (int)
    """

    return b.int if signed else b.uint


def float_to_binary(f, nbits=64):
    """
    Converts a float to its binary representation.

    :param f: (float)
    :param nbits: (int) number of bits used to represent
        the input float
    :return: (bitstring.BitArray)
    """

    return bitstring.BitArray(float=f, length=nbits)


def binary_to_float(b):
    """
    Converts a binary representation to its float.

    :param b: (bitstring.BitArray)
    :return: (float)
    """

    return b.float
