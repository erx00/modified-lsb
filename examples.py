from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
import string

from lsb import encode_lsb, decode_lsb
from fourier_lsb import encode_lsb_fourier, decode_lsb_fourier
from gabor_lsb import encode_gabor_lsb, decode_gabor_lsb


def normalize(image):
    if np.min(image) >= 0 and np.max(image) <= 255:
        return image / 255.
    if np.min(image) < 0:
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    else:
        return image / np.max(image)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, choices=['original', 'fourier', 'gabor'])
    parser.add_argument('--dir', type=str)
    parser.add_argument('--file', type=str)

    args = parser.parse_args()

    color = Image.open(args.file)
    gray = color.convert(mode='L')

    color = np.array(color).astype(float)
    gray = np.array(gray).astype(float)

    m = ''.join(random.choice(
        string.ascii_letters + string.digits
        + string.punctuation + ' '
    ) for _ in range(2000))

    if args.dir[-1] != '/':
        args.dir += '/'

    if args.alg == 'original':
        stego = encode_lsb(color, m) / 255.
        plt.imsave(args.dir + 'color.png', stego)

        stego = encode_lsb(gray, m) / 255.
        plt.imsave(args.dir + 'gray.png', stego[:, :, None], cmap='gray')
    elif args.alg == 'fourier':
        for i in range(4):
            copy = np.copy(color)
            stego = encode_lsb_fourier(copy, m, i + 1, dim=2)
            plt.imsave(args.dir + str(i+1) + '_dim2_color.png', normalize(stego))

            copy = np.copy(gray)
            stego = encode_lsb_fourier(copy, m, i + 1, dim=2)
            plt.imsave(args.dir + str(i+1) + '_dim2_gray.png', normalize(stego)[:, :, None], cmap='gray')

        for i in range(8):
            copy = np.copy(color)
            stego = encode_lsb_fourier(copy, m, i + 1, dim=3)
            plt.imsave(args.dir + str(i+1) + '_dim3_color.png', normalize(stego))
    elif args.alg == 'gabor':
        for i in range(4):
            for j in range(2):
                copy = np.copy(color)
                stego = encode_gabor_lsb(copy, m, i*90, j)
                blurb = "edge" if not j else "invert"
                plt.imsave(args.dir + str(i*90) + '_' + blurb + '_color.png', normalize(stego))

                copy = np.copy(gray)
                stego = encode_gabor_lsb(copy, m, i*90, j)
                plt.imsave(args.dir + str(i*90) + '_' + blurb + '_gray.png', normalize(stego)[:, :, None], cmap='gray')


if __name__ == '__main__':
    main()
