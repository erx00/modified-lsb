import numpy as np
from PIL import Image
import random
import string
import progressbar
import argparse
import os
import re

from lsb import encode_lsb, decode_lsb
from fourier_lsb import encode_lsb_fourier, decode_lsb_fourier
from gabor_lsb import encode_gabor_lsb, decode_gabor_lsb

from metrics import mse, ssim


def get_images(path, n, grayscale=False):
    """
    Gets images from folder.

    :param path: (str) path to folder containing images
    :param n: (int) number of photos to evaluate
    :param grayscale: (bool) if true, converts images to
        grayscale
    :return: (List[ndarray])
    """

    images = []
    for i in range(n):
        if path[-1] != '/':
            path = path + '/'
        image = Image.open(path + str(i) + '.png')
        if grayscale:
            image = image.convert(mode='L')
        images.append(np.array(image))

    return images


def eval_lsb(images, ntrials, max_len=0, output=None,
             checkpoint=False):
    """
    Evaluates the LSB algorithm.

    :param images: (List[ndarray]) images
    :param ntrials: (int) number of trials per image
    :param max_len: (int) maximum message length
    :param output: (str) where to save data in real time
    :param checkpoint: (bool) if true, load previous data
        from output
    :return: (List[float], List[float]) MSE, SSIM
    """

    if not max_len:
        max_len = 1000

    mses, ssims = [], []

    if checkpoint:
        with open(output, 'r') as f:
            data = f.readlines()

        for datum in data:
            m = re.search(r'mse = (?P<mse>\S+) \| ssim = (?P<ssim>\S+)', datum)

            if m:
                mses.append(float(m.group('mse')))
                ssims.append(float(m.group('ssim')))

        if output:
            output = open(output, 'a')
    else:
        if output:
            output = open(output, 'w')

    so_far = len(mses)

    progress = progressbar.ProgressBar(
        maxval=len(images) - so_far,
        widgets=['progress',
                 progressbar.Bar('=', '[', ']'),
                 ' ',
                 progressbar.Percentage(),
                 ' ',
                 progressbar.ETA()]
    )
    progress.start()

    for i in range(so_far, len(images)):
        mse_, ssim_ = 0, 0
        for _ in range(ntrials):
            message = ''.join(random.choice(
                string.ascii_letters + string.digits
                + string.punctuation + ' '
            ) for _ in range(max_len))

            if len(images[i].shape) == 2:
                image = images[i][0:256, 0:256, None]  # normalization
            else:
                image = images[i][0:256, 0:256, :]
            cover = np.copy(image)

            stego = encode_lsb(image, message)

            assert decode_lsb(stego) == message, \
                'decoded message does not match original'

            mse_ += mse(cover, stego)
            ssim_ += ssim(cover, stego)

        if output:
            print('mse =', mse_ / ntrials, '| ssim =', ssim_ / ntrials,
                  file=output, flush=True)
        mses.append(mse_ / ntrials)
        ssims.append(ssim_ / ntrials)

        progress.update(i - so_far)

    progress.finish()

    return mses, ssims


def eval_lsb_fourier(images, ntrials, nfreq, channel_first=False,
                     lowest_first=False, dim=2, max_len=0,
                     output=None, checkpoint=False):
    """
    Evaluates the LSB algorithm.

    :param images: (List[ndarray]) images
    :param ntrials: (int) number of trials per image
    :param nfreq: (int) number of frequencies to embed
        bits in
    :param channel_first: (bool) if true, then finishes
        embedding in current channel before moving onto
        the next
    :param lowest_first: (bool) if true, then embed bits
        into the lowest nfreq frequencies
    :param dim: (int) dimension of Fourier transform
        (either 2 or 3)
    :param max_len: (int) maximum message length
    :param output: (str) where to save data in real time
    :param checkpoint: (bool) if true, load previous data
        from output
    :return: (List[float], List[float]) MSE, SSIM
    """

    if not max_len:
        max_len = 1000

    mses, ssims = [], []

    if checkpoint:
        with open(output, 'r') as f:
            data = f.readlines()

        for datum in data:
            m = re.search(r'mse = (?P<mse>\S+) \| ssim = (?P<ssim>\S+)', datum)

            if m:
                mses.append(float(m.group('mse')))
                ssims.append(float(m.group('ssim')))

        if output:
            output = open(output, 'a')
    else:
        if output:
            output = open(output, 'w')

    so_far = len(mses)

    progress = progressbar.ProgressBar(
        maxval=len(images) - so_far,
        widgets=['progress',
                 progressbar.Bar('=', '[', ']'),
                 ' ',
                 progressbar.Percentage(),
                 ' ',
                 progressbar.ETA()]
    )
    progress.start()

    for i in range(so_far, len(images)):

        mse_, ssim_ = 0, 0
        for _ in range(ntrials):
            message = ''.join(random.choice(
                string.ascii_letters + string.digits
                + string.punctuation + ' '
            ) for _ in range(max_len))

            if len(images[i].shape) == 2:
                image = images[i][0:256, 0:256, None].astype(float)
            else:
                image = images[i][0:256, 0:256, :].astype(float)
            cover = np.copy(image)

            stego = encode_lsb_fourier(
                image, message, nfreq, channel_first,
                lowest_first, dim
            )

            assert decode_lsb_fourier(
                stego, nfreq, channel_first,
                lowest_first, dim
            ) == message, \
                'decoded message does not match original'

            mse_ += mse(cover, stego)
            ssim_ += ssim(cover, stego)

        if output:
            print('mse =', mse_ / ntrials, '| ssim =', ssim_ / ntrials,
                  file=output, flush=True)
        mses.append(mse_ / ntrials)
        ssims.append(ssim_ / ntrials)

        progress.update(i)

    progress.finish()

    return mses, ssims


def main():
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument('--alg', type=str, choices=['original', 'fourier', 'gabor'])
    parser.add_argument('--nimages', type=int, default=256)
    parser.add_argument('--ntrials', type=int, default=10)
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    parser.add_argument('--msg_len', type=int, default=1000)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')

    # settings for fourier lsb
    parser.add_argument('--nfreq', type=int, default=3)
    parser.add_argument('--channel_first', default='channel_first',
                        action='store_true')
    parser.add_argument('--lowest_first', dest='lowest_first', action='store_true')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3])

    parser.set_defaults(grayscale=False, channel_first=False, lowest_first=False,
                        checkpoint=False)

    args = parser.parse_args()

    images = get_images('images', args.nimages, args.grayscale)

    if args.alg == 'original':
        mses, ssims = eval_lsb(
            images, args.ntrials, args.msg_len,
            args.output_path, args.checkpoint
        )
    elif args.alg == 'fourier':
        mses, ssims = eval_lsb_fourier(
            images, args.ntrials, args.nfreq, args.channel_first,
            args.lowest_first, args.dim, args.msg_len,
            args.output_path, args.checkpoint
        )


if __name__ == '__main__':
    main()


