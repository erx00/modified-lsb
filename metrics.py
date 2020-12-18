import numpy as np
from scipy.signal import convolve


def normalize(image):
    return (image - np.min(image)) / (np.max(image)-np.min(image))


def mse(image1, image2):
    """
    Computes the mean squared error (MSE) between two images.

    :param image1: (ndarray)
    :param image2: (ndarray)
    :return: (float) MSE
    """

    return np.mean(np.square(normalize(image1) - normalize(image2)))


def to_grayscale(image):
    """
    Convert an RGB image to grayscale using the color
    encoding standard Rec. 601

    :param image: (ndarray) RGB image
    :return: (ndarray) grayscale image
    """

    return 0.2989 * image[:, :, 0] \
        + 0.581 * image[:, :, 1] \
        + 0.114 * image[:, :, 2]


def gaussian_kernel(n, sigma):
    """
    Taken from Yale CPSC 475 Homework 3 Problem 5 Part (a).
    Creates a 2D Gaussian function with bandwidth SIGMA and
    size N x N.

    :param n: (int) size of kernel
    :param sigma: (float) bandwidth
    :return: (ndarray) A Gaussian kernel
    """

    rad = (n - 1) / 2.
    xs, ys = np.meshgrid(
        np.linspace(-rad, rad, n),
        np.linspace(rad, -rad, n)
    )

    kernel = np.exp(-1 * (np.square(xs) + np.square(ys))
                    / (2 * sigma**2))

    return kernel / np.sum(kernel)


def ssim(image1, image2):
    """
    Computes the Structural Similarity Index Measure (SSIM)
    between two images.

    :param image1: (ndarray)
    :param image2: (ndarray)
    :return: (float) SSIM
    """

    image1 = normalize(image1)
    image2 = normalize(image2)

    if len(image1.shape) == 2:
        image1 = image1.reshape((image1.shape[0], image1.shape[1], 1))
        image2 = image2.reshape((image1.shape[0], image1.shape[1], 1))

    kernel = gaussian_kernel(11, 1.5)[:, :, None]

    mu_x = convolve(image1, kernel, mode='valid')
    mu_y = convolve(image2, kernel, mode='valid')

    mu_x_2 = np.square(mu_x)
    mu_y_2 = np.square(mu_y)
    mu_x_mu_y = mu_x * mu_y

    var_x = convolve(np.square(image1), kernel, mode='valid') - mu_x_2
    var_y = convolve(np.square(image2), kernel, mode='valid') - mu_y_2

    sigma_xy = convolve(image1 * image2, kernel, mode='valid') - mu_x_mu_y

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    ssim_mat = (2 * mu_x_mu_y + c1) * (2 * sigma_xy + c2) \
        / ((mu_x_2 + mu_y_2 + c1) * (var_x + var_y + c2))

    return np.mean(ssim_mat)

