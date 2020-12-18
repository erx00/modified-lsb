import re
import numpy as np
import os
import matplotlib.pyplot as plt
import string
import json
import argparse
from scipy.stats import sem


def get_stats(directory, error):
    """
    Compute statistics of given directory.

    :param directory: (str) directory containing data files
    :param error: (str) whether to use stdev, stderr, or
        95% confidence interval
    :return:
    """

    files = sorted(os.listdir(directory))

    filenames = []
    mu_mses, mu_ssims = [], []
    sigma_mses, sigma_ssims = [], []
    for file in files:
        if 'fourier' in file or 'original' in file:
            filenames.append(file)

            mse_, ssim_ = [], []

            if directory[-1] != '/':
                directory += '/'
            with open(directory + file, 'r') as f:
                data = f.readlines()

            for datum in data:
                m = re.search(r'mse = (?P<mse>\S+) \| ssim = (?P<ssim>\S+)', datum)

                if m:
                    mse_.append(float(m.group('mse')))
                    ssim_.append(float(m.group('ssim')))

            mu_mses.append(np.mean(mse_))
            mu_ssims.append(np.mean(ssim_))

            if error == 'stdev':
                sigma_mses.append(np.std(mse_))
                sigma_ssims.append(np.std(ssim_))
            elif error == 'stderr':
                sigma_mses.append(sem(mse_))
                sigma_ssims.append(sem(ssim_))
            else:           # confidence interval
                sigma_mses.append([np.quantile(mse_, 0.025), np.quantile(mse_, 0.975)])
                sigma_ssims.append([np.quantile(ssim_, 0.025), np.quantile(ssim_, 0.975)])

    return filenames, np.array(mu_mses), np.array(mu_ssims), \
        np.array(sigma_mses), np.array(sigma_ssims)


def get_plot(mus, sigmas, ylabel, title='', path=''):

    # only handles categorical plots with <= 26 data points
    labels = []
    for i in range(len(mus)):
        labels.append(string.ascii_uppercase[i])

    fig, ax = plt.subplots()
    if len(sigmas.shape) == 2:
        ax.bar(range(len(labels)), mus,
               yerr=[mus - sigmas[:, 0], sigmas[:, 1] - mus],
               align='center', alpha=0.5, ecolor='black', capsize=3)
    else:
        ax.bar(range(len(labels)), mus, yerr=sigmas, align='center',
               alpha=0.5, ecolor='black', capsize=3)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    if title:
        ax.set_title(title)

    plt.savefig(path)


def save_data(filenames, mu_mses, mu_ssims, sigma_mses, sigma_ssims,
              path=''):

    data = {}
    for i, filename in enumerate(filenames):
        stats = {
            'mu_mse': mu_mses[i],
            'mu_ssim': mu_ssims[i],
            'sigma_mse': sigma_mses[i] if isinstance(sigma_mses[i], float)
            else list(sigma_mses[i]),
            'sigma_ssim': sigma_ssims[i] if isinstance(sigma_ssims[i], float)
            else list(sigma_ssims[i])
        }

        data[filename] = stats

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str)
    parser.add_argument('--mse_graph', type=str)
    parser.add_argument('--ssim_graph', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--error', type=str, choices=['stdev', 'stderr', 'confint'])

    args = parser.parse_args()

    filenames, mu_mses, mu_ssims, sigma_mses, sigma_ssims = get_stats(args.dir, args.error)

    save_data(filenames, mu_mses, mu_ssims, sigma_mses, sigma_ssims, path=args.data)

    get_plot(mu_mses, sigma_mses, 'MSEs', path=args.mse_graph)
    get_plot(mu_ssims, sigma_ssims, 'SSIMs', path=args.ssim_graph)


if __name__ == '__main__':
    main()
