import matplotlib.pyplot as plt
import numpy as np


def plot_line_chart(all_y, all_model_names):
    x_length = all_y.shape[-1]
    loc, ticks = list(range(x_length)), [224, 240, 260, 300, 380, 456, 528, 600]

    for y, plot_name in zip(all_y, all_model_names):
        y = np.asarray(y)

        plt.plot(loc, y, 'o--', label=plot_name)
        plt.legend()

    plt.xticks(loc, ticks)
    plt.xlabel('Image shape (3, size, size)')
    plt.ylabel('Latency (ms)')

    plt.savefig('bjj.png')
    plt.close()


if __name__ == '__main__':
    all_y = np.asarray([[0.3020110395984375, 0.29990467650350183, 0.29828336329956073,
                         0.2976852484003757, 0.28501639430178327, 0.29033255520043894,
                         0.2927533374997438, 0.28535580430034313],
                        [0.09817643930000486, 0.10615955569810467, 0.10187393239903031,
                         0.11734106359508586, 0.12151471519755433, 0.1079640667012427,
                         0.10361214489967097, 0.1068273856988526]])
    all_model_names = ['SSDlite', 'Faster R-CNN']

    plot_line_chart(all_y, all_model_names)
