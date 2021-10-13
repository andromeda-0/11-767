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

    plt.savefig('bjj_detection.png')
    plt.close()


if __name__ == '__main__':
    all_y = np.asarray([[59.3508323,
                         61.75211514,
                         60.5766515,
                         61.01209092,
                         61.40358621,
                         59.47480755,
                         59.14292878,
                         61.64967894
                         ],
                        [60.34500893,
                         60.29418543,
                         61.19365031,
                         59.66429393,
                         59.89195554,
                         60.16866669,
                         60.16818478,
                         60.94942234
                         ]])
    all_model_names = ['Original Faster R-CNN', 'Quantized Faster R-CNN']

    plot_line_chart(all_y, all_model_names)
