import matplotlib.pyplot as plt
import numpy as np


def plot_line_chart(all_y, all_model_names):
    x_length = all_y.shape[-1]
    loc, ticks = list(range(x_length)), [1 << i for i in range(x_length)]

    for y, plot_name in zip(all_y, all_model_names):
        y = np.asarray(y)

        plt.plot(loc, y, 'o--', label=plot_name)
        plt.legend()

    plt.xticks(loc, ticks)
    plt.xlabel('batch size')
    plt.ylabel('efficiency metric')

    plt.savefig('bjj.png')
    plt.close()


if __name__ == '__main__':

    all_y = np.asarray([[0.042, 0.047, 0.028, 0.029, 0.038, 0.056], [0.046, 0.040, 0.025, 0.027, 0.09, 1.52]])
    all_model_names = ['resnet_18', 'mobilenets_v2']

    plot_line_chart(all_y, all_model_names)
