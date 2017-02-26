from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np


def visualize_data(data_2d_list):
    plt.figure(num=0)
    index = 1
    for i in range(0, len(data_2d_list)):
        ax = plt.subplot(len(data_2d_list) / 3, 3, index)
        ax.imshow(data_2d_list[i],
                  interpolation='none',
                  cmap=cm.jet,
                  vmax=1,
                  vmin=-1)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        index += 1
    plt.subplots_adjust(hspace=0.001,
                        wspace=0.001,
                        left=0,
                        right=1,
                        top=1,
                        bottom=0)
    plt.show()


def animate_data(data_2d_list):
    fig = plt.figure()

    img_list = []
    for i in np.arange(0, len(data_2d_list)):
        img_list.append(
            [plt.imshow(data_2d_list[i],
                        interpolation='bilinear',
                        cmap=cm.jet,
                        vmax=1.2,
                        vmin=-1.2,
                        animated=True)]
        )

    ani = animation.ArtistAnimation(fig, img_list,
                                    interval=1000, blit=True)

    ani.save('data_visualized.mp4', fps=1)
    plt.subplots_adjust(hspace=0.001,
                        wspace=0.001,
                        left=0.04,
                        right=0.96,
                        top=1,
                        bottom=0)
    plt.show()
