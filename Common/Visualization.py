from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os, config


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


def animate_data(data_2d_list, labels=[], stock_code=None):
    fig = plt.figure()

    # plt.xticks(np.arange(len(labels)), labels, rotation=90)
    def onpick(event):
        x = event.xdata
        if x:
            x = int(x)
            y = event.ydata
            label = labels[x]
            print("x:{0}\t Column: {1}\t ".format(x, label))

    fig.canvas.mpl_connect('button_press_event', onpick)
    img_list = []
    for i in np.arange(0, len(data_2d_list)):
        df = pd.DataFrame(data_2d_list[i], columns=labels)
        img_list.append(
            [plt.imshow(df,
                        interpolation='bilinear',
                        cmap=cm.jet,
                        vmax=1.2,
                        vmin=-1.2,
                        animated=True)]
        )

    ani = animation.ArtistAnimation(fig, img_list,
                                    interval=1000, blit=True)

    if stock_code is not None:
        fpath = os.path.join(config.PROJECT_ROOT,
                             "DataVisualized",
                             stock_code + '-data_visualized.mp4')
        ani.save(fpath, fps=1)

    plt.subplots_adjust(hspace=0.001,
                        wspace=0.001,
                        left=0.04,
                        right=0.96,
                        top=1,
                        bottom=0)
    plt.show()
