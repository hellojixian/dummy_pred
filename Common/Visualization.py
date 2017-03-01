from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
import Common.config as config


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
    return


def animate_data3d(data_2d_list, labels=[], stock_code=None):
    # transforming data
    X = range(data_2d_list[0].shape[1])
    Y = range(data_2d_list[0].shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = data_2d_list[0]
    Z_MIN, Z_MAX = -6, 6

    # declare the plot
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 10)

    ax = fig.add_subplot(gs[0, :6], projection='3d')
    ax.view_init(elev=30, azim=-70)
    ax.set_zlim(-10, 10)
    ax.set_xlim(0, data_2d_list[0].shape[1] - 1)
    ax.set_ylim(data_2d_list[0].shape[0] - 1, 0)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.autoscale(enable=False)

    ax2 = fig.add_subplot(gs[0, 6:])
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Timestamp')
    # ax2.set_ylim(data_2d_list[0].shape[0], 0)
    img2d = ax2.imshow(Z,
                       interpolation='bilinear',
                       cmap=cm.jet,
                       vmax=1.2,
                       vmin=-1.2,
                       animated=True)

    ax3 = fig.add_subplot(gs[1, :6], projection='3d')
    ax3.view_init(elev=4, azim=-90)
    ax3.set_zlim(Z_MIN, Z_MAX)
    ax3.autoscale(enable=False)

    ax4 = fig.add_subplot(gs[1, 5:], projection='3d')
    ax4.view_init(elev=4, azim=0)
    ax4.set_zlim(Z_MIN, Z_MAX)
    ax4.autoscale(enable=False)


    def update(i):
        Z = data_2d_list[i]
        # if i == 8:
        #     p=pd.DataFrame(Z)
        #     print(p)
        ax.clear()
        ax3.clear()
        ax4.clear()

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet, vmax=1.2, vmin=-1.2, )
        ax.contourf(X, Y, Z, zdir='z', offset=Z_MIN, cmap=plt.cm.jet, alpha=0.5)
        ax.set_zlim(Z_MIN, Z_MAX)
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        ax.set_xlabel('Features')
        ax.set_ylabel('Timestamp')
        ax.set_zlabel('Values')

        ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet, vmax=1.2, vmin=-1.2, )
        ax3.contourf(X, Y, Z, zdir='z', offset=Z_MIN, cmap=plt.cm.jet, alpha=0.5)
        ax3.set_zlim(Z_MIN, Z_MAX)
        ax3.autoscale(enable=False)
        ax3.set_aspect('equal')
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Timestamp')
        ax3.set_zlabel('Values')

        ax4.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet, vmax=1.2, vmin=-1.2, )
        ax4.contourf(X, Y, Z, zdir='z', offset=Z_MIN, cmap=plt.cm.jet, alpha=0.5)
        ax4.set_zlim(Z_MIN, Z_MAX)
        ax4.autoscale(enable=False)
        ax4.set_aspect('equal')
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Timestamp')
        ax4.set_zlabel('Values')

        img2d.set_data(Z)
        plt.suptitle("CODE: {0} - {1}".format(str(stock_code), str(i)))
        return

    ani = animation.FuncAnimation(fig, update, frames=data_2d_list.shape[0],
                                  interval=1000, blit=False)

    plt.subplots_adjust(hspace=0.001,
                        wspace=0.01,
                        left=0.00,
                        right=0.96,
                        top=1,
                        bottom=0)
    fig.subplots_adjust(hspace=0.001,
                        wspace=0.01,
                        left=0.00,
                        right=0.96,
                        top=1,
                        bottom=0)

    if stock_code is not None:
        fpath = os.path.join(config.PROJECT_ROOT,
                             "DataVisualized",
                             stock_code + '-3d.mp4')
        ani.save(fpath, fps=1)


    plt.show()
    return
