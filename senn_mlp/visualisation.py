import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image
import glob


def plot_decision_boundary(pred_func, X, Y, step, save_path):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # predict on the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the contour
    levels = np.linspace(0.0, 1.0, 41)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.contourf(xx, yy, Z, cmap='PRGn', levels=levels)
    cb = fig.colorbar(cax)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='Paired')

    # number of colorbar ticks
    tick_locator = ticker.MaxNLocator(nbins=10)
    cb.locator = tick_locator
    cb.update_ticks()

    # add a title with epoch
    plt.title("Epoch: " + '{:04d}'.format(step))

    # Turn off the grid for this plot
    ax.grid(False)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'decision_boundary_step_' + '{:04d}'.format(step) + '.png'),
                bbox_inches='tight')


def animate_decision_boundary(save_path):
    frames = []
    imgs = glob.glob(save_path + '/*decision_boundary_step*.png')
    sorted_imgs = sorted(imgs)  # I guess the wild card returns unsorted stuff

    for i in sorted_imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(os.path.join(save_path, 'decision_boundary_animation.gif'), format='GIF',
                   append_images=frames[1:], duration=750, save_all=True, loop=0)


def visualize_fisherD(F_logD, save_path):
    ticks_font_size = 10

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(F_logD)
    ax.grid(False)
    plt.xticks(fontsize=ticks_font_size)
    plt.yticks(fontsize=ticks_font_size)
    plt.xlabel('Optimization steps', fontsize=ticks_font_size + 4)
    plt.ylabel('|F|', fontsize=ticks_font_size + 4)

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'fisher_logD.png'), bbox_inches='tight')

    # also save original data to be able to change/reproduce plots later
    np.save(os.path.join(save_path, 'fisher_logD.npy'), np.array(F_logD))
