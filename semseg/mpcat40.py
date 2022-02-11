import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpcat40 = [
    [0, 'void', np.array([255, 255, 255])],
    [1, 'wall', np.array([174, 199, 232])],
    [2, 'floor', np.array([112, 128, 144])],
    [3, 'chair', np.array([152, 223, 138])],
    [4, 'door', np.array([197, 176, 213])],
    [5, 'table', np.array([255, 127,  14])],
    [6, 'picture', np.array([214,  39,  40])],
    [7, 'cabinet', np.array([ 31, 119, 180])],
    [8, 'cushion', np.array([188, 189,  34])],
    [9, 'window', np.array([255, 152, 150])],
    [10, 'sofa', np.array([ 44, 160,  44])],
    [11, 'bed', np.array([227, 119, 194])],
    [12, 'curtain', np.array([222, 158, 214])],
    [13, 'chest_of_drawers', np.array([148, 103, 189])],
    [14, 'plant', np.array([140, 162,  82])],
    [15, 'sink', np.array([132,  60,  57])],
    [16, 'stairs', np.array([158, 218, 229])],
    [17, 'ceiling', np.array([156, 158, 222])],
    [18, 'toilet', np.array([231, 150, 156])],
    [19, 'stool', np.array([ 99, 121,  57])],
    [20, 'towel', np.array([140,  86,  75])],
    [21, 'mirror', np.array([219, 219, 141])],
    [22, 'tv_monitor', np.array([214,  97, 107])],
    [23, 'shower', np.array([206, 219, 156])],
    [24, 'column', np.array([231, 186,  82])],
    [25, 'bathtub', np.array([ 57,  59, 121])],
    [26, 'counter', np.array([165,  81, 148])],
    [27, 'fireplace', np.array([173,  73,  74])],
    [28, 'lighting', np.array([181, 207, 107])],
    [29, 'beam', np.array([ 82,  84, 163])],
    [30, 'railing', np.array([189, 158,  57])],
    [31, 'shelving', np.array([196, 156, 148])],
    [32, 'blinds', np.array([247, 182, 210])],
    [33, 'gym_equipment', np.array([107, 110, 207])],
    [34, 'seating', np.array([255, 187, 120])],
    [35, 'board_panel', np.array([199, 199, 199])],
    [36, 'furniture', np.array([140, 109,  49])],
    [37, 'appliances', np.array([231, 203, 148])],
    [38, 'clothes', np.array([206, 109, 189])],
    [39, 'objects', np.array([ 23, 190, 207])],
    [40, 'misc', np.array([127, 127, 127])],
    [41, 'unlabeled', np.array([0, 0, 0])]
]

if __name__ == '__main__':
    nrows = 14

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40
    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title('Semantic Colors', fontsize=24, loc='left', pad=10)
    for idx, cat_name, color in mpcat40:
        color = color.astype(np.float) / 255.0
        row = idx % nrows
        col = idx // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, cat_name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                        height=18, facecolor=color, edgecolor='0.7')
        )

    plt.show()

