default_path = '../'
import os
os.chdir(default_path)
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import patches
from matplotlib.patches import Patch

from umap import UMAP
from sklearn.decomposition import PCA

from lxg.util import load_pkl, dump_pkl
from lxg.models import DNFClassifier, CoBoT, CoBoTExitCode
import _variables_cobot as _variables

#########

font_size_increase = 0
mpl.rcParams.update({
    'font.size':       12+font_size_increase,   # base font sizes
    'axes.titlesize':  14+font_size_increase,
    'axes.labelsize':  12+font_size_increase,
    'xtick.labelsize': 10+font_size_increase,
    'ytick.labelsize': 10+font_size_increase,
    'legend.fontsize': 10+font_size_increase,
})


def plot_cobot(cobot: CoBoT,
              validation_history,
              title_str='', fname=None):
    history = cobot.history


    DISPLAY_CODES = [
        CoBoTExitCode.SUCCESS,
        CoBoTExitCode.ITEMSET_MISSING,
        CoBoTExitCode.NO_BOXES,
        CoBoTExitCode.WRONG_LABEL,
        # CoBoTExitCode.ATTRIBUTION_ERROR
    ]

    EXIT_CODE_COLORS = {
        CoBoTExitCode.ITEMSET_MISSING: "#999999",  # medium gray
        CoBoTExitCode.SUCCESS: "#4daf4a",  # soft green
        CoBoTExitCode.WRONG_LABEL: "#e41a1c",  # rich red
        CoBoTExitCode.NO_BOXES: "#377eb8",  # medium blue
        # CoBoTExitCode.ATTRIBUTION_ERROR: "#984ea3"  # purple, mysterious and judgmental
    }

    if not history:
        print("Wow. Empty history. Nothing to plot.")
        return

    filtered_history = [(t, code) for t, code in history if code in DISPLAY_CODES]
    if not filtered_history:
        print("None of the relevant exit codes were found. Congratulations, I guess?")
        return

    n_success, n_not_success = 0, 0
    success_ratios = []
    for u in cobot.history:
        if u[1] == CoBoTExitCode.SUCCESS:
            n_success += 1
        else:
            n_not_success += 1
        success_ratios.append(n_success / (n_success + n_not_success))

    exit_codes = [code for _, code in filtered_history]
    counts = Counter(exit_codes)
    bar_values = [counts.get(code, 0) for code in DISPLAY_CODES]

    fig, axs = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True)

    # Global distribution bar plot
    axs[0].bar(
        [code.name for code in DISPLAY_CODES],
        bar_values,
        color=[EXIT_CODE_COLORS[code] for code in DISPLAY_CODES]
    )
    fig.suptitle(f'{len(cobot.samples)} samples | {len(cobot.history_compression)} updates | {title_str}')
    axs[0].set_title("Global Exit Code Distribution")
    axs[0].set_ylabel("Count")

    # Timeline bar plot
    timestamps, codes = zip(*filtered_history)
    colors = [EXIT_CODE_COLORS[code] for code in codes]
    axs[1].bar(timestamps, [1] * len(timestamps), color=colors, width=1.0)
    axs[1].set_title("Exit Code Over Time")
    axs[1].set_xlabel("Timestamp")
    axs[1].set_yticks([])

    # Manual legend construction
    legend_handles = [
        Patch(color=EXIT_CODE_COLORS[code], label=code.name)
        for code in DISPLAY_CODES
    ]
    axs[1].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

    _update_ticks = [i[0] for i in cobot.history if i[1] not in [CoBoTExitCode.SUCCESS, CoBoTExitCode.ATTRIBUTION_ERROR]]

    axs[2].set_title("misc")
    axs[2].plot(_update_ticks, cobot.history_compression, c='red', label='compression')
    axs[2].plot(_update_ticks, [v['tp'] for v in validation_history
                                ], c='green', label='validation precision')
    axs[2].plot(np.arange(len(success_ratios)), success_ratios, c='purple', label='success rate')
    ax22 = axs[2].twinx()

    ax22.scatter([o[0] for o in cobot.history_n_itemsets],
              [o[1] for o in cobot.history_n_itemsets],
              label='n_itemsets', c='#999999')
    ax22.plot(_update_ticks, cobot.history_n_singletons, label='n_singletons')
    ax22.plot(_update_ticks, cobot.history_n_boxes, label='n_boxes')
    # ax22.plot(_update_ticks, [no-ns for no, ns in zip(cobot.history_n_boxes, cobot.history_n_singletons)], label='n_boxes-n_singeltons')


    ax22.plot([],[], c='red', label='compression [1-(n_boxes/n_samples)]')
    ax22.plot([],[], c='purple', label='success rate')
    ax22.plot([], [], c='green', label='validation precision')
    ax22.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    if fname is not None:
        _png = str(fname[:-4])+'.png'
        plt.savefig(fname, bbox_inches='tight', format='pgf')
        plt.savefig(_png, bbox_inches='tight', format='png')



def plot_cobot_curves(cobot: CoBoT,
                     validation_history,
                     title_str='', fname=None):
    history = cobot.history


    DISPLAY_CODES = [
        CoBoTExitCode.SUCCESS,
        CoBoTExitCode.ITEMSET_MISSING,
        CoBoTExitCode.NO_BOXES,
        CoBoTExitCode.WRONG_LABEL,
        CoBoTExitCode.ATTRIBUTION_ERROR
    ]

    EXIT_CODE_COLORS = {
        CoBoTExitCode.ITEMSET_MISSING: "#999999",  # medium gray
        CoBoTExitCode.SUCCESS: "#4daf4a",  # soft green
        CoBoTExitCode.WRONG_LABEL: "#e41a1c",  # rich red
        CoBoTExitCode.NO_BOXES: "#377eb8",  # medium blue
        CoBoTExitCode.ATTRIBUTION_ERROR: "#984ea3"  # purple,
    }

    filtered_history = [(t, code) for t, code in history if code in DISPLAY_CODES]
    if not filtered_history:
        print("None of the relevant exit codes were found. Congratulations, I guess?")
        return

    n_success, n_not_success = 0, 0
    success_ratios = []
    for u in cobot.history:
        if u[1] == CoBoTExitCode.SUCCESS:
            n_success += 1
        else:
            n_not_success += 1
        success_ratios.append(n_success / (n_success + n_not_success))

    # 1) new figure size, no constrained layout
    fig, axs = plt.subplots(figsize=(15, 7), constrained_layout=False)

    # 2) shove the axes down to make room for legends above
    fig.subplots_adjust(top=0.7, bottom=0.2, left=0.1, right=0.9)
    fig.suptitle(f'{title_str}')

    # data
    _update_ticks = [i[0] for i in cobot.history
                     if i[1] not in [CoBoTExitCode.SUCCESS, CoBoTExitCode.ATTRIBUTION_ERROR]]
    comp = cobot.history_compression
    valp = [v['tp'] for v in validation_history] # actually that's accuracy

    _calc_coverage = lambda outputs: 1-(np.sum([o in [CoBoTExitCode.NO_BOXES, CoBoTExitCode.ITEMSET_MISSING, CoBoTExitCode.ATTRIBUTION_ERROR]
            for o in outputs]) / len(outputs))
    coverage = [_calc_coverage(v['outputs']) for v in validation_history]

    succ_x, succ_y = np.arange(len(success_ratios)), success_ratios

    # main axis plots
    l1, = axs.plot(_update_ticks, comp, c='tab:red', label='Compression')
    l2, = axs.plot(_update_ticks, valp, c='tab:green', label='Validation Accuracy')
    l3, = axs.plot(succ_x, succ_y, c='tab:purple', label='Success Rate')
    lcoverage, = axs.plot(_update_ticks, coverage, c='tab:cyan', label='Coverage')
    axs.set_ylim(0, 1)
    axs.set_xlim(-5, len(cobot.history))

    # axs.scatter(_update_ticks, [0.0]*len(_update_ticks), c='k', marker='|', s=3)

    # twin axis plots
    ax2 = axs.twinx()


    axs.set_xlabel('Number of observations')
    axs.set_ylabel('Ratios')
    ax2.set_ylabel('Counts')

    s1 = ax2.scatter([o[0] for o in cobot.history_n_itemsets],
                     [o[1] for o in cobot.history_n_itemsets],
                     c='tab:gray', label=r'\#Feature Sets')
    l4, = ax2.plot(_update_ticks, cobot.history_n_singletons, c='tab:blue', label='\#Singletons')
    l5, = ax2.plot(_update_ticks, cobot.history_n_boxes, c='tab:orange', label='\#Boxes')

    axs.tick_params(axis='both', labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)

    legend1 = axs.legend(
        handles=[l1, l2, l3, lcoverage],
        fontsize=11,  # bump legend text
        loc='lower left',
        bbox_to_anchor=(0, 1.02),
        ncol=2, frameon=False
    )
    axs.add_artist(legend1)

    legend2 = ax2.legend(
        handles=[s1, l4, l5],
        fontsize=11,  # bump legend text
        loc='lower right',
        bbox_to_anchor=(1, 1.02),
        ncol=2, frameon=False
    )
    ax2.add_artist(legend2)

    if fname is not None:
        _png = str(fname)[:-4]+'.png'
        plt.savefig(fname, bbox_inches='tight', format='pgf',
                    pad_inches=0.1,
                    bbox_extra_artists=(legend1, legend2))
        plt.savefig(_png, bbox_inches='tight', format='png',
                    pad_inches=0.1,
                    bbox_extra_artists=(legend1, legend2))
    plt.show()


def plot_itemset_boxsystem(itemset, boxsystem, samples, labels, x_lim, y_lim, path=None):
    colors = [f"C{label}" for label in labels]
    _eps = 0.05
    if samples.shape[1] != len(itemset):
        samples = samples[:, itemset]
    print(f"itemset = {itemset}")
    fig, axs = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
    def plot_rect_2d(fig, _ax, rect, **kwargs):
        min_size = 1e-2
        try:
            (xmin, xmax), (ymin, ymax) = rect
        except ValueError:
            print(f"value error from rectangle {rect}")
            (xmax, ymax) = rect[0]
            xmin, ymin = xmax+1e-10, ymax+1e-10

        width = xmax - xmin
        height = ymax - ymin
        if width > 1e-10 and height > 1e-10:
            _ax.add_patch(patches.Rectangle(
                (xmin, ymin), width, height,
                **kwargs
            ))
        elif width <= 1e-10 and height <= 1e-10:
            _ax.scatter(xmax-0.5*(xmax-xmin), ymax-0.5*(ymax-ymin), marker='s', **kwargs)
        elif width <= 1e-10:
            _ax.plot([xmin, xmin], [ymin, ymax], linewidth=1, c=kwargs['facecolor'], alpha=kwargs['alpha'])
        elif height <= 1e-10:
            _ax.plot([xmin, xmax], [ymin, ymin], linewidth=1, c=kwargs['facecolor'], alpha=kwargs['alpha'])
        else:
            raise ValueError

    ax = axs
    # if itemset == (3,6):  # cut out for example in paper
    #     print(f"3,6")
    #     x_lim = (-1.4, 1.1)
    #     y_lim = (-1.1, 0.75)
    # else:
    #     x_lim = (x_lim[0] - _eps, x_lim[1]+_eps)
    #     y_lim = (y_lim[0] - _eps, y_lim[1]+_eps)

    x_lim = (x_lim[0] - _eps, x_lim[1] + _eps)
    y_lim = (y_lim[0] - _eps, y_lim[1] + _eps)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    for block, theta, label in boxsystem:
        plot_rect_2d(fig, ax, block, edgecolor='none', facecolor=f"C{label}", alpha=0.2)
    ax.scatter(samples[:, 0], samples[:, 1], marker='.', c=colors, alpha=0.5)
    # fig.suptitle(f"{itemset}")
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
        path = str(path)[:-4]+'.pgf'
        plt.savefig(path, bbox_inches='tight', format='pgf')
        plt.close()
    #plt.show()


def rect_contains(rect, sample):
    lower = [l<=s for l, s in zip(rect.lower_bounds, sample)]
    upper = [s<=l for l, s in zip(rect.upper_bounds, sample)]
    return all(lower) and all(upper)

def get_support_labels(cbx, itemset, support_idxs_itemset):
    samples = []
    labels = []
    boxes = cbx.boxsystems[itemset]
    boxes = [b[0] for b in boxes]
    assert len(cbx.samples) == len(cbx.labels)
    supp_samples = [cbx.samples[i] for i in support_idxs_itemset]
    supp_labels = [cbx.labels[i] for i in support_idxs_itemset]
    for s, l in zip(supp_samples, supp_labels):
        _s = [s[i] for i in itemset]
        if np.any([rect_contains(box, _s) for box in boxes]):
            samples.append(s)
            labels.append(l)
    return samples, labels


def plot_boxsystems(cbxs: list[CoBoT], task: str, expl_method:str):

    final_cbx = cbxs[-1]
    cbxs = cbxs[:-1]

    x_lims, y_lims = {}, {}
    for itemset, boxsystem in final_cbx.boxsystems.items():
        if len(itemset) == 1:
            continue
        _x_lim, _y_lim = (np.inf, -np.inf), (np.inf, -np.inf)
        for b in boxsystem:
            (rect, _, _) = b
            lx, ly = rect.lower_bounds
            ux, uy = rect.upper_bounds
            _x_lim = (min(lx, _x_lim[0]), max(ux, _x_lim[1]))
            _y_lim = (min(ly, _y_lim[0]), max(uy, _y_lim[1]))

        x_lims[itemset] = _x_lim
        y_lims[itemset] = _y_lim

    _boxsystems_history = {k: [] for k in final_cbx.boxsystems.keys()}
    _boxsystems_history_supports = {k: [] for k in final_cbx.boxsystems.keys()}
    for cbx in cbxs:
        for itemset, boxsystem in cbx.boxsystems.items():
            if len(itemset) == 1:
                continue
            if len(_boxsystems_history[itemset]) > 0:
                if _boxsystems_history[itemset][-1] == boxsystem:
                    continue

            _support = cbx.support_idxs[itemset]
            _samples_orig = np.array(cbx.samples)[_support]
            _labels_orig = np.array(cbx.labels)[_support]
            _samples, _labels = get_support_labels(cbx, itemset, _support)
            _samples = np.array(_samples)
            _boxsystems_history_supports[itemset].append((_samples, _labels))
            _boxsystems_history[itemset].append(boxsystem)

    for itemset in _boxsystems_history.keys():
        if len(itemset) != 2:
            continue
        pth = f'./plots/cobot/{task}/42/itemsets/{expl_method}/{itemset}/'
        _variables.__create_dir(Path(pth))
        boxs = _boxsystems_history[itemset]
        data = _boxsystems_history_supports[itemset]
        x_lim = x_lims[itemset]
        y_lim = y_lims[itemset]
        for i, (bx, (d, l)) in enumerate(zip(boxs, data)):
            plot_itemset_boxsystem(itemset, bx, d, l, x_lim, y_lim,
                                   path=Path(pth, f'{i}.png'))


def cbx_tables(pd_results):

    # cbx = row['final_cbx']
    # accuracy = row['validation_history'][-1]
    # n_updates = cbx._n_updates
    # n_itemsets = len(cbx.itemsets)
    # n_boxes = cbx._n_boxes
    # n_singletons = cbx._n_singletons

    # column in dataframe or variable as defined above:
    # 'task', 'binarization', 'ex', accuracy, n_updates, n_itemsets, n_boxes, n_singletons
    # name for latex column:
    # dataset, k, \phi, precision, \#updates, \#itemsets, \#boxes, \#singletons

    # layout: l|c|c||c|c|c|c|c

    def coverage_row(r):
        outputs = r['validation_history'][-1]['outputs']
        missing = 1 - np.sum([o in [CoBoTExitCode.NO_BOXES, CoBoTExitCode.ITEMSET_MISSING, CoBoTExitCode.ATTRIBUTION_ERROR]
                          for o in outputs]) / len(outputs)
        return missing

    rows = []
    for _, row in pd_results.iterrows():
        cbx = row['final_cbx']
        rows.append({
            'dataset':    row['task'],
            r'\localexplainer':      row['ex'],
            'k':          row['binarization'][-1],
            'Acc.':       row['validation_history'][-1]['tp'],
            'Cover.':     coverage_row(row),
            r'\#updates': cbx._n_updates,
            r'\#\itemset':len(cbx.itemsets),
            r'\#\bobox':    cbx._n_boxes,
            r'\#singletons': cbx._n_singletons
        })

    df = pd.DataFrame(rows)

    latex = df.to_latex(
        index=False,
        column_format='l|c|c||c|c|c|c|c',
        escape=False,              # allow the \# and \phi to pass through
        float_format="%.2f"        # format accuracy to 3 decimal places
    )
    print(latex)
    return latex


def cbx_2d_embedding_simple(cbx: CoBoT, task, k, ex, method='pca'):
    samples = cbx.samples
    labels = cbx.labels
    # circle, tri up, square, star, thin X
    _markers = ['o','^', 'x', '*', 's', 'd', '+']
    # colors
    itemset_to_sample_idx = cbx.support_idxs
    itemsets = cbx.itemsets
    itemsets_idx = {i: j for j, i in enumerate(itemsets)}
    n_colors = len(itemsets)
    if len(itemsets) <= 10:
        name = 'tab10'
        cmap = plt.get_cmap(name, n_colors)
    elif len(itemsets) <= 20:
        name = 'tab20'
        base_cmap = plt.get_cmap(name, n_colors)
        colors = base_cmap(np.arange(n_colors))  # shape (n,4) RGBA array

        # build new ordering: all evens first, then all odds
        evens = np.arange(0, n_colors, 2)
        odds = np.arange(1, n_colors, 2)
        new_order = np.concatenate([evens, odds])

        shuffled_colors = colors[new_order]

        # wrap back into a colormap
        cmap = ListedColormap(shuffled_colors, name=f"{name}_shuffled")
    else:
        name = 'gist_rainbow'
        cmap = plt.get_cmap(name, n_colors)

    if method == 'umap':
        U = UMAP(n_components=2, n_neighbors=5, random_state=42)#, metric='sqeuclidean')
        _samples_2d = U.fit_transform(samples)
    elif method == 'pca':
        pca = PCA(n_components=2)
        _samples_2d = pca.fit_transform(samples)
    else:
        raise ValueError(f'method {method} not implemented')

    with mpl.rc_context({
        "axes.labelsize": 14,
        "legend.fontsize": 16,
    }):
        colors = cmap(np.arange(n_colors))
        fig, ax = plt.subplots(figsize=(20, 10))
        alpha = 1.
        for itemset in itemsets:
            idxs = itemset_to_sample_idx[itemset]
            for i in idxs:
                s = _samples_2d[i]
                l = labels[i]
                c = colors[itemsets_idx[itemset]]
                m = _markers[l]
                if m in ['o', '^', 'D', 's']:
                    ax.scatter(s[0], s[1], edgecolors=c, facecolors='none', marker=m, alpha=alpha)
                else:
                    ax.scatter(s[0], s[1], c=c, marker=m, alpha=alpha)

        legend_handles = []
        if n_colors < 20:
            for itemset in itemsets:
                _lh = ax.scatter([], [], c=colors[itemsets_idx[itemset]], label=f'{itemset}')
                legend_handles.append(_lh)
        for l in np.unique(labels):
            if _markers[l] == 'x':
                _lh = ax.scatter([], [], color='k',
                                 label=f'C{l}', marker=_markers[l])

            else:
                _lh = ax.scatter([], [], facecolors='none', edgecolors='k',
                                 label=f'C{l}', marker=_markers[l])
            legend_handles.append(_lh)

        # after fig.subplots_adjust(top=..., bottom=..., etc.)
        legend1 = ax.legend(
            handles=legend_handles,
            fontsize=14,
            loc='lower left',
            frameon=False,
            bbox_to_anchor=(0., 1.02, 1., 0.15),  # (x0, y0, width, height) in axes coords
            mode='expand',
            ncol=int(np.ceil((n_colors + cbx.n_classes) / 2)),
            borderaxespad=0
        )
        ax.add_artist(legend1)

        pth = f'./plots/cobot/{task}/42/simple_{method}'
        _variables.__create_dir(Path(pth))
        name = f'{task}_k={k}_ex={ex}'
        fig.suptitle(f'{task} | ex={ex} |  k={k} | n_itemsets = {n_colors}')
        if legend1 is not None:
            plt.savefig(f'{pth}/{name}.pgf', format='pgf',
                        bbox_inches='tight',
                        bbox_extra_artists=[legend1])
        else:
            plt.savefig(f'{pth}/{name}.pgf', format='pgf',
                        bbox_inches='tight')
        plt.savefig(f'{pth}/{name}.png')#, bbox_inches='tight')
        plt.show()
        plt.close()


def plot_cbx_results(pd_results=None, tasks=['abalone', 'breastw', 'beans']):
    if pd_results is None:
        results = []
        for task in tasks:
            _dir = Path(f'results/cobot/{task}/42/')
            try:
                print(f"loading {task} data")
                p = load_pkl(Path(_dir, f'{task}_cobot_results.pkl'))
            except FileNotFoundError:
                print(f"couldn't find {Path(_dir, f'{task}_cobot_results.pkl')}")
                exit(-1)
            p['task'] = task
            results.append(p)
        pd_results = pd.concat(results).reset_index(drop=True)

    ###### TABLE
    cbx_tables(pd_results)

    ###### UMAP/ PCA
    for idx, row in pd_results.iterrows():
        task = row['task']
        k = row['binarization'][-1]
        ex = row['ex']
        cbx_2d_embedding_simple(row['final_cbx'], task, k, ex, method='umap')

    ###### 2D BOXES
    _cbxs_k2 = pd_results[pd_results['binarization'] == 'top_2']
    for idx, cbx_2 in _cbxs_k2.iterrows():
        cbxs = cbx_2['cbxs']
        t = cbx_2['task']
        expl_method = cbx_2['ex']
        plot_boxsystems(cbxs, t, expl_method)

    ###### CURVES
    for idx, row in pd_results.iterrows():
        task = row['task']
        _plt_dir = f'plots/cobot/{task}/42/'
        _variables.__create_dir(Path(_plt_dir))
        _fname = f"{task}-{row['binarization']}-{row['ex']}.pgf"
        title_str = f"{task} | ex={row['ex']} | k={row['binarization']}"
        plot_cobot_curves(row['final_cbx'], title_str=title_str, validation_history=row['validation_history'],
                  fname=Path(_plt_dir, _fname))
    print(f"done")

if __name__ == '__main__':

    plot_cbx_results()

