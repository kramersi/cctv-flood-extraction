import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from trend_analysis.qse_engine import GeneralQSE
from trend_analysis.qse_utils import square_diff, cosine_diff, cross_corr, cross_entropy, classification_error


def plot_qse(res, res_gt, coeff_nr, prim_nr, prim, save_path=None, text=None):
    offset = 2 * coeff_nr + 1
    state_ranges = [np.arange(offset + prim_nr[0], offset + 2 * prim_nr[0]), np.arange(offset + prim_nr[1], offset + 2 * prim_nr[1])]
    nr = np.arange(res.shape[0])
    subplot_nr = 3
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    ratios = [2] + [1] * 2

    # plot signal and filtered signal
    fig, ax = plt.subplots(nrows=subplot_nr, ncols=1, sharex=True, figsize=(12, 6),
                           gridspec_kw={'height_ratios': ratios})
    l1 = ax[0].plot(nr, res[:, 0], 'k-', label='raw SOFI')
    l2 = ax[0].plot(nr, res[:, 1], 'g-', label='smoothed SOFI')
    ax[0].set_ylabel('SOFI [-]')

    ax2 = ax[0].twinx()   # add second axis
    l3 = ax2.plot(nr, res_gt[:, 0], 'r--', label='reference signal')
    ax2.set_ylabel('level [cm]', color='r')
    ax2.tick_params('y', colors='r')
    ax3 = ax2.twinx()
    ax3.plot(nr, res[:, -1])

    # add legend
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc='lower right')
    if text is not None:
        ax[0].text(0.8, 0.95, text, transform=ax[0].transAxes, fontsize=12,
                      verticalalignment='top', bbox=dict(facecolor='grey', alpha=0.3))

    # plot states probabilities
    for i, r in enumerate([res, res_gt]):
        name = 'reference' if i==1 else 'SOFI'
        axi = ax[i+1]
        axi.stackplot(nr, *r[:, state_ranges[i]].T, labels=prim[i])
        axi.set_ylabel(name +' prob. [-]')
        axi.set_yticks(np.arange(0, 1.5, step=0.5))
        axi.legend(loc='lower right')

    axi.set_xlabel('Sample index [-]')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

    # # old way of printing
    # for i, r in enumerate(state_range):
    #     axi = ax[i+1]
    #     axi.plot(nr, res[:, r], '-', color=colors[i], label='SOFI trend')
    #     axi.plot(nr, res_gt[:, r], '--', color=colors[i], label='reference trend')
    #     axi.set_ylabel('Prob. ' + prim[i])
    #     axi.set_yticks(np.arange(0, 1.5, step=0.5))
    #     axi.legend(loc='lower right')
    # ax[prim_nr].set_xlabel('Sample index [-]')


def ref_pred_comparison(y_pred, y_truth, p, store=None, bw_ref=10):

    bw_opt_sens = dict(n_support=bw_ref, min_support=40, max_support=400, ici_span=4.4, rel_threshold=0.85, irls=False)
    qse_sens = GeneralQSE(kernel='tricube', order=3, delta=0.05, transitions=p['trans'], bw_estimation='fix',
                          bw_options=bw_opt_sens)

    bw_opt_sofi = dict(n_support=p['bw'], min_support=p['min_sup'], max_support=p['max_sup'],
                       ici_span=p['ici'], rel_threshold=p['rel_th'], irls=p['irls'])
    qse_sofi = GeneralQSE(kernel='tricube', order=3, sigma_eps=p['sig_e'], delta=p['delta'], transitions=p['trans'],
                          bw_estimation=p['bw_est'], bw_options=bw_opt_sofi)

    # run algorithms
    res_sofi = qse_sofi.run(y_pred)
    res_sens = qse_sens.run(y_truth)

    # calculate difference
    coeff_nr = qse_sofi.coeff_nr
    prim_nrs = [qse_sofi.prim_nr, qse_sens.prim_nr]
    prims = [qse_sofi.primitives, qse_sens.primitives]
    sig_sens = res_sens[:, 1]
    sig_sofi = res_sofi[:, 1]
    feat_sens = res_sens[:, 1 + 2 * coeff_nr + prim_nrs[1]:2 * coeff_nr + 2 * prim_nrs[1] + 1]
    feat_sofi = res_sofi[:, 1 + 2 * coeff_nr + prim_nrs[0]:2 * coeff_nr + 2 * prim_nrs[0] + 1]

    # ccor = cross_corr(sig_sofi, sig_sens, col=1)
    ce = cross_entropy(feat_sofi, feat_sens)
    ac = classification_error(feat_sofi, feat_sens)
    square_diff(feat_sofi, feat_sens)
    cosine_diff(feat_sofi, feat_sens, axis=1)
    cosine_diff(feat_sofi, feat_sens, axis=0)

    # plot results
    textstr = 'cross entropy$=%.2f$\naccuracy$=%.2f$' % (ce, ac)
    # qse_sens.plot(res_sens)
    # qse_sofi.plot(res_sofi)
    plot_qse(res_sofi, res_sens, coeff_nr, prim_nrs, prims, text=textstr, save_path=store)

    return ce, ac


if __name__ == '__main__':
    path = "/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/signal"  # mac
    #path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\signal"  # windows

    files = ['cam1_intra_0_0.2_0.4__ly4ftr16w2__cam1_0_0.2_0.4.csv',
             'ft_l5b3e200f16_dr075i2res_lr__FloodX_cam1__signal.csv',
             'ft_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv',
             'ft_l5b3e200f16_dr075i2res_lr__HoustonHarveyGarage__signal.csv',
             'ft_l5b3e200f16_dr075i2res_lr__ChaskaAthleticPark__signal.csv',
             'aug_l5b3e200f16_dr075i2res_lr__FloodX_cam1__signal.csv',
             'aug_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv',
             'aug_l5b3e200f16_dr075i2res_lr__HoustonHarveyGarage__signal.csv',
             'aug_l5b3e200f16_dr075i2res_lr__ChaskaAthleticPark__signal.csv',
             'ref_l5b3e200f16_dr075i2res_lr__FloodX_cam1__signal.csv',
             'ref_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv',
             'ref_l5b3e200f16_dr075i2res_lr__HoustonHarveyGarage__signal.csv',
             'ref_l5b3e200f16_dr075i2res_lr__ChaskaAthleticPark__signal.csv',
             ]

    # zero_shift = [0, 0, 0.1, 0,    0.15, 0.14, 0.07, 0,        0.15, 0.27, 0.12, 0]
    # delta = [0.5, 0.3, 0.1, 0.1,      0.5, 0.3, 0.1, 0.1,     0.5, 0.3, 0.1, 0.1]
    # bw_ref = [80, 80, 20, 40,         80, 80, 20, 40,         80, 80, 20, 40]

    # setup and initialization with tunning parameters
    epsi = 0.000001

    transLU = [['L', 'L', 0.5], ['L', 'U', 0.5],
               ['U', 'L', 0.5], ['U', 'U', 0.5], ['F', 'F', 0.5]]

    transLUF = [['F', 'F', 0.33], ['F', 'L', 0.33], ['F', 'U', 0.33],
                ['L', 'F', 0.5], ['L', 'L', 0.5], ['L', 'U', epsi],
                ['U', 'F', 0.5], ['U', 'L', epsi], ['U', 'U', 0.5]]

    stay = 1  # how much more probable to stay in same primitive
    transLUF_stay = [['F', 'F', 0.33 * stay], ['F', 'L', 0.33], ['F', 'U', 0.33],
                     ['L', 'F', 0.50], ['L', 'L', 0.50 * stay], ['L', 'U', epsi],
                     ['U', 'F', 0.50], ['U', 'L', epsi], ['U', 'U', 0.50 * stay]]


    trans = [['Q0', 'Q0', 0.50 * stay], ['Q0', 'L', epsi], ['Q0', 'U', 0.50], ['Q0', 'F+', epsi],
             ['L', 'Q0', 0.33], ['L', 'L', 0.33 * stay], ['L', 'U', epsi], ['L', 'F+', 0.33],
             ['U', 'Q0', epsi], ['U', 'L', epsi], ['U', 'U', 0.50 * stay], ['U', 'F+', 0.50],
             ['F+', 'Q0', epsi], ['F+', 'L', 0.33], ['F+', 'U', 0.33], ['F+', 'F+', 0.33 * stay]]

    # sc0: tool just up and down and no signal smoothing, standard used in paper without smoothing
    # sc1: + smoothing
    # sc2: + primitive flat + delta
    # sc3: + sigma epsilon est
    # sc4: + adaptive bandwidht
    # sc5: + irls
    # sc6: + stay change in markov state
    params = {
        'sc0': dict(bw=3, min_sup=1, max_sup=1, ici=4.4, rel_th=0.85, irls=False, delta=0.0, bw_est='fix',
                    trans=transLU, sig_e=0.001),
        'sc1': dict(bw=200, min_sup=1, max_sup=1, ici=4.4, rel_th=0.85, irls=False, delta=0.0, bw_est='fix',
                    trans=transLU, sig_e=0.001),
        'sc2': dict(bw=200, min_sup=1, max_sup=1, ici=4.4, rel_th=0.85, irls=False, delta=0.05, bw_est='fix',
                    trans=transLUF, sig_e=0.001),
        'sc3': dict(bw=200, min_sup=1, max_sup=1, ici=4.4, rel_th=0.85, irls=False, delta=0.07, bw_est='fix',
                    trans=transLUF, sig_e='auto'),
        'sc4': dict(bw=200, min_sup=60, max_sup=400, ici=0.2, rel_th=None, irls=False, delta=[0.16, 0.05, 1], bw_est='fix',
                    trans=trans, sig_e='auto'),
        'sc5': dict(bw=200, min_sup=60, max_sup=400, ici=4.4, rel_th=0.85, irls=True, delta=0.07, bw_est='ici',
                    trans=transLUF, sig_e='auto'),
        'sc6': dict(bw=200, min_sup=60, max_sup=400, ici=4.4, rel_th=0.85, irls=True, delta=0.07, bw_est='ici',
                    trans=transLUF_stay, sig_e='auto')
    }

    for i, file_name in enumerate(files):  # loop through files
        if i in [0]:
            file_path = os.path.join(path, file_name)

            # load data from csv
            df = pd.read_csv(file_path, sep=',', dtype={'reference level': np.float64})
            df = df.interpolate()
            y_sofi = df['extracted sofi'].values
            y_sens = df['reference level'].values

            for sc in params:  # loop through scenarios
                if sc in ['sc4']:  # ['sc0', 'sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'sc6']:
                    # define figure name
                    store_name = file_name[:-10] + 'trend_' + sc + '.png'
                    store_path = os.path.join(path, store_name)

                    # trend analyis of prediction and reference and calculate differences and plot
                    ref_pred_comparison(y_sofi, y_sens, params[sc], store=store_path, bw_ref=40)
