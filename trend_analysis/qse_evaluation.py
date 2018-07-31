import numpy as np
import pandas as pd
from trend_analysis.qse_engine import GeneralQSE
from trend_analysis.qse_utils import square_diff, cosine_diff, cross_corr, cross_entropy, find_csv_filenames, tune_bandwidth
from scipy.optimize import minimize
from scipy.stats import spearmanr

from os.path import isfile
import matplotlib.pylab as plt

def information_thres(x, y):
    # ToDo: include a threshold for the spearman correlation if to low, don't use refernce and sofi pair for tuning parameter.
    corr = cross_corr(x, y, col=1)
    return corr > 0.5


def plot_qse(res, res_gt, coeff_nr=3, prim_nr=4, prim=['Flat', 'Down', 'Zero', 'Up'], save_path=None, text=None):
    offset = 2 * coeff_nr + 1
    state_range = np.arange(offset + prim_nr, offset + 2 * prim_nr)
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
        axi.stackplot(nr, *r[:, state_range].T, labels=prim)
        axi.set_ylabel(name +' prob. [-]')
        axi.set_yticks(np.arange(0, 1.5, step=0.5))
        axi.legend(loc='lower right')

    axi.set_xlabel('Sample index [-]')
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    # # old way of printing
    # for i, r in enumerate(state_range):
    #     axi = ax[i+1]
    #     axi.plot(nr, res[:, r], '-', color=colors[i], label='SOFI trend')
    #     axi.plot(nr, res_gt[:, r], '--', color=colors[i], label='reference trend')
    #     axi.set_ylabel('Prob. ' + prim[i])
    #     axi.set_yticks(np.arange(0, 1.5, step=0.5))
    #     axi.legend(loc='lower right')
    # ax[prim_nr].set_xlabel('Sample index [-]')


def plot_presentation():
    import os

    # path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\signal"  # windows
    # path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\02_QualitativeTrendAnalysis\\data"  # windows
    # # path = "/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/02_QualitativeTrendAnanalyis/data"  # mac
    #
    # file_name = 'ft_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv'
    # file_name = 'cam1_intra_0_0.2_0.4__ly4ftr16w2__cam1_0_0.2_0.4.csv'
    # file_path = os.path.join(path, file_name)
    #
    # # load data from csv
    # df = pd.read_csv(file_path, sep=',', dtype={'reference level': np.float64})
    # df = df.interpolate()
    # y_sofi = df['extracted sofi'].values
    # y_sens = df['reference level'].values

    path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\02_QualitativeTrendAnalysis\\data"  # windows
    # path = "/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/02_QualitativeTrendAnanalyis/data"  # mac

    file_name = 'cam1_intra_0_0.2_0.4__ly4ftr16w2__cam1_0_0.2_0.4.csv'
    file_path = os.path.join(path, file_name)

    # load data from csv
    df = pd.read_csv(file_path, sep=',', dtype={'sensor_value': np.float64})
    df = df.interpolate()
    y_sofi = df['flood_index'].values
    y_sens = df['sensor_value'].values

    # setup and initialization with tunning parameters
    epsi = 0.000001
    stay = 1  # how much more probable to stay in same primitive
    trans = [['Q0', 'Q0', 0.50 * stay], ['Q0', 'L', epsi], ['Q0', 'U', 0.50], ['Q0', 'F+', epsi],
             ['L', 'Q0', 0.33], ['L', 'L', 0.33 * stay], ['L', 'U', epsi], ['L', 'F+', 0.33],
             ['U', 'Q0', epsi], ['U', 'L', epsi], ['U', 'U', 0.50 * stay], ['U', 'F+', 0.50],
             ['F+', 'Q0', epsi], ['F+', 'L', 0.33], ['F+', 'U', 0.33], ['F+', 'F+', 0.33 * stay]]

    bw_opt_sens = dict(n_support=80, min_support=40, max_support=400, ici_span=4.4, rel_threshold=0.85, irls=False)
    bw_opt_sofi = dict(n_support=350, min_support=30, max_support=350, ici_span=4.4, rel_threshold=0.85, irls=True)

    qse_sens = GeneralQSE(kernel='tricube', order=3, delta=0.05, transitions=trans, bw_estimation='fix', bw_options=bw_opt_sens)
    qse_sofi = GeneralQSE(kernel='tricube', order=3, delta=0.2, transitions=trans, bw_estimation='fix', bw_options=bw_opt_sofi)

    # run algorithms
    res_sofi = qse_sofi.run(y_sofi)
    res_sens = qse_sens.run(y_sens)

    # calculate difference
    coeff_nr = 3
    prim_nr = 4
    sig_sens = res_sens[:, 1]
    sig_sofi = res_sofi[:, 1]
    feat_sens = res_sens[:, 1 + 2 * coeff_nr + prim_nr:2 * coeff_nr + 2 * prim_nr + 1]
    feat_sofi = res_sofi[:, 1 + 2 * coeff_nr + prim_nr:2 * coeff_nr + 2 * prim_nr + 1]


    ccor = cross_corr(sig_sofi, sig_sens, col=1)
    ce = cross_entropy(feat_sofi, feat_sens)
    square_diff(feat_sofi, feat_sens)
    cosine_diff(feat_sofi, feat_sens, axis=1)
    cosine_diff(feat_sofi, feat_sens, axis=0)

    # plot results
    textstr = 'cross corr$=%.2f$\ncross entropy$=%.2f$' % (ccor, ce)
    plot_qse(res_sofi, res_sens, text=textstr)

    print('finish')


def read_calc_before():
    diff_all = dict(file=[], cosine_diff_0=[], cosine_diff_1=[], sqe_diff=[], correlation_trend=[], correlation_signal=[],
                    correlation_smoothed_signal=[])

    path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\02_QualitativeTrendAnalysis\\data"
    path_store = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\02_QualitativeTrendAnalysis\\results\\results180607"

    # read in files
    if 0:
        for f in find_csv_filenames(path):
            file = path + f
            print('evaluate csv-file: ', f)
            df = pd.read_csv(file, sep=',', dtype={'sensor_value': np.float64})
            df = df.interpolate()

            print(df.corr(method='pearson'))
            epsi = 0.000001
            # d = 0.2 tendency to stay at same signal
            trans2 = [['Q0', 'Q0', 0.50], ['Q0', 'L+', epsi], ['Q0', 'U+', 0.50], ['Q0', 'F+', epsi],
                      ['L+', 'Q0', 0.33], ['L+', 'L+', 0.33], ['L+', 'U+', epsi], ['L+', 'F+', 0.33],
                      ['U+', 'Q0', epsi], ['U+', 'L+', epsi], ['U+', 'U+', 0.50], ['U+', 'F+', 0.50],
                      ['F+', 'Q0', epsi], ['F+', 'L+', 0.33], ['F+', 'U+', 0.33], ['F+', 'F+', 0.33]]


            run_configs = [dict(col='sensor_value', bw=300, trans=trans2, bw_est='ici', bw_tune=False),
                           dict(col='flood_index', bw=300, trans=trans2, bw_est='ici', bw_tune=False)]

            # run_configs = [dict(col='sensor_value', bw=None, trans=trans2, bw_est=True)]
            result = np.full((df.shape[0], 5, len(run_configs)), np.nan)

            for i, rc in enumerate(run_configs):
                new_path = path_store + f.split('.cs')[0] + '-' + rc['col']
                if not isfile(new_path + '.csv'):

                    bw_opt = dict(n_support=rc['bw'], min_support=20, max_support=400, ici_span=4.4, rel_threshold=0.85,
                                  irls=False)
                    qse = GeneralQSE(kernel='tricube', order=3, delta=0.01, transitions=rc['trans'], bw_estimation=rc['bw_est'],
                                     bw_options=bw_opt)

                    signal = df[rc['col']].values

                    if rc['bw_tune']:
                        # scores = []
                        # bws = [5, 10, 15, 20, 40, 60, 100, 120, 160, 200, 250, 300, 400, 500, 600, 700, 800]
                        # for bw in bws:
                        #     scores.append(tune_bandwidth(bw, qse, result[:, :, 0], signal))
                        #     bw_best = bws[np.argmin(scores)]
                        # print('bw won', bw_best)
                        minimum = minimize(tune_bandwidth, np.array([rc['bw']]), method='Nelder-Mead', args=(qse, result[:, :, 0], signal))
                        qse.n_support = int(minimum['x'])  # bws[np.argmin(scores)]
                        qse.delay = float((int(minimum['x']) - 1) / 2)
                    res = qse.run(signal)
                    pd.DataFrame(res).to_csv(new_path + '.csv')
                    qse.plot(res, path=new_path)

    # calculate_difference
    if 0:
        df_pair = dict(sensor_value=[], flood_index=[])
        files = find_csv_filenames(path_store)
        print(files)
        for i, f in enumerate(sorted(files)):
            f_split = f.split('-')
            if len(f_split) > 1:
                if f_split[1] == 'sensor_value.csv':
                    df_pair['sensor_value'].append(f)
                    diff_all['file'].append(f_split[0])
                else:
                    df_pair['flood_index'].append(f)

        for j, _ in enumerate(df_pair['sensor_value']):
            df_sens = pd.read_csv(path_store + df_pair['sensor_value'][j], sep=',')
            df_sofi = pd.read_csv(path_store + df_pair['flood_index'][j], sep=',')

            coeff_nr = 3
            prim_nr = 4

            sig_sens = df_sens.values[:, 1]
            sig_sofi = df_sofi.values[:, 1]

            sig_sm_sens = df_sens.values[:, 2]
            sig_sm_sofi = df_sofi.values[:, 2]

            feat_sens = df_sens.values[:, 1 + 2 * coeff_nr + prim_nr:2 * coeff_nr + 2 * prim_nr + 1]
            feat_sofi = df_sofi.values[:, 1 + 2 * coeff_nr + prim_nr:2 * coeff_nr + 2 * prim_nr + 1]

            plt.figure()
            plt.subplot(3, 2, 1)
            plt.plot(sig_sens, sig_sofi, '.')
            plt.subplot(3, 2, 2)
            plt.plot(sig_sm_sens, sig_sm_sofi, '.')
            plt.subplot(3, 2, 3)
            plt.plot(feat_sens[:, 0]-feat_sofi[:, 0], '-')
            plt.subplot(3, 2, 4)
            plt.plot(feat_sens[:, 1]-feat_sofi[:, 1], '-')
            plt.subplot(3, 2, 5)
            plt.plot(feat_sens[:, 2]-feat_sofi[:, 2], '-')
            plt.subplot(3, 2, 6)
            plt.plot(feat_sens[:, 3]-feat_sofi[:, 3], '-')
            plt.show()

            diff_all['sqe_diff'].append(square_diff(feat_sens, feat_sofi))
            diff_all['cosine_diff_0'].append(cosine_diff(feat_sens, feat_sofi, axis=0))
            diff_all['cosine_diff_1'].append(cosine_diff(feat_sens, feat_sofi, axis=1))
            diff_all['correlation_trend'].append(cross_corr(feat_sens, feat_sofi, col=prim_nr))
            diff_all['correlation_signal'].append(cross_corr(sig_sens, sig_sofi, col=1))
            diff_all['correlation_smoothed_signal'].append(cross_corr(sig_sm_sens, sig_sm_sofi, col=1))

        pd.DataFrame(diff_all).to_csv(path_store + 'correlation_results.csv')


if __name__ == '__main__':
    plot_presentation()
