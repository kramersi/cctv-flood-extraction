import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trend_analysis.qse_engine import GeneralQSE
from trend_analysis.qse_utils import square_diff, cosine_diff, cross_corr, cross_entropy, classification_error


def ref_pred_comparison(y_pred, y_truth, p, store=None, bw_ref=40):
    epsi = 0.000001
    trans = [['Q0', 'Q0', 0.50], ['Q0', 'L', epsi], ['Q0', 'U', 0.50], ['Q0', 'F+', epsi],
                 ['L', 'Q0', 0.33], ['L', 'L', 0.33], ['L', 'U', epsi], ['L', 'F+', 0.33],
                 ['U', 'Q0', epsi], ['U', 'L', epsi], ['U', 'U', 0.50], ['U', 'F+', 0.50],
                 ['F+', 'Q0', epsi], ['F+', 'L', 0.33], ['F+', 'U', 0.33], ['F+', 'F+', 0.33]]

    bw_opt_sens = dict(n_support=bw_ref, min_support=40, max_support=400, ici_span=4.4, rel_threshold=0.85, irls=False)
    qse_sens = GeneralQSE(kernel='tricube', order=3, delta=[0.05, 0.03, 0], sigma_eps='auto', transitions=trans,
                          bw_estimation='fix', bw_options=bw_opt_sens)

    bw_opt_sofi = dict(n_support=p['bw'], min_support=p['min_sup'], max_support=p['max_sup'],
                       ici_span=p['ici'], rel_threshold=p['rel_th'], irls=p['irls'])
    qse_sofi = GeneralQSE(kernel='tricube', order=3, sigma_eps=p['sig_e'], delta=p['delta'], transitions=p['trans'],
                          bw_estimation=p['bw_est'], bw_options=bw_opt_sofi)

    # run algorithms
    res_sofi = qse_sofi.run(y_pred, save_ici_path=store)
    res_sens = qse_sens.run(y_truth)

    # calculate difference
    coeff_nr = qse_sofi.coeff_nr
    prim_nrs = [qse_sofi.prim_nr, qse_sens.prim_nr]
    sig_sens = res_sens[:, 1]
    sig_sofi = res_sofi[:, 1]
    feat_sens = res_sens[:, 1 + 2 * coeff_nr + prim_nrs[1]:2 * coeff_nr + 2 * prim_nrs[1] + 1]
    feat_sofi = res_sofi[:, 1 + 2 * coeff_nr + prim_nrs[0]:2 * coeff_nr + 2 * prim_nrs[0] + 1]

    # ccor = cross_corr(sig_sofi, sig_sens, col=1)
    ce = cross_entropy(feat_sofi, feat_sens)
    ac = classification_error(feat_sofi, feat_sens)
    square_diff(feat_sofi, feat_sens)
    cosine_diff(feat_sofi, feat_sens, axis=1)

    # plot results
    text_str = 'cross entropy$=%.2f$\naccuracy$=%.2f$' % (ce, ac)
    qse_sofi.plot(res_sofi, res_sens, text=text_str, save_path=store, plot_prim_prob=False, plot_bw=True)

    return ce, ac

def create_scenarios(delta=[0.16, 0.05,1], ici=0.2):
    # setup and initialization with tunning parameters
    epsi = 0.000001

    transLU = [['L', 'L', 0.5], ['L', 'U', 0.5],
               ['U', 'L', 0.5], ['U', 'U', 0.5], ['F+', 'F+', 0.0], ['Q0', 'Q0', 0.0]]

    transLUFQ = [['Q0', 'Q0', 0.50], ['Q0', 'L', epsi], ['Q0', 'U', 0.50], ['Q0', 'F+', epsi],
                 ['L', 'Q0', 0.33], ['L', 'L', 0.33], ['L', 'U', epsi], ['L', 'F+', 0.33],
                 ['U', 'Q0', epsi], ['U', 'L', epsi], ['U', 'U', 0.50], ['U', 'F+', 0.50],
                 ['F+', 'Q0', epsi], ['F+', 'L', 0.33], ['F+', 'U', 0.33], ['F+', 'F+', 0.33]]

    stay = 10  # how much more probable to stay in same primitive

    transLUFQ_s = [['Q0', 'Q0', 0.50 * stay], ['Q0', 'L', epsi], ['Q0', 'U', 0.50], ['Q0', 'F+', epsi],
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
    delta_tuned = [0.16, 0.05, 1]
    delta_tuned2 = [0.16, 0.01, 1]
    ici_tuned3 = 0.3
    ici_tuned1 = 0.2
    ici_tuned2= 0.1
    params = {
        '0 Standard': dict(bw=3, min_sup=1, max_sup=1, ici=None, rel_th=None, irls=False, delta=0.0, bw_est='fix',
                    trans=transLU, sig_e=0.01),
        '1 Smoothed': dict(bw=200, min_sup=1, max_sup=1, ici=None, rel_th=None, irls=False, delta=0.0, bw_est='fix',
                    trans=transLU, sig_e=0.01),
        '2 Zero classed': dict(bw=200, min_sup=1, max_sup=1, ici=None, rel_th=None, irls=False, delta=delta, bw_est='fix',
                    trans=transLUFQ, sig_e=0.01),
        '3 Error estimated': dict(bw=200, min_sup=1, max_sup=1, ici=None, rel_th=None, irls=False, delta=delta, bw_est='fix',
                    trans=transLUFQ, sig_e='auto'),
        '4 Bandwidth adapted': dict(bw=200, min_sup=9, max_sup=400, ici=ici, rel_th=0.85, irls=False, delta=delta, bw_est='ici',
                    trans=transLUFQ, sig_e='auto'),
        '5 Outlier weighted': dict(bw=200, min_sup=9, max_sup=400, ici=ici, rel_th=0.85, irls=True, delta=delta, bw_est='ici',
                    trans=transLUFQ, sig_e='auto'),
        '6 Markov transitioned': dict(bw=200, min_sup=9, max_sup=400, ici=ici, rel_th=0.85, irls=True, delta=delta, bw_est='ici',
                    trans=transLUFQ_s, sig_e='auto')
    }

    return params


def bar_plot_results(ce, acc, labels, save_path=None):
    n_vid = len(list(ce.values())[0])  # get lenght of value entries
    n_sc = len(list(ce.keys()))
    ind = np.arange(n_vid)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
    width = 0.12         # the width of the bars

    for i, sc in enumerate(sorted(acc)):
        ax[0].bar(ind + i * width, acc[sc], width, align='edge')
        #ax[0].set_xticks(ind + width / 2. * n_sc)
        #ax[0].set_xticklabels(labels)
        ax[0].get_xaxis().set_ticks([])
        ax[0].set_ylabel('Accuracy [-]')

        ax[1].bar(ind + i * width, ce[sc], width, align='edge')
        ax[1].set_xticks(ind + width / 2 * n_sc)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel('Cross Entropy [-]')

    lgd = ax[0].legend(sorted(acc.keys()), bbox_to_anchor=(1.23, 1.0), borderaxespad=0)

    plt.tight_layout(h_pad=0.1)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'QSEComparison.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plt.show()


if __name__ == '__main__':
    #path = "/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/signal"  # mac
    path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\signal"  # windows
    s_path = "C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\trends_report"

    if not os.path.isdir(s_path):
        os.mkdir(s_path)

    videos = ['AthleticPark', 'FloodXCam1', 'FloodXCam5', 'HoustonGarage', 'HarveyParking', 'BayouBridge']
    models = ['train_test_l5_refaug', 'train_test_l5_aug_reduced', 'train_test_l5_']
    deltas = [[0.1, 0.04, 1], [0.16, 0.01, 1], [0.14, 0.01, 1], [0.16, 0.01, 1], [0.07, 0.03, 1], [0.1, 0.02, 1]]
    icis = [0.2, 0.2, 0.2, 0.2, 0.2, 0.5]
    scs = ['sc4', 'sc4', 'sc4', 'sc4', 'sc4', 'sc5']
    files = [models[0] + '__' + vid + '__signal.csv' for vid in videos]

    # files = ['cam1_intra_0_0.2_0.4__ly4ftr16w2__cam1_0_0.2_0.4.csv',
    #          'ft_l5b3e200f16_dr075i2res_lr__FloodX_cam1__signal.csv',
    #          'ft_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv',
    #          'ft_l5b3e200f16_dr075i2res_lr__HoustonHarveyGarage__signal.csv',
    #          'ft_l5b3e200f16_dr075i2res_lr__ChaskaAthleticPark__signal.csv',
    #          'aug_l5b3e200f16_dr075i2res_lr__FloodX_cam1__signal.csv',
    #          'aug_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv',
    #          'aug_l5b3e200f16_dr075i2res_lr__HoustonHarveyGarage__signal.csv',
    #          'aug_l5b3e200f16_dr075i2res_lr__ChaskaAthleticPark__signal.csv',
    #          'ref_l5b3e200f16_dr075i2res_lr__FloodX_cam1__signal.csv',
    #          'ref_l5b3e200f16_dr075i2res_lr__FloodX_cam5__signal.csv',
    #          'ref_l5b3e200f16_dr075i2res_lr__HoustonHarveyGarage__signal.csv',
    #          'ref_l5b3e200f16_dr075i2res_lr__ChaskaAthleticPark__signal.csv',
    #          ]

    # zero_shift = [0, 0, 0.1, 0,    0.15, 0.14, 0.07, 0,        0.15, 0.27, 0.12, 0]
    #de = [0.16, 0.05, 1]
    #delta1 = [0.2,0.2 , 0.5, 0.5,      0.5, 0.3, 0.1, 0.1,     0.5, 0.3, 0.1, 0.1]
    # bw_ref = [80, 80, 20, 40,         80, 80, 20, 40,         80, 80, 20, 40]

    params = create_scenarios()
    all_ac = {}
    all_ce = {}
    vids = []
    sel_sc = ['4 Bandwidth adapted'] #['0 Standard', '1 Smoothed', '2 Zero classed', '3 Error estimated', '4 Bandwidth adapted', '5 Outlier weighted'] # 'Markov transitioned'
    sel_vid = [2]  # [0, 1]
    for key in params:
        if key in sel_sc:
            all_ac[key] = []
            all_ce[key] = []

    for i, file_name in enumerate(files):  # loop through files
        if i in sel_vid:
            file_path = os.path.join(path, file_name)
            # load data from csv
            df = pd.read_csv(file_path, sep=',', dtype={'reference level': np.float64})
            df = df.interpolate()
            y_sofi = df['extracted sofi'].values
            y_sens = df['reference level'].values

            vids.append(file_name.split('__')[1])

            # construct params
            params = create_scenarios(delta=deltas[i], ici=icis[i])
            # sel_sc = [scs[i]]

            for sc in params:  # loop through scenarios
                if sc in sel_sc:
                    # define figure name
                    store_name = file_name[:-10] + 'trend_' + sc
                    store_path = os.path.join(s_path, store_name)
                    # trend analysis of prediction and reference and calculate differences and plot
                    ce, ac = ref_pred_comparison(y_sofi, y_sens, params[sc], store=store_path, bw_ref=120)
                    all_ac[sc].append(ac)
                    all_ce[sc].append(ce)

    bar_plot_results(all_ce, all_ac, tuple(vids), save_path=s_path)