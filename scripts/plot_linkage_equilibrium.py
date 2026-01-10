import pickle
import sys
import os

import numpy
import scipy.stats as stats
import scipy.spatial as spatial

import collections
import data_utils
import config

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors


#import calculate_linkage_disequilibria
import calculate_linkage_equilibria
import diversity_utils
import plot_divergence
import plot_utils
import stats_utils




def predict_prob_lambda_nonzero(f0, n, NR):
    return 1 - (1 + n/(2*NR))**(-2*NR*(f0**2))


def plot_le_dist(votu, min_n_site_pairs=80, reference_distance=9, fourfold_status=3, min_dist=100):

    #ld_count_dict['data'][variant_type][dist_12][f0]['rsquared_numerators'] 
    le_counts_dict_path = calculate_linkage_equilibria.le_counts_dict_path % votu

    if os.path.exists(le_counts_dict_path) == False:
        sys.stderr.write("No LD file for %s! Skipping... \n" % votu)
        
    else:
        sys.stderr.write("Plotting LD for %s ....\n" % votu)
        le_counts_dict = pickle.load(open(le_counts_dict_path, "rb"))

        distances = numpy.asarray(list(le_counts_dict['data'][fourfold_status].keys()))
        distances = numpy.sort(distances)

        f0_all = numpy.asarray(list(le_counts_dict['data'][fourfold_status][distances[0]].keys()))
        f0_all = f0_all[f0_all <= 1]
        n = len(le_counts_dict['genomes'])

        lambda_range = numpy.linspace(0, 8, 100)

        f0_range_all = [[0.02, 0.03], [0.1, 0.25]]

        fig, ax = plt.subplots(figsize=(4,4))
        for f0_range in f0_range_all:

            f0_subset = f0_all[(f0_all >= f0_range[0]) & (f0_all <= f0_range[1])]

            lambda_stat_all = []

            for dist in distances[distances >= min_dist]:

                for f0 in f0_subset:

                    rsquared_numerators = le_counts_dict['data'][fourfold_status][dist][f0]['rsquared_numerators'] 
                    rsquared_denominators = le_counts_dict['data'][fourfold_status][dist][f0]['rsquared_denominators'] 

                    if (rsquared_numerators <= 0) or (rsquared_denominators <= 0):
                        continue

                    lambda_stat = rsquared_numerators/rsquared_denominators

                    if lambda_stat > 10:
                        continue

                    lambda_stat_all.append(lambda_stat)
            

            print(min(lambda_stat_all), max(lambda_stat_all))
            
            survival_array = stats_utils.make_survival_dist(numpy.asarray(lambda_stat_all), lambda_range, probability=True)
            
            #label = r'$f_{0} \in $'
            label = rf"$f_{0} \in  [ {f0_range[0]},\ {f0_range[1]} ]$"

            ax.plot(lambda_range, survival_array, lw=2, ls='-', label=label)


        ax.legend(loc='upper right')

        max_x = 4
        ax.set_xlim([0, max_x])
        y_min = min(survival_array[lambda_range <= max_x])

        ax.set_ylim([y_min,1])
        ax.set_yscale('log', base=10)
        
        title = r'$\ell \in [10^{2}, 10^{3}]$' + rf'$, \; n = {n}$'
        ax.set_title(title, fontsize=12)

        #ax.set_xlabel('Measured ' + r'$\Lambda  |_{n_{A}, n_{B} \sim n f_{0}}$', fontsize=12)
        ax.set_xlabel('Measured ' + r'$\Lambda$', fontsize=12)
        ax.set_ylabel('Fraction ' + r'$\geq \Lambda $', fontsize=12)

        fig.subplots_adjust(hspace=0.15, wspace=0.15)
        fig_name = "%sle_dist/%s.png" % (config.analysis_directory, votu)
        fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()




def plot_prob_lambda_nonzero(votu, min_n_site_pairs=80, reference_distance=9, fourfold_status=3, min_dist=100, max_lambda=10):

    le_counts_dict_path = calculate_linkage_equilibria.le_counts_dict_path % votu

    if os.path.exists(le_counts_dict_path) == False:
        sys.stderr.write("No LD file for %s! Skipping... \n" % votu)
        
    else:
        sys.stderr.write("Plotting LD for %s ....\n" % votu)
        le_counts_dict = pickle.load(open(le_counts_dict_path, "rb"))

        distances = numpy.asarray(list(le_counts_dict['data'][fourfold_status].keys()))
        distances = numpy.sort(distances)

        f0_all = numpy.asarray(list(le_counts_dict['data'][fourfold_status][distances[0]].keys()))
        f0_all = f0_all[f0_all <= 1]
        n = len(le_counts_dict['genomes'])

        prob_lambda_nonzero_all = []
        for f0 in f0_all:

            lambda_stat_all = []

            for dist in distances[distances >= min_dist]:

                rsquared_numerators = le_counts_dict['data'][fourfold_status][dist][f0]['rsquared_numerators'] 
                rsquared_denominators = le_counts_dict['data'][fourfold_status][dist][f0]['rsquared_denominators'] 

                if (rsquared_numerators < 0) or (rsquared_denominators <= 0):
                    continue

                lambda_stat = rsquared_numerators/rsquared_denominators

                if lambda_stat > max_lambda:
                    continue

                lambda_stat_all.append(lambda_stat)

            lambda_stat_all = numpy.asarray(lambda_stat_all)

            prob_lambda_nonzero = sum(lambda_stat_all>0)/len(lambda_stat_all)
            prob_lambda_nonzero_all.append(prob_lambda_nonzero)
        


        fig, ax = plt.subplots(figsize=(4,4))
        ax.scatter(f0_all, prob_lambda_nonzero_all, c='#87CEEB')

        # theory
        f0_range = numpy.logspace(numpy.log10(min(f0_all)), 0, 1000)

        NR_all = [0.1, 1, 10]
        ls_all = [':', '--', '-']

        print(max(f0_all))

        for NR_idx, NR in enumerate(NR_all):
            prob_lambda_nonzero_ = predict_prob_lambda_nonzero(f0_range, n, NR)
            ax.plot(f0_range, prob_lambda_nonzero_, lw=2, ls=ls_all[NR_idx], c='k', label='NR = ' + str(NR))

        ax.set_xlim([min(f0_range), 1])
        ax.set_ylim([0,1])
        ax.set_xscale('log', base=10)

        ax.legend(loc='lower left')

        title = r'$\ell \in [10^{2}, 10^{3}]$' + rf'$, \; n = {n}$'
        ax.set_title(title, fontsize=12)

        ax.set_xlabel('Frequency scale, ' + r'$f_{0}$', fontsize=12)
        ax.set_ylabel(r'$P(\Lambda  > 0 | f_{0}, n)$', fontsize=12)

        fig.subplots_adjust(hspace=0.15, wspace=0.15)
        fig_name = "%sprob_lambda_nonzero/%s.png" % (config.analysis_directory, votu)
        fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()




if __name__ == "__main__":

    votu_all = data_utils.get_single_votus()

    votu = 'vOTU-000001'

    #for votu in votu_all:

    #plot_le_dist(votu)

    plot_prob_lambda_nonzero(votu)