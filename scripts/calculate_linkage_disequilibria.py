import numpy
import sys
import os
import random
import math

import pickle
from itertools import combinations
from collections import Counter
import data_utils
import diversity_utils
import config



numpy.random.seed(123456789)
random.seed(123456789)



ld_counts_dict_path = config.data_directory + 'ld_counts_dict_all/%s.pkl'


min_sample_size = config.between_host_min_sample_size
min_ld_sample_size = config.between_host_ld_min_sample_size


def build_ld_counts_dict(votu, max_fraction_nan=0.05, max_d=1e3):
    
    sys.stderr.write("Loading allele counts for %s...\n" % votu)
    allele_counts_map = pickle.load(open(data_utils.allele_counts_map_path % votu, "rb"))
    sys.stderr.write("Filtering allele counts...\n")
    allele_counts_map_filtered = data_utils.filter_allele_counts_map(allele_counts_map,  max_fraction_nan=max_fraction_nan, min_sample_size=min_sample_size, only_biallelic=True)
    
    if len(allele_counts_map_filtered) == 0:
        sys.stderr.write("Insufficient # sites! Skipping...\n")
    
    else:
        # get sites with not too many NaNs
        sites_final = list(allele_counts_map_filtered['aligned_sites'].keys())
        sites_final.sort()
        
        
        sys.stderr.write("Calculating LD for...\n")
        
        
        ld_count_dict = {}
        ld_count_dict['data'] = {}
        ld_count_dict['genomes'] = allele_counts_map_filtered['genomes']
        for variant_type in data_utils.variant_types + ['all']:
            ld_count_dict['data'][variant_type] = {}
            #ld_count_dict['data'][variant_type]['site_pairs'] = []
            #ld_count_dict['data'][variant_type]['ns'] = []
            #ld_count_dict['data'][variant_type]['n11s'] = []
            #ld_count_dict['data'][variant_type]['n10s'] = []
            #ld_count_dict['data'][variant_type]['n01s'] = []
            #ld_count_dict['data'][variant_type]['n00s'] = []
            

        
        #for site_pair_idx, site_pair in enumerate(sites_final_pairs):
        n_pairs_processed = 0
        #n_pairs_processed_var = 0
        for site_1_idx in range(len(sites_final)):
            
            for site_2_idx in range(site_1_idx):
                    
                #s_1 = site_pair[0]
                #s_2 = site_pair[1]
                
                s_1 = sites_final[site_1_idx]
                s_2 = sites_final[site_2_idx]
                
                #site_pair = (s_1, s_2)
                
                dist_12 = int(abs(s_1 - s_2))
                
                if dist_12 >= max_d:
                    continue
                
                if (n_pairs_processed % 100000 == 0) and (n_pairs_processed > 0):                
                    sys.stderr.write("%d site pairs processed...\n" % n_pairs_processed)  
                
                # genomes with nucleotides in both sites
                no_nan_bool_idx_1 = allele_counts_map_filtered['aligned_sites'][s_1]['no_nan_bool_idx']
                no_nan_bool_idx_2 = allele_counts_map_filtered['aligned_sites'][s_2]['no_nan_bool_idx']
                no_nan_bool_idx_inter = no_nan_bool_idx_1 * no_nan_bool_idx_2
                
                #n = sum(no_nan_bool_idx_inter)
                
                allele_bool_idx_1 = allele_counts_map_filtered['aligned_sites'][s_1]['allele_bool_idx']
                allele_bool_idx_2 = allele_counts_map_filtered['aligned_sites'][s_2]['allele_bool_idx']

                allele_bool_idx_final_1 = allele_bool_idx_1[no_nan_bool_idx_inter]
                allele_bool_idx_final_2 = allele_bool_idx_2[no_nan_bool_idx_inter]
                
                n11 = sum(allele_bool_idx_final_1*allele_bool_idx_final_2)
                n10 = sum(allele_bool_idx_final_1*(~allele_bool_idx_final_2))
                n01 = sum((~allele_bool_idx_final_1)*allele_bool_idx_final_2)
                n00 = sum((~allele_bool_idx_final_1)*(~allele_bool_idx_final_2))
                
                rsquared_numerators, rsquared_denominators = diversity_utils.calculate_unbiased_sigmasquared(n11, n10, n01, n00)

                #print(rsquared_denominators)
                
                
                if dist_12 not in ld_count_dict['data']['all']:
                    ld_count_dict['data']['all'][dist_12] = {}
                    ld_count_dict['data']['all'][dist_12]['rsquared_numerators'] = 0
                    ld_count_dict['data']['all'][dist_12]['rsquared_denominators'] = 0
                    ld_count_dict['data']['all'][dist_12]['n_site_pairs'] = 0
                    ld_count_dict['data']['all'][dist_12]['n_site_pair_observations'] = 0

                
                ld_count_dict['data']['all'][dist_12]['rsquared_numerators'] += rsquared_numerators
                ld_count_dict['data']['all'][dist_12]['rsquared_denominators'] += rsquared_denominators
                ld_count_dict['data']['all'][dist_12]['n_site_pairs'] += 1
                ld_count_dict['data']['all'][dist_12]['n_site_pair_observations'] += len(allele_bool_idx_final_2)

            
                #ld_count_dict['data']['all']['site_pairs'].append(site_pair)
                #ld_count_dict['data']['all']['ns'].append(n)
                #ld_count_dict['data']['all']['n11s'].append(n11)
                #ld_count_dict['data']['all']['n10s'].append(n10)
                #ld_count_dict['data']['all']['n01s'].append(n01)
                #ld_count_dict['data']['all']['n00s'].append(n00)
            
                fourfold_status_1 = allele_counts_map_filtered['aligned_sites'][s_1]['fourfold_status']
                fourfold_status_2 = allele_counts_map_filtered['aligned_sites'][s_2]['fourfold_status']
                            
                for variant_type in data_utils.variant_types:
                    # check whether both sites have trhe same fourfold status in each genome
                    variant_type_idx = (fourfold_status_1 == variant_type) & (fourfold_status_2 == variant_type)
                    
                    # no genomes that have the same variant type in both sites
                    if sum(variant_type_idx) == 0:
                        continue
                    
                    #n_var = sum(no_nan_bool_idx_inter*variant_type_idx)
                    allele_bool_var_idx_1 = allele_bool_idx_1[no_nan_bool_idx_inter*variant_type_idx]
                    allele_bool_var_idx_2 = allele_bool_idx_2[no_nan_bool_idx_inter*variant_type_idx]
                                    
                    # make sure the site is biallelic
                    if (sum(allele_bool_var_idx_1) + sum(allele_bool_var_idx_2) == 0) or (sum(allele_bool_var_idx_1) + sum(allele_bool_var_idx_2) == len(allele_bool_var_idx_1) + len(allele_bool_var_idx_2)):
                        continue
                    
                    
                    n11_var = sum(allele_bool_var_idx_1*allele_bool_var_idx_2)
                    n10_var = sum(allele_bool_var_idx_1*(~allele_bool_var_idx_2))
                    n01_var = sum((~allele_bool_var_idx_1)*allele_bool_var_idx_2)
                    n00_var = sum((~allele_bool_var_idx_1)*(~allele_bool_var_idx_2))
                    
                    rsquared_numerators_var, rsquared_denominators_var = diversity_utils.calculate_unbiased_sigmasquared(n11_var, n10_var, n01_var, n00_var)

                    
                    if dist_12 not in ld_count_dict['data'][variant_type]:
                        ld_count_dict['data'][variant_type][dist_12] = {}
                        ld_count_dict['data'][variant_type][dist_12]['rsquared_numerators'] = 0
                        ld_count_dict['data'][variant_type][dist_12]['rsquared_denominators'] = 0
                        ld_count_dict['data'][variant_type][dist_12]['n_site_pairs'] = 0
                        ld_count_dict['data'][variant_type][dist_12]['n_site_pair_observations'] = 0

                    
                    ld_count_dict['data'][variant_type][dist_12]['rsquared_numerators'] += rsquared_numerators_var
                    ld_count_dict['data'][variant_type][dist_12]['rsquared_denominators'] += rsquared_denominators_var
                    ld_count_dict['data'][variant_type][dist_12]['n_site_pairs'] += 1
                    ld_count_dict['data'][variant_type][dist_12]['n_site_pair_observations'] += len(allele_bool_var_idx_2)
    
                    #ld_count_dict['data'][variant_type]['site_pairs'].append(site_pair)
                    #ld_count_dict['data'][variant_type]['ns'].append(n_var)
                    #ld_count_dict['data'][variant_type]['n11s'].append(n11_var)
                    #ld_count_dict['data'][variant_type]['n10s'].append(n10_var)
                    #ld_count_dict['data'][variant_type]['n01s'].append(n01_var)
                    #ld_count_dict['data'][variant_type]['n00s'].append(n00_var)
                    
                    #n_pairs_processed_var += 1
                    
                    
                n_pairs_processed += 1  
            

        sys.stderr.write("Saving dictionary....\n")
        ld_counts_dict_path_ = ld_counts_dict_path % votu
        with open(ld_counts_dict_path_, 'wb') as f:
            pickle.dump(ld_count_dict, f)
        sys.stderr.write("Done!\n")


    





if __name__ == "__main__":

    #votu = 'vOTU-000010'
    #votu_all = [votu]
    #build_allele_counts_map(votu)
    votu_all = data_utils.get_single_votus()
  
    idx_ = votu_all.index('vOTU-000077')
    
    #votu = 'vOTU-000078'
    for votu in votu_all[idx_:]:

        build_ld_counts_dict(votu, max_fraction_nan=0.0)
    
    
