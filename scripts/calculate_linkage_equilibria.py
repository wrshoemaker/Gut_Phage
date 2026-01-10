import numpy
import sys
import os
import random
import math
import sympy
from scipy.special import perm

import pickle
from itertools import combinations
from collections import Counter
import data_utils
import diversity_utils
import config



numpy.random.seed(123456789)
random.seed(123456789)

# just changing the file name
le_counts_dict_path = config.data_directory + 'le_counts_dict_all/%s.pkl'

min_sample_size = config.between_host_min_sample_size
min_ld_sample_size = config.between_host_ld_min_sample_size

f0s = numpy.hstack([numpy.logspace(-3,-0.5,20),[1e02]])


def factorial_fraction(n, i):
    # calculate n! / (n-i)!
    res = 1
    for i in range(i):
        res *= n - i
    return res


def sterling_factorial(n):
    # return log(n!) using sterling's approximation
    # if n=0, return 0 (because 0! = 1)
    n = numpy.array(n)
    res = numpy.zeros(n.shape)
    res[n<=0] = 0
    n_mask = n[n>0]
    res[n>0] = n_mask * numpy.log(n_mask) - n_mask + numpy.log(2*numpy.pi*n_mask) / 2. + numpy.log(1 + 1. / 12 / n_mask)
    return res


def _M(i,j,k,l,n10,n01,n11,n00, n, f0):
    """
    The moment estimator M in appendix H
    This is the version when both A and B are rare
    TODO: double check the appendix after revision
    """
    res = numpy.power(1 - 1./n/f0, n10 - i) * numpy.power(1 - 1./n/f0, n01 - j) * numpy.power(1 - 2./n/f0, n11 - k)
    res *= perm(n10, i) / numpy.power(n, i)
    res *= perm(n01, j) / numpy.power(n, j)
    res *= perm(n11, k) / numpy.power(n, k)
    res *= perm(n00, l) / numpy.power(n, l)
    # res *= factorial_fraction(n10, i) * factorial_fraction(n01, j) * factorial_fraction(n11, k) * factorial_fraction(n00, l)
    # res /= np.power(n, i + j + k + l)
    # res /= perm(n, i+j+k+l)  # used in Ben's rare LD scripts
    return res


def any_poly(mono_coeffs, n_obs, n, f0):
    # compute estimator for any polynomial of fAb, faB, fAB, fab
    # let each monomial be C * fAb**i * faB**j * fAB**k * fab**l
    # mono_coeffs will be a list of ((i,j,k,l), C)
    res = 0
    for powers, coeff in mono_coeffs:
        res += coeff * M(powers[0], powers[1], powers[2], powers[3], n_obs, n, f0)
    return res


def M(i,j,k,l,n_obs,n,f0):
    # n_obs.shape = (# observations, 4)
    # n_obs = [n10, n01, n11, n00]  Make sure the order is correct!
    return _M(i,j,k,l,n_obs[:,0],n_obs[:,1],n_obs[:,2],n_obs[:,3],n,f0)


def poly_to_mono_coeffs(p):
    return [(m, int(p.coeff_monomial(m))) for m in p.monoms()]

def generate_LE_poly():
    x, y, z, w = sympy.symbols('f_Ab,f_aB,f_AB,f_ab')
    exp = ((x+z)*(y+w)*(y+z)*(x+w))**2
    denom = sympy.Poly(exp, x, y, z, w)
    denom_monos = poly_to_mono_coeffs(denom)

    exp = x*y*z*w
    numer = sympy.Poly(exp, x, y, z, w)
    numer_monos = poly_to_mono_coeffs(numer)
    return numer_monos, denom_monos


def calculate_LE(n_obs, f0):
    """
    Calculate the numerator and denominator of Lambda_2 from haplotype counts

    Since the denominator involves too many terms to be written out, we first use the sympy to calculate the coefficients of the polynomial.

    We next use any_poly to calculate the numerator and denominator
    """
    numer_monos, denom_monos = generate_LE_poly()
    n_tots = numpy.sum(n_obs, axis=1)
    numer = any_poly(numer_monos, n_obs, n_tots, f0)
    denom = any_poly(denom_monos, n_obs, n_tots, f0)
    return numer, denom



#n_obs[:, 2], n_obs[:, 0], n_obs[:, 1], n_obs[:, 3],
# should be 
# n11s, n10s, n01s, n00s


def build_le_counts_dict(votu, max_fraction_nan=0.05, max_d=1e3):
    
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
        
        
        sys.stderr.write("Calculating LE for...\n")
        
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

                # format for LE
                n_obs = numpy.vstack([n10, n01, n11, n00]).T
                #capped_f0s = numpy.clip(f0s,0,1.5)


                if dist_12 not in ld_count_dict['data']['all']:
                    ld_count_dict['data']['all'][dist_12] = {}
                    for f0 in f0s:
                        ld_count_dict['data']['all'][dist_12][f0] = {}
                        ld_count_dict['data']['all'][dist_12][f0]['rsquared_numerators'] = 0
                        ld_count_dict['data']['all'][dist_12][f0]['rsquared_denominators'] = 0
                        ld_count_dict['data']['all'][dist_12][f0]['n_site_pairs'] = 0
                        ld_count_dict['data']['all'][dist_12][f0]['n_site_pair_observations'] = 0


                for f0 in f0s:
                    rsquared_numerators, rsquared_denominators = calculate_LE(n_obs, f0)
                    rsquared_numerators = rsquared_numerators[0]
                    rsquared_denominators = rsquared_denominators[0]

                    if numpy.isnan(rsquared_numerators) or numpy.isnan(rsquared_denominators):
                        continue

                    #if numpy.isnan(rsquared_numerators) 

                    #print(numpy.isnan(rsquared_numerators), ld_count_dict['data']['all'][dist_12][f0]['rsquared_numerators'])
                    #if rsquared_numerators[0] != 0:
                    #    print(f0, n_obs, rsquared_numerators)
                    ld_count_dict['data']['all'][dist_12][f0]['rsquared_numerators'] += rsquared_numerators
                    ld_count_dict['data']['all'][dist_12][f0]['rsquared_denominators'] += rsquared_denominators
                    ld_count_dict['data']['all'][dist_12][f0]['n_site_pairs'] += 1
                    ld_count_dict['data']['all'][dist_12][f0]['n_site_pair_observations'] += len(allele_bool_idx_final_2)


                # loop through 1D and 4D sites
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

                    n_var_obs = numpy.vstack([n10_var, n01_var, n11_var, n00_var]).T

                    if dist_12 not in ld_count_dict['data'][variant_type]:
                        ld_count_dict['data'][variant_type][dist_12] = {}
                        for f0 in f0s:
                            ld_count_dict['data'][variant_type][dist_12][f0] = {}
                            ld_count_dict['data'][variant_type][dist_12][f0]['rsquared_numerators'] = 0
                            ld_count_dict['data'][variant_type][dist_12][f0]['rsquared_denominators'] = 0
                            ld_count_dict['data'][variant_type][dist_12][f0]['n_site_pairs'] = 0
                            ld_count_dict['data'][variant_type][dist_12][f0]['n_site_pair_observations'] = 0

                    
                    for f0 in f0s:
                        rsquared_numerators, rsquared_denominators = calculate_LE(n_var_obs, f0)

                        rsquared_numerators = rsquared_numerators[0]
                        rsquared_denominators = rsquared_denominators[0]

                        if numpy.isnan(rsquared_numerators) or numpy.isnan(rsquared_denominators):
                            continue

                        ld_count_dict['data'][variant_type][dist_12][f0]['rsquared_numerators'] += rsquared_numerators
                        ld_count_dict['data'][variant_type][dist_12][f0]['rsquared_denominators'] += rsquared_denominators
                        ld_count_dict['data'][variant_type][dist_12][f0]['n_site_pairs'] += 1
                        ld_count_dict['data'][variant_type][dist_12][f0]['n_site_pair_observations'] += len(allele_bool_idx_final_2)


                n_pairs_processed += 1  


        sys.stderr.write("Saving dictionary....\n")
        le_counts_dict_path_ = le_counts_dict_path % votu
        with open(le_counts_dict_path_, 'wb') as f:
            pickle.dump(ld_count_dict, f)
        sys.stderr.write("Done!\n")



if __name__ == "__main__":

    #votu = 'vOTU-000010'
    #votu_all = [votu]
    #build_allele_counts_map(votu)
    votu_all = data_utils.get_single_votus()
  
    idx_ = votu_all.index('vOTU-000001')
    
    #votu = 'vOTU-000078'
    for votu in votu_all[idx_:]:
        #for votu in votu_all:

        build_le_counts_dict(votu, max_fraction_nan=0.0)

