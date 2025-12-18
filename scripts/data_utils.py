
import gzip
import config
import os
from math import radians, cos, sin, asin, sqrt
import json
import sys
import pickle
import copy
import glob
import random
import random
import math

from pathlib import Path


from collections import Counter

import numpy
from scipy import stats
import pandas

#from matplotlib import colors
#from matplotlib import cm
#import matplotlib as mpl


import pypangraph


from Bio import SeqIO


n_fna_characters = 80

numpy.random.seed(123456789)
random.seed(123456789)



map_sites_core_to_original_path = config.data_directory + 'Single_vOTUs/map_sites_core_to_original/%s.pkl'
allele_counts_map_path = config.data_directory + 'allele_counts_map_all/%s.pkl'



nucleotides = ['A', 'C', 'G', 'T']
base_table = {'A':'T','T':'A','G':'C','C':'G'}
votu_to_skip = ['vOTU-000015', 'vOTU-007481']
variant_types = [0, 3]


# translation table 11
codon_table = { 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'CGT': 'R', 'CGC': 'R', 'CGA':'R',
'CGG':'R', 'AGA':'R', 'AGG':'R', 'AAT':'N', 'AAC':'N', 'GAT':'D', 'GAC':'D', 'TGT':'C', 'TGC':'D', 'CAA':'Q', 'CAG':'Q', 'GAA':'E', 'GAG':'E', 'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G', 'CAT':'H', 'CAC':'H', 'ATT':'I', 'ATC':'I', 'ATA':'I', 'TTA':'L', 'TTG':'L', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L', 'AAA':'K', 'AAG':'K', 'ATG':'M', 'TTT':'F', 'TTC':'F', 'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P', 'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'AGT':'S', 'AGC':'S', 'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T', 'TGG':'W', 'TAT':'Y', 'TAC':'Y', 'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V', 'TAA':'!', 'TGA':'!', 'TAG':'!' }


start_codons = ['TTG', 'CTG', 'ATT', 'ATC', 'ATA', 'ATG', 'GTG']

# calculate number of synonymous opportunities for each codon
# Original phage annotations all used translation table 11
# To-do: check if new annotations all use table 11
codon_synonymous_opportunity_table = {}
for codon in codon_table.keys():
    codon_synonymous_opportunity_table[codon] = {}
    for i in range(0,3):
        codon_synonymous_opportunity_table[codon][i] = -1 # since G->G is by definition synonymous, but we don't want to count it
        codon_list = list(codon)
        for base in ['A','C','T','G']:
            codon_list[i]=base
            new_codon = "".join(codon_list)
            if codon_table[codon]==codon_table[new_codon]:
                # synonymous!
                codon_synonymous_opportunity_table[codon][i] += 1




# these repeated vOTUs are not an issue if you only use MGV lines
repeated_votu_representative_all = ['vOTU-010833', 'vOTU-014541', 'vOTU-013486']
# ignore 'Not-determined' quality labels
checkv_quality_all = ['Complete', 'High-quality', 'Medium-quality', 'Low-quality']

checkv_quality_score_dict = {'Low-quality':0, 'Medium-quality':1, 'High-quality':2, 'Complete':3}

taxon_abbr_dict = {'d':'domain', 'p':'phylum', 'c':'class', 'o':'order', 'f':'family', 'g':'genus', 's':'species'}
taxon_ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

ictv_taxonomy_ranks = ['realm', 'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus']
uhgv_taxonomy_ranks = ['family', 'subfamily', 'genus', 'otu']
phage_taxonomy_ranks_dict = {'ictv_taxonomy': ictv_taxonomy_ranks, 'uhgv_taxonomy': uhgv_taxonomy_ranks}


lifestyle_all = ['lytic', 'temperate']
lifestyle_all_and_both = ['lytic', 'temperate', 'both']


uhgv_taxonomy_name_no_cap_dict = {'otu':'vOTU', 'genus': 'genus', 'subfamily':'subfamily', 'family':'family'}
str_to_boolean_dict = {'Yes': True, 'No': False}
country_code_dict = {'RUS': 'Russia', 'MNG':'Mongolia', 'FRA':'France', 'FIN':'Finland', 'AUT':'Austria', 'DEU': 'Denmark', 'GBR':'United Kingdom', 'EST':'Estonia', 'CAN':'Canada', 'PER':'Peru', 'ITA':'Italy', 'JPN':'Japan', 'FJI':'Fiji', 'BGD':'Bangladesh', 'SWE':'Sweeden', 'CHN':'China', 'KAZ':'Kazakistan', 'NLD':'Netherlands', 'ESP':'Spain', 'DNK':'Denmark', 'SLV':'El Salvador', 'ISR':'Israel', 'TZA':'Tanzania', 'USA': 'USA'}



annotation_dict_path = '%sannotation_dict.pickle' % config.data_directory
syn_sites_pangraph_dict_path = config.data_directory + 'syn_sites_pangraph_dict_all/%s.pickle' 
syn_sites_alignment_dict_path = config.data_directory + 'syn_sites_alignment_dict_all/%s.pickle' 


lifestyle_color_dict = {'both':'k', 'temperate':'#87CEEB', 'lytic':'#FF6347'}





def index_to_pair(index, n):
    '''
    Maps a unique index to a unique pair (i, j) such that i < j
    Solve for i using inverse triangular number
    '''
    
    i = n - 2 - int(math.floor(math.sqrt(-8*index + 4*n*(n-1)-7)/2.0 - 0.5))
    total_before_i = i * n - i * (i + 1) // 2
    
    # calculate k
    j = index - total_before_i + i + 1
    return i, j



def random_unique_pairs(my_list, k):
    n = len(my_list)
    total_pairs = n * (n - 1) // 2

    # Sample k unique indices from 0 to total_pairs - 1
    sampled_indices = random.sample(range(total_pairs), k)

    # Convert each index to a unique (i, j) pair
    result = []
    for idx in sampled_indices:
        i, j = index_to_pair(idx, n)
        result.append((my_list[i], my_list[j]))

    return result



def read_sample_metadata():

    '''
    Parses the Metagenomic Gut Virus sample metadata and returns a dictionary
    Key: Phage contig_id 
    '''

    sample_metagenome_dict = {}

    with gzip.open('%smgv_sample_info.tsv.gz' % config.data_directory, 'rt') as f:
        header = f.readline()
        header = header.strip().split('\t')

        # line[0] = contig_id
        # line[1] = assembly_source. Sequence database
        # line[2] = assembly_name (e.g., SRR3740115). May also be in uhgv_metadata.tsv
        # line[3] = study_accession (e.g., SRP077685)
        # line[4] = sample_accession (e.g., SRS1537996). May also be in uhgv_metadata.tsv
        # line[5] = run_accessions (e.g., SRR3740115). May also be in uhgv_metadata.tsv
        # line[6] = continent
        # line[7] = country_code
        # line[8] = sex
        # line[9] = age
        # line[10] = health
        # line[11] = disease

        for line in f:
            line = line.strip().split('\t')

            contig_id = line[0]
            assembly_source = line[1]
            assembly_name = line[2]
            study_accession = line[3]
            sample_accession = line[4]
            run_accessions = line[5]

            continent = line[6]
            country_code = line[7]
            sex = line[8]
            age = line[9]

            health = line[10]
            disease = line[11]

            sample_metagenome_dict[contig_id] = {}
            sample_metagenome_dict[contig_id]['assembly_source'] = assembly_source
            sample_metagenome_dict[contig_id]['assembly_name'] = assembly_name
            sample_metagenome_dict[contig_id]['study_accession'] = study_accession
            sample_metagenome_dict[contig_id]['sample_accession'] = sample_accession

            # run_accessions will be a LIST
            sample_metagenome_dict[contig_id]['run_accessions'] = run_accessions.split(',')

            sample_metagenome_dict[contig_id]['continent'] = continent

            sample_metagenome_dict[contig_id]['country_code'] = country_code
            sample_metagenome_dict[contig_id]['sex'] = sex
            sample_metagenome_dict[contig_id]['age'] = age

            sample_metagenome_dict[contig_id]['health'] = health
            sample_metagenome_dict[contig_id]['disease'] = disease

            #sample_metagenome_dict[]


        f.close()

    
    return sample_metagenome_dict




def read_microbe_metadata(min_completeness=0.9, max_contamination=0.05):

    '''
    Parses the Unified Human Gastrointestinal Genome metadata and returns a nested dictionary
    sra_dict = {sra_sample #: genome_id: {....}}
    Value only contains taxonomy of microbial genome (for now, feel free to add)

    min_completeness = Minimum completeness of assembly
    max_contamination = Maximum tolerated contamination
    '''
    
    sra_dict = {}

    with gzip.open('%suhgg_genomes-nr_metadata.tsv.gz' % config.data_directory, 'rt') as f:
        header = f.readline()
        header = header.strip().split('\t')

        # line[0] = Genome ID
        # line[1] = Original name (e.g., 11861_6_55)
        # line[2] = Study set (e.g., HBC, NCBI). 
        # line[3] = Genome_type (e.g., Isolate). 
        # line[4] = Length
        # line[5] = N_contigs
        # line[6] = N50
        # line[7] = GC_content
        # line[8] = Completeness
        # line[9] = Contamination
        # line[10] = CMseq (Not sure)
        # line[11] = rRNA_5S (completeness)
        # line[12] = rRNA_16S (completeness)
        # line[13] = rRNA_23S (completeness)
        # line[14] = tRNAs (#)
        # line[15] = Genome_accession (e.g., GCA_003480885)
        # line[16] = Species_rep (e.g., GUT_GENOME000001)
        # line[17] = MGnify_accession (e.g., MGYG-HGUT-00001)
        # line[18] = Lineage (taxonomy)
        # line[19] = Sample_accession (e.g., ERS370061, SRS1992821)
        # line[20] = Study_accession (e.g., ERP105624)
        # line[21] = Country
        # line[22] = Continent
        # line[23] = FTP_download


        for line in f:
            line = line.strip().split('\t')


            genome_id = line[0]
            completeness = float(line[8])/100
            contamination = float(line[9])/100
            sra_sample = line[19]
            lineage = line[18]

            if completeness < min_completeness:
                continue

            if contamination > max_contamination:
                continue


            if sra_sample not in sra_dict:
                sra_dict[sra_sample] = {}

            # genome_id is unique
            sra_dict[sra_sample][genome_id] = {}
            sra_dict[sra_sample][genome_id]['taxonomy'] = {}

            for t in lineage.split(';'):

                t_abbr, t_name = t.split('__')

                if t_name == '':
                    t_name = 'NA'

                sra_dict[sra_sample][genome_id]['taxonomy'][taxon_abbr_dict[t_abbr]] = t_name
                

        f.close()

    
    return sra_dict



def read_uhgv_metadata(checkv_quality='High-quality', checkv_quality_cumulative=True, viral_confidence_criterion='Confident', viralverify_prediction_criterion='Virus'):

    #  subset_votu_representative=False, 

    '''
    Parses the Unified Human Gastrointestinal Genome metadata and returns two dictionaries
    
    votu_dict = Dictionary containing metadata for a given vOTU. Each metadata variable is a list and each item
    in the list is the metadata for a single genome within said vOTU. This can be rewritten to be nested if useful.

    vgenome_dict = Dictionary containing metadata for a given UHGV microbial genome.

    checkv_quality = Required assembly quality 
    checkv_quality_cumulative = Only keep assembles with a quality greater than or equal to checkv_quality (bool)
    viral_confidence_criterion = Confidence criterion
    viralverify_prediction_criterion = Viralverify prediction criterion
    '''

    #sra_dict = {}
    votu_dict = {}
    vgenome_dict = {}

    with gzip.open('%suhgv_metadata.tsv.gz' % config.data_directory, 'rt') as f:
        header = f.readline()
        header = header.strip().split('\t')

        # line[0] = uhgv_genome (e.g., UHGV-1495677)
        # line[1] = uhgv_votu (e.g., vOTU-000135)
        # line[2] = votu_representative (Yes/No)
        # line[3] = original_study_alias (e.g., MGV)
        # line[4] = original_id (e.g., MGV-GENOME-0354054)
        # line[5] = genome_length
        # line[6] = is_jumbo_phage (Yes/No)
        # line[7] = gc_percent
        # line[8] = checkv_quality (e.g., High-quality)
        # line[9] = checkv_completeness (%)
        # line[10] = checkv_completeness_method (e.g., AAI-based)
        # line[11] = checkv_trimmed (Yes/No)
        # line[12] = viral_confidence (e.g., Confident)
        # line[13] = genomad_virus_score 
        # line[14] = genomad_virus_hallmarks (represents if the detection method for the phages (genomad) found a phage hallmark gene (eg. capsid, terminase, etc.))
        # line[15] = genomad_plasmid_hallmarks (?)
        # line[16] = viralverify_prediction (e.g., Virus)
        # line[17] = viralverify_score (numeric)
        # line[18] = checkv_viral_markers (numeric)
        # line[19] = checkv_host_markers (numeric)
        # line[20] = cds_count (numeric)
        # line[21] = cds_density (numeric) ?
        # line[22] = avg_cds_length (numeric)
        # line[23] = genetic_code (numeric)
        # line[24] = is_recoded (Yes/No)
        # line[25] = trna_count_total
        # line[26] = trna_count_suppressor (A suppressor tRNA is a tRNA with a mutation (usually) in the anticodon that allows it to recognize a stop codon and insert an amino acid in its place) 10.1128/JB.186.20.6714-6720.2004
        # line[27] = sra_run (e.g., ERR414592,ERR414311 or SRR5274010)
        # line[28] = sra_sample (e.g., ERS396364 or SRS1992821)
        # line[29] = biosample (e.g., SAMEA2338694)
        # line[30] = country
        # line[31] = latitude
        # line[32] = longitude

        for line in f:
            line = line.strip().split('\t')

            uhgv_genome = line[0]
            uhgv_votu = line[1]

            original_study_alias = line[3]
            original_id = line[4]
            genome_length = line[5]

            quality = line[8]
            checkv_completeness = line[9]

            viral_confidence = line[12]
            viralverify_prediction = line[16]
            sra_sample = line[28]

            country = line[30]
            latitude = line[31]
            longitude = line[32]
            
            if quality == 'Not-determined':
                continue

            if checkv_completeness == 'NULL':
                continue
            
            checkv_completeness = float(checkv_completeness)/100
            
            checkv_quality_score = checkv_quality_score_dict[quality]

            if checkv_quality_cumulative == True:
                if checkv_quality_score < checkv_quality_score_dict[checkv_quality]:
                    continue
            else:
                if (quality != checkv_quality):
                    continue


            if original_study_alias != 'MGV':
                continue
            
            if (viral_confidence != viral_confidence_criterion):
                continue

            if (viralverify_prediction != viralverify_prediction_criterion):
                continue
            
            if line[28] == 'Null':
                continue

            if uhgv_votu not in votu_dict:
                votu_dict[uhgv_votu] = {}
                votu_dict[uhgv_votu]['uhgv_genome'] = []
                votu_dict[uhgv_votu]['original_study_alias'] = []
                votu_dict[uhgv_votu]['original_id'] = []
                votu_dict[uhgv_votu]['genome_length'] = []
                votu_dict[uhgv_votu]['quality'] = []
                votu_dict[uhgv_votu]['checkv_completeness'] = []
                votu_dict[uhgv_votu]['viral_confidence'] = []
                votu_dict[uhgv_votu]['viralverify_prediction'] = []
                votu_dict[uhgv_votu]['sra_sample'] = []
                votu_dict[uhgv_votu]['country'] = []


                
            votu_dict[uhgv_votu]['uhgv_genome'].append(uhgv_genome)
            votu_dict[uhgv_votu]['original_study_alias'].append(original_study_alias)
            votu_dict[uhgv_votu]['original_id'].append(original_id)
            votu_dict[uhgv_votu]['genome_length'].append(genome_length)
            votu_dict[uhgv_votu]['quality'].append(quality)
            votu_dict[uhgv_votu]['checkv_completeness'].append(checkv_completeness)
            votu_dict[uhgv_votu]['viral_confidence'].append(viral_confidence)
            votu_dict[uhgv_votu]['viralverify_prediction'].append(viralverify_prediction)
            votu_dict[uhgv_votu]['sra_sample'].append(sra_sample)
            votu_dict[uhgv_votu]['country'].append(country)


            # return everything for vgenome_dict
            vgenome_dict[uhgv_genome] = {}
            vgenome_dict[uhgv_genome]['uhgv_votu'] = uhgv_votu
            vgenome_dict[uhgv_genome]['original_study_alias'] = original_study_alias
            vgenome_dict[uhgv_genome]['original_id'] = original_id
            vgenome_dict[uhgv_genome]['genome_length'] = genome_length
            vgenome_dict[uhgv_genome]['quality'] = quality
            vgenome_dict[uhgv_genome]['checkv_completeness'] = checkv_completeness
            vgenome_dict[uhgv_genome]['viral_confidence'] = viral_confidence
            vgenome_dict[uhgv_genome]['viralverify_prediction'] = viralverify_prediction
            vgenome_dict[uhgv_genome]['sra_sample'] = sra_sample
            vgenome_dict[uhgv_genome]['country'] = country

        f.close()


    return votu_dict, vgenome_dict



def read_uhgv_votu_metadata():

    '''
    Returns metadata dictionary for UHGV vOTUS
    '''

    uhgv_genome_metadata_dict = {}

    # one line per genome AND vOTU
    # representative genomes
    with gzip.open('%suhgv_votus_metadata.tsv.gz' % config.data_directory, 'rt') as f:
        
        header = f.readline()
        header = header.strip().split('\t')

        # line[0] = uhgv_genome (e.g., UHGV-1495677)
        # line[1] = uhgv_votu (e.g., vOTU-000135)
        # line[2] = checkv_completeness (0-100)
        # line[3] = checkv_quality (e.g., Complete)
        # line[4] = viral_confidence (e.g., Confident)
        # line[5] = genome_length
        # line[6] = cds_count
        # line[7] = ictv_taxonomy
        # line[8] = uhgv_taxonomy
        # line[9] = host_lineage
        # line[10] = lifestyle
        # line[11] = http_url


        for line in f:
            line = line.strip().split('\t')

            uhgv_genome = line[0]
            uhgv_votu = line[1]
            checkv_completeness = line[2]
            checkv_quality = line[3]
            viral_confidence = line[4]
            genome_length = line[5]
            cds_count = line[6]
            lifestyle = line[10]

            uhgv_genome_metadata_dict[uhgv_votu] = {}
            uhgv_genome_metadata_dict[uhgv_votu]['uhgv_genome'] = uhgv_genome
            uhgv_genome_metadata_dict[uhgv_votu]['checkv_completeness'] = checkv_completeness
            uhgv_genome_metadata_dict[uhgv_votu]['checkv_quality'] = checkv_quality
            uhgv_genome_metadata_dict[uhgv_votu]['viral_confidence'] = viral_confidence
            uhgv_genome_metadata_dict[uhgv_votu]['genome_length'] = genome_length
            uhgv_genome_metadata_dict[uhgv_votu]['cds_count'] = cds_count
            uhgv_genome_metadata_dict[uhgv_votu]['lifestyle'] = lifestyle

            ictv_taxonomy_split = line[7].split(';')
            uhgv_taxonomy_split = line[8].split(';')
            host_lineage = line[9]

            uhgv_genome_metadata_dict[uhgv_votu]['ictv_taxonomy'] = {}
            uhgv_genome_metadata_dict[uhgv_votu]['uhgv_taxonomy'] = {}
            uhgv_genome_metadata_dict[uhgv_votu]['host_lineage'] = {}

            if host_lineage == 'NULL':
                for t in taxon_ranks:
                    uhgv_genome_metadata_dict[uhgv_votu]['host_lineage'][t] = 'NA'

            else:
                host_lineage_split = host_lineage.split(';')
                # skip domain, uninformative
                host_lineage_split = [h.split('__')[1] for h in host_lineage_split][1:]

                for taxon_ranks_i_idx, taxon_ranks_i in enumerate(taxon_ranks):
                    if taxon_ranks_i_idx >= len(host_lineage_split):
                        uhgv_genome_metadata_dict[uhgv_votu]['host_lineage'][taxon_ranks_i] = 'NA'

                    else:
                        uhgv_genome_metadata_dict[uhgv_votu]['host_lineage'][taxon_ranks_i] = host_lineage_split[taxon_ranks_i_idx]

            

            for ictv_taxonomy_ranks_i_idx, ictv_taxonomy_ranks_i in enumerate(ictv_taxonomy_ranks):

                if ictv_taxonomy_ranks_i_idx >= len(ictv_taxonomy_split):
                    uhgv_genome_metadata_dict[uhgv_votu]['ictv_taxonomy'][ictv_taxonomy_ranks_i] = 'NA'

                else:
                    uhgv_genome_metadata_dict[uhgv_votu]['ictv_taxonomy'][ictv_taxonomy_ranks_i] = ictv_taxonomy_split[ictv_taxonomy_ranks_i_idx]


            for uhgv_taxonomy_ranks_i_idx, uhgv_taxonomy_ranks_i in enumerate(uhgv_taxonomy_ranks):

                if uhgv_taxonomy_ranks_i_idx >= len(uhgv_taxonomy_split):
                    uhgv_genome_metadata_dict[uhgv_votu]['uhgv_taxonomy'][uhgv_taxonomy_ranks_i] = 'NA'

                else:
                    uhgv_genome_metadata_dict[uhgv_votu]['uhgv_taxonomy'][uhgv_taxonomy_ranks_i] = uhgv_taxonomy_split[uhgv_taxonomy_ranks_i_idx]


    return uhgv_genome_metadata_dict
        





def parse_mgv_uhgv_species():

    '''
    Returns dictionary of taxonomic information for vOTU and for its host
    '''
     
    mgv_uhgv_species_dict = {}

    with open('%smgv_uhgv_species_list.txt' % config.data_directory, 'r') as f:

        header = f.readline()
        header = header.strip().split('\t')

        for line in f:
            
            line = line.strip().split('\t')
            votu = line[0]
            
            mgv_uhgv_species_dict[votu] = {}
            mgv_uhgv_species_dict[votu]['repgenome'] = line[1]
            mgv_uhgv_species_dict[votu]['num_genomes'] = line[2]
            mgv_uhgv_species_dict[votu]['complete'] = line[3]
            mgv_uhgv_species_dict[votu]['high_quality'] = line[4]
            mgv_uhgv_species_dict[votu]['medium_quality'] = line[5]
            mgv_uhgv_species_dict[votu]['order'] = line[6]
            mgv_uhgv_species_dict[votu]['family'] = line[7]
            mgv_uhgv_species_dict[votu]['length'] = line[8]
            mgv_uhgv_species_dict[votu]['prophage_avg'] = line[9]
            mgv_uhgv_species_dict[votu]['rep_temperate_score'] = line[10]
            mgv_uhgv_species_dict[votu]['max_temperate_score'] = line[11]
            mgv_uhgv_species_dict[votu]['avg_temperate_score'] = line[12]
            mgv_uhgv_species_dict[votu]['avg_temperate_complete'] = line[13]
            mgv_uhgv_species_dict[votu]['host_lineage'] = line[14]

            
            
    return mgv_uhgv_species_dict







def votu_dict_to_pres_abs(phage_metadata_dict):

    '''
    Takes the entries in phage_metadata_dict and returns a presence-absence vOTU-by-SRAsample dictionary
    '''

    votu_pres_abs = pandas.DataFrame({k:Counter(v['sra_sample']) for k, v in phage_metadata_dict.items()}).T.fillna(0).astype(int)

    votu_pres_abs_array = votu_pres_abs.values   

    sample_names = votu_pres_abs.columns.values
    votu_names = votu_pres_abs.index.values

    # samples are COLUMNS
    # vOTUs are ROWS

    return votu_pres_abs_array, votu_names, sample_names





def coarse_grain_abundances_by_taxonomy(count_array, votus, uhgv_votu_metadata_dict, taxonomic_level='genus', phage_taxonomic_system='uhgv_taxonomy', pres_abs=True):

    '''
    Coarse-grain vOTU presence-absence data to a specified phage tanomic level.
    Requires uhgv_votu_metadata_dict to map vOTU names to taxonomic level.


    taxonomic_level = Phage taxonomic level that the data is coarse-grained to
    phage_taxonomic_system = uhgv_taxonomy OR ictv_taxonomy
    pres_abs = Specifies that you want presence/absence data. Otherwise counts are returned (bool)

    '''

    taxon_labels_all = [uhgv_votu_metadata_dict[v][phage_taxonomic_system][taxonomic_level] for v in votus]

    taxa_to_keep_idx = numpy.asarray(taxon_labels_all)
    taxa_to_keep_idx = taxa_to_keep_idx != "NA"

    taxon_labels_all_clean = [t for t in taxon_labels_all if t != "NA"]

    # return this list for null coarse-grained site-by-species matrix
    n_collapsed_nodes = list(dict(Counter(taxon_labels_all_clean)).values())

    # remove duplicates from taxon_labels_all_clean while keeping original order
    def unique(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    taxon_labels_all_clean_set = unique(taxon_labels_all_clean)
    taxon_labels_all_clean_set = numpy.asarray(taxon_labels_all_clean_set)
    taxon_labels_all_clean_copy = taxon_labels_all_clean.copy()

    to_remove_idx = [i for i, x in enumerate(taxon_labels_all) if x == "NA"]
    to_remove_idx = numpy.asarray(to_remove_idx)
    count_array_ = numpy.copy(count_array)

    # delete NA
    if len(to_remove_idx) > 0:
        count_array_ = numpy.delete(count_array_, to_remove_idx, axis=0)

    for taxon_i in taxon_labels_all_clean_set:

        taxon_i_idx = [i for i, x in enumerate(taxon_labels_all_clean_copy) if x == taxon_i]
        taxon_i_idx = numpy.asarray(taxon_i_idx)

        # remove taxon from the list
        taxon_labels_all_clean_copy = [s for s in taxon_labels_all_clean_copy if s != taxon_i]

        count_array_to_merge = count_array_[taxon_i_idx,]
        count_array_ = numpy.delete(count_array_, taxon_i_idx, axis=0)

        count_array_sum = numpy.sum(count_array_to_merge, axis=0)
        count_array_ = numpy.vstack((count_array_, count_array_sum))

    
    if pres_abs == True:
        count_array_[count_array_>0] = 1


    return count_array_, taxon_labels_all_clean_set, n_collapsed_nodes, taxa_to_keep_idx



def get_single_votus(remove_bad_votus=True):

    directory = Path('%sSingle_vOTUs/Core_Alignments/' % config.data_directory)
    files = [f.name for f in directory.iterdir() if f.is_file() and ('.fna' in f.name)]
    votu_all = [f.split('_')[2] for f in files]
    votu_all.sort()
    
    if remove_bad_votus == True:
        votu_all_ = [x for x in votu_all if x not in votu_to_skip]
    else:
        votu_all_ = votu_all
    
    return votu_all_





def build_votu_fasta(votu='vOTU-000085', checkv_quality='Complete', subset_single_votus=True):

    '''
    Builds a fasta file for a given vOTU
    '''

    if subset_single_votus == True:
        # get target genomes from the FASTA Jimmy processed
        file_path = glob.glob('%sSingle_vOTUs/Core_Alignments/*%s*.fna' % (config.data_directory, votu))[0]
        fasta_all_genomes = classFASTA(file_path).readFASTA()
        fasta_genome_dict = {x[0]:x[1] for x in fasta_all_genomes}
        target_genome_all = list(fasta_genome_dict.keys())

    else:
        uhgv_votu_metadata_dict, uhgv_genome_metadata_dict = read_uhgv_metadata(checkv_quality=checkv_quality, checkv_quality_cumulative=True)
        target_genome_all = uhgv_votu_metadata_dict[votu]['uhgv_genome']


    fasta_file_path = '%suhgv_mgv_otu_fna/%s.fna' % (config.data_directory, votu) 
    fasta_file_open = open(fasta_file_path, 'w')

    with gzip.open('%suhgv_mgv.fna.gz' % config.data_directory, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):

            if record.id in target_genome_all:

                fasta_file_open.write('>%s\n' % record.id)

                clean_sites_seq_split = [record.seq[i:i+n_fna_characters] for i in range(0, len(record.seq), n_fna_characters)]

                for seq in clean_sites_seq_split:
                    fasta_file_open.write('%s\n' % seq)

                fasta_file_open.write('\n')


    fasta_file_open.close()



#def build_votu_fasta(votu='vOTU-000085', checkv_quality='Complete'):





def blast(input_fasta_filename, output_directory, single_file=True, max_n=100):

    '''
    Slow code to run blast on a fasta file using biopython
    '''

    # Note: NEED TO DEFINE A SET OF CONTIG IDS
    
    if input_fasta_filename.endswith('.gz'):
        input_file = gzip.open(input_fasta_filename,"rt")
    else:
        input_file = open(input_fasta_filename,"r")
    
    if single_file:
        output_file = open(output_filename,"w")
    else:
        os.system('mkdir -p %s' % output_directory)
    
    num_recorded = 0
    for record in SeqIO.parse(input_file, "fasta"):
        if record.id in allowed_contigs:
            #print(record.id)
            if single_file:
                SeqIO.write(record, output_file, "fasta")
            else:
                output_filename = "%s/%s.fasta" % (output_directory,record.id)
            
            output_file = open(output_filename,"w")
            SeqIO.write(record, output_file, "fasta")
            output_file.close()
            num_recorded+=1
            
            if (max_n) > 0 and (num_recorded>=max_n):
                break
    
    output_file.close()
    input_file.close()




class classFASTA:

    '''
    Class to load and read fasta file, returning sequencies as a nested list
    '''

    def __init__(self, fileFASTA):
        self.fileFASTA = fileFASTA

    def readFASTA(self):
        '''Checks for fasta by file extension'''
        file_lower = self.fileFASTA.lower()
        '''Check for three most common fasta file extensions'''
        if file_lower.endswith('.txt') or file_lower.endswith('.fa') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.fna') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.frn') or \
        file_lower.endswith('.faa') or file_lower.endswith('.ffn'):
            with open(self.fileFASTA, "r") as f:
                return self.ParseFASTA(f)
        else:
            print("Not in FASTA format.")

    def ParseFASTA(self, fileFASTA):
        '''Gets the sequence name and sequence from a FASTA formatted file'''
        fasta_list=[]
        for line in fileFASTA:
            if line[0] == '>':
                try:
                    fasta_list.append(current_dna)
            	#pass if an error comes up
                except UnboundLocalError:
                    #print "Inproper file format."
                    pass
                current_dna = [line.lstrip('>').rstrip('\n'),'']
            else:
                current_dna[1] += "".join(line.split())
        fasta_list.append(current_dna)
        '''Returns fasa as nested list, containing line identifier \
            and sequence'''
        return fasta_list
    


def get_top_n_votus(n_votus_per_lifestyle=20):

    '''
    Code to find and subset the n most prevalent OTUs.
    '''

    uhgv_votu_metadata_dict = read_uhgv_votu_metadata()

    # get phage presence/absence dictionary
    votu_dict, vgenome_dict = read_uhgv_metadata(checkv_quality='Complete', checkv_quality_cumulative=True)
    # build the presence/absence matrix
    votu_pres_abs_array, votu_names, sample_names = votu_dict_to_pres_abs(votu_dict)
    # get country IDs from mgv_sample_info.tsv.gz using sample_accession

    lifestyle_votu_all = numpy.asarray([uhgv_votu_metadata_dict[v]['lifestyle'] for v in votu_names])

    file_path = '%stop_%s_votus_per_lifestyle.csv' % (config.data_directory, n_votus_per_lifestyle) 
    file_open = open(file_path, 'w')
    file_open.write('lifestyle,votu,n_genomes\n')

    for l in lifestyle_all:

        votu_names_l = votu_names[lifestyle_votu_all==l]

        n_genomes_l = numpy.asarray([len(votu_dict[i]['original_id']) for i in votu_names_l] )

        n_genomes_l_argsort_idx = numpy.argsort(n_genomes_l)

        votu_names_l = votu_names_l[n_genomes_l_argsort_idx][::-1]
        n_genomes_l = n_genomes_l[n_genomes_l_argsort_idx][::-1]

        for k in range(n_votus_per_lifestyle):
            file_open.write('%s,%s,%d\n' % (l, votu_names_l[k], n_genomes_l[k]))


    file_open.close()



def load_pangraph_data(pangraph_file_path):

    '''
    Load json formatted pangraph data
    '''

    pangraph_file = open(pangraph_file_path, 'r')
    pangraph_data = json.load(pangraph_file)
    pangraph_file.close()

    return pangraph_data



def get_pangraph_genome_names(pangraph_data):

    '''
    Get pangraph genome names
    '''

    # all genome names
    names_all = []
    for i in pangraph_data['paths']:
        names_i = [j['name'] for j in i['blocks']]
        names_all.extend(names_i)

    names_all = list(set(names_all))

    return names_all



def calculate_annotated_divergence_across_pangraph_blocks(genome_1, genome_2, pangraph_data, syn_sites_dict):

    '''
    Function to calculate synonymous and nonsynonymous divergence between a pair of genomes within the same vOTU 

    genome_1, genome_2 = genome IDs (str)
    pangraph_data = dictionary loaded using data_utils.load_pangraph_data(minimap_path % votu)
    syn_sites_dict = dictionary for site status. Generated using make_syn_sites_votu_dict()
    '''

    cumulative_block_len = 0

    # returns a vector of the positions of mutations and the positions of annotated sites 
    cumulative_mut_position_syn = []
    cumulative_mut_position_nonsyn = []

    cumulative_block_position_syn = []
    cumulative_block_position_nonsyn = []

    block_inter_id = []
    block_position_dict = {}

    # the order of the blocks should be sorted
    for block_i_idx, block_i in enumerate(pangraph_data['blocks']):

        block_i_id = block_i['id']
        sequence_i = block_i['sequence']
        # length of the block
        len_sequence_i = len(sequence_i)

        # save the positions of the block f
        block_position_dict[block_i_id] = [cumulative_block_len, cumulative_block_len+len_sequence_i]

        block_i_name_genomes = [i[0]['name'] for i in block_i['positions']]
        # dict_keys(['id', 'sequence', 'gaps', 'mutate', 'insert', 'delete', 'positions'])
        if (genome_1 in block_i_name_genomes) and (genome_2 in block_i_name_genomes):

            mutatate_1 = [k for k in block_i['mutate'] if k[0]['name'] == genome_1][0]
            mutatate_2 = [k for k in block_i['mutate'] if k[0]['name'] == genome_2][0]

            
            # get all sites shared between the two genomes within the block
            shared_site_dict_i = {}

            # get positions and alleles for each variant in genome_1
            for mutatate_1_l in mutatate_1[1]:
                mutatate_1_l_pos, mutatate_1_l_allele = mutatate_1_l

                # the mutation position refers to the position in the block
                # we should not need to use 'strand' because we have the block positions in syn_sites_dict
                if mutatate_1_l_pos not in shared_site_dict_i:
                    shared_site_dict_i[mutatate_1_l_pos] = {}

                shared_site_dict_i[mutatate_1_l_pos]['genome_1'] = mutatate_1_l_allele


            for mutatate_2_l in mutatate_2[1]:
                mutatate_2_l_pos, mutatate_2_l_allele = mutatate_2_l

                if mutatate_2_l_pos not in shared_site_dict_i:
                    shared_site_dict_i[mutatate_2_l_pos] = {}

                shared_site_dict_i[mutatate_2_l_pos]['genome_2'] = mutatate_2_l_allele


            # identify fourfold status of the site in each genome
            examine_syn_block = False
            if syn_sites_dict != None: # annotation dict exists
                if syn_sites_dict[block_i_id]['keep_block'] == True: # blocks that mapped to the fasta
                    # make sure genome_1 and genome_2 have genes in this same block
                    if (genome_1 in syn_sites_dict[block_i_id]['data']['genomes']) and (genome_2 in syn_sites_dict[block_i_id]['data']['genomes']):

                        site_block_position_1 = syn_sites_dict[block_i_id]['data']['genomes'][genome_1]['site_block_position']
                        site_block_position_2 = syn_sites_dict[block_i_id]['data']['genomes'][genome_2]['site_block_position']
                        site_syn_status_1 = syn_sites_dict[block_i_id]['data']['genomes'][genome_1]['site_syn_status']
                        site_syn_status_2 = syn_sites_dict[block_i_id]['data']['genomes'][genome_2]['site_syn_status']

                        # find CDS sites shared by genomes in the block
                        site_block_position_inter = numpy.intersect1d(site_block_position_1, site_block_position_2)
                        site_syn_status_1_inter = numpy.asarray([site_syn_status_1[numpy.where(site_block_position_1 == x)[0][0]] for x in site_block_position_inter])
                        site_syn_status_2_inter = numpy.asarray([site_syn_status_2[numpy.where(site_block_position_2 == x)[0][0]] for x in site_block_position_inter])

                        # calculate subset of sites that have the same (non)syn status in both of the genomes
                        nonsyn_idx = (site_syn_status_1_inter == 0) & (site_syn_status_2_inter == 0)
                        syn_idx = (site_syn_status_1_inter == 3) & (site_syn_status_2_inter == 3)

                        site_block_position_inter_nonsyn = site_block_position_inter[nonsyn_idx]
                        site_block_position_inter_syn = site_block_position_inter[syn_idx]

                        # save positions if you are calculating divergence along genome.
                        cumulative_block_position_nonsyn.extend((cumulative_block_len + site_block_position_inter_nonsyn).tolist())
                        cumulative_block_position_syn.extend((cumulative_block_len + site_block_position_inter_syn).tolist())

                        # reset
                        examine_syn_block = True

                        # save the block
                        block_inter_id.append(block_i_id)


            # go back through and identify alleles
            if examine_syn_block == True:
                for pos_k, allele_dict_k in shared_site_dict_i.items():

                    # same allele, do not count towards divergence
                    if ('genome_1' in allele_dict_k) and ('genome_2' in allele_dict_k):
                        
                        # same allele, do not count towards divergence
                        if allele_dict_k['genome_1'] == allele_dict_k['genome_2']:
                            continue
    
                    
                    if examine_syn_block == True:

                        if pos_k in site_block_position_inter_nonsyn:
                            # we only calculate divergence on blocks present in both genomes
                            cumulative_mut_position_nonsyn.append(cumulative_block_len + pos_k)

                        if pos_k in site_block_position_inter_syn:
                            cumulative_mut_position_syn.append(cumulative_block_len + pos_k)


        # add block length
        cumulative_block_len += len_sequence_i



    cumulative_mut_position_syn = numpy.asarray(cumulative_mut_position_syn)
    cumulative_mut_position_nonsyn = numpy.asarray(cumulative_mut_position_nonsyn)

    cumulative_block_position_syn = numpy.asarray(cumulative_block_position_syn)
    cumulative_block_position_nonsyn = numpy.asarray(cumulative_block_position_nonsyn)

    block_inter_id = numpy.asarray(block_inter_id)
    
    return cumulative_mut_position_syn, cumulative_mut_position_nonsyn, cumulative_block_position_syn, cumulative_block_position_nonsyn, block_inter_id, block_position_dict

 






def calculate_divergence_across_pangraph_blocks(genome_1, genome_2, pangraph_data, syn_sites_dict, bin_n = 1000, calculate_binned_divergence=False):

    '''
    Calculates divergence **without** considering annotation status
    Likely not useful moving forward.

    pangraph_data = dictionary loaded using data_utils.load_pangraph_data(minimap_path % votu)
    bin_n = Bin size for divergence calculations
    calculate_binned_divergence = specifies that you are calculating divergences using bins
    '''

    # knit pangraph blocks together along the entire alignment

    # dict_keys(['name', 'offset', 'circular', 'position', 'blocks'])
    block_id_final = []
    block_position = []
    cumulative_block_position = []

    # get all the blocks THEN calculate distance so you can go across blocks..
    cumulative_block_len = 0
    cumulative_inter_block_len = 0
    cumulative_union_block_len = 0

    cumulative_block_len_syn = 0
    cumulative_block_len_nonsyn = 0
    cumulative_n_syn = 0
    cumulative_n_nonsyn = 0

    cumulative_block_position_syn = []
    cumulative_block_position_nonsyn = []

    nonsyn_sites = []
    syn_sites = []

    # the order of the blocks should be sorted
    for block_i in pangraph_data['blocks']:
        
        block_i_id = block_i['id']
        sequence_i = block_i['sequence']
        # length of the block
        len_sequence_i = len(sequence_i)

        block_i_name_genomes = [i[0]['name'] for i in block_i['positions']]
        # dict_keys(['id', 'sequence', 'gaps', 'mutate', 'insert', 'delete', 'positions'])

        if (genome_1 in block_i_name_genomes) or (genome_1 in block_i_name_genomes):
            cumulative_union_block_len += len_sequence_i
        
        if (genome_1 in block_i_name_genomes) and (genome_2 in block_i_name_genomes):

            cumulative_inter_block_len += len_sequence_i

            #positions_i = block_i['positions']
            
            #position_1_idx = block_i_name_genomes.index(genome_1)
            #position_2_idx = block_i_name_genomes.index(genome_2)

            #start_1, stop_1 = positions_i[position_1_idx][1]
            #start_2, stop_2 = positions_i[position_2_idx][1]

            # always set orientation to true...
            #strand_1 = positions_i[position_1_idx][0]['strand']
            #strand_2 = positions_i[position_2_idx][0]['strand']

            mutatate_1 = [k for k in block_i['mutate'] if k[0]['name'] == genome_1][0]
            mutatate_2 = [k for k in block_i['mutate'] if k[0]['name'] == genome_2][0]

            #block_i_id_genomes = [i[0]['name'] for i in block_i['positions']]
            # get all sites shared between the two genomes within the block
            shared_site_dict_i = {}

            # get positions and alleles for each variant in genome_1
            for mutatate_1_l in mutatate_1[1]:
                mutatate_1_l_pos, mutatate_1_l_allele = mutatate_1_l

                # the mutation position refers to the position in the block
                # we should not need to use 'strand' because we have the block positions in syn_sites_dict
                
                #if mutatate_1[0]['strand'] == False:
                #    continue
                #    mutatate_1_l_pos = len_sequence_i - mutatate_1_l_pos

                if mutatate_1_l_pos not in shared_site_dict_i:
                    shared_site_dict_i[mutatate_1_l_pos] = {}

                shared_site_dict_i[mutatate_1_l_pos]['genome_1'] = mutatate_1_l_allele


            for mutatate_2_l in mutatate_2[1]:
                mutatate_2_l_pos, mutatate_2_l_allele = mutatate_2_l

                if mutatate_2_l_pos not in shared_site_dict_i:
                    shared_site_dict_i[mutatate_2_l_pos] = {}

                shared_site_dict_i[mutatate_2_l_pos]['genome_2'] = mutatate_2_l_allele


            # identify fourfold status of the site in each genome
            examine_syn_block = False
            if syn_sites_dict != None: # annotation dict exists
                if syn_sites_dict[block_i_id]['keep_block'] == True: # blocks that mapped to the fasta
                    # make sure genome_1 and genome_2 have genes in this same block
                    if (genome_1 in syn_sites_dict[block_i_id]['data']['genomes']) and (genome_2 in syn_sites_dict[block_i_id]['data']['genomes']):

                        mutate_block_position_1 = syn_sites_dict[block_i_id]['data']['genomes'][genome_1]['site_block_position']
                        mutate_block_position_2 = syn_sites_dict[block_i_id]['data']['genomes'][genome_2]['site_block_position']
                        mutate_syn_status_1 = syn_sites_dict[block_i_id]['data']['genomes'][genome_1]['site_syn_status']
                        mutate_syn_status_2 = syn_sites_dict[block_i_id]['data']['genomes'][genome_2]['site_syn_status']

                        # find CDS sites shared by genomes in the block
                        mutate_block_position_inter = numpy.intersect1d(mutate_block_position_1, mutate_block_position_2)
                        mutate_syn_status_1_inter = numpy.asarray([mutate_syn_status_1[numpy.where(mutate_block_position_1 == x)[0][0]] for x in mutate_block_position_inter])
                        mutate_syn_status_2_inter = numpy.asarray([mutate_syn_status_2[numpy.where(mutate_block_position_2 == x)[0][0]] for x in mutate_block_position_inter])

                        # calculate subset of sites that have the same (non)syn status in both of the genomes
                        nonsyn_idx = (mutate_syn_status_1_inter == 0) & (mutate_syn_status_2_inter == 0)
                        syn_idx = (mutate_syn_status_1_inter == 3) & (mutate_syn_status_2_inter == 3)

                        mutate_block_position_inter_nonsyn = mutate_block_position_inter[nonsyn_idx]
                        mutate_block_position_inter_syn = mutate_block_position_inter[syn_idx]

                        n_sites_nonsyn_block = sum(nonsyn_idx)
                        n_sites_syn_block = sum(syn_idx)
                        

                        # save positions if you are calculating divergence along genome.
                        if calculate_binned_divergence == True:
                            nonsyn_sites.extend((cumulative_inter_block_len + mutate_block_position_inter_nonsyn).tolist())
                            syn_sites.extend((cumulative_inter_block_len + mutate_block_position_inter_syn).tolist())


                        # save cumulative # sites
                        cumulative_block_len_nonsyn += n_sites_nonsyn_block
                        cumulative_block_len_syn += n_sites_syn_block

                        examine_syn_block = True


            # go back through and identify alleles
            # pos_k = position in the block
            for pos_k, allele_dict_k in shared_site_dict_i.items():

                # positions in genome start counting at one
                #if 'genome_1' not in allele_dict_k:
                #    allele_1 = sequence_i_1[pos_k-1]
                #else:
                #    allele_1 = allele_dict_k['genome_1']


                #if 'genome_2' not in allele_dict_k:
                #    allele_2 = sequence_i_2[pos_k-1]
                #else:
                #    allele_2 = allele_dict_k['genome_2']

                # same allele, do not count towards divergence
                if ('genome_1' in allele_dict_k) and ('genome_2' in allele_dict_k):
                    
                    # same allele, do not count towards divergence
                    if allele_dict_k['genome_1'] == allele_dict_k['genome_2']:
                        continue

                    
                
                #if allele_1 == allele_2:
                #    continue
                
                block_id_final.append(block_i['id'])
                block_position.append(pos_k)
                cumulative_block_position.append(cumulative_block_len + pos_k)
                #allele_all_1.append(allele_1)
                #allele_all_2.append(allele_2)

                if examine_syn_block == True:

                    if pos_k in mutate_block_position_inter_nonsyn:
                        cumulative_n_nonsyn += 1
                        # we only calculate divergence on blocks present in both genomes
                        cumulative_block_position_nonsyn.append(cumulative_inter_block_len + pos_k)

                    if pos_k in mutate_block_position_inter_syn:
                        cumulative_n_syn += 1
                        cumulative_block_position_syn.append(cumulative_inter_block_len + pos_k)


        cumulative_block_len += len_sequence_i


    block_id_final = numpy.asarray(block_id_final)
    block_position = numpy.asarray(block_position)
    cumulative_block_position = numpy.asarray(cumulative_block_position)
    cumulative_block_position_nonsyn = numpy.asarray(cumulative_block_position_nonsyn)
    cumulative_block_position_syn = numpy.asarray(cumulative_block_position_syn)
    #allele_all_1 = numpy.asarray(allele_all_1)
    #allele_all_2 = numpy.asarray(allele_all_2)

    if calculate_binned_divergence == True:

        all_bins = list(range(0, cumulative_block_len-1, bin_n))
        all_bins.append(cumulative_block_len)

        all_bins = numpy.asarray(all_bins)

        binned_divergence = numpy.asarray([sum((cumulative_block_position >= all_bins[i]) & (cumulative_block_position < all_bins[i+1])) for i in range(len(all_bins)-1)])
        binned_divergence = binned_divergence/bin_n

        
        syn_sites = numpy.asarray(syn_sites)
        nonsyn_sites = numpy.asarray(nonsyn_sites)

        all_bins_syn_nonsyn = list(range(min(numpy.concatenate((syn_sites, nonsyn_sites), axis=0)), max(numpy.concatenate((syn_sites, nonsyn_sites), axis=0))-1, bin_n))
        binned_n_mut_syn = numpy.asarray([sum((cumulative_block_position_syn >= all_bins_syn_nonsyn[i]) & (cumulative_block_position_syn < all_bins_syn_nonsyn[i+1])) for i in range(len(all_bins_syn_nonsyn)-1)])
        binned_n_mut_nonsyn = numpy.asarray([sum((cumulative_block_position_nonsyn >= all_bins_syn_nonsyn[i]) & (cumulative_block_position_nonsyn < all_bins_syn_nonsyn[i+1])) for i in range(len(all_bins_syn_nonsyn)-1)])

        binned_n_sites_syn = numpy.asarray([sum((syn_sites >= all_bins_syn_nonsyn[i]) & (syn_sites < all_bins_syn_nonsyn[i+1])) for i in range(len(all_bins_syn_nonsyn)-1)])
        binned_n_sites_nonsyn = numpy.asarray([sum((nonsyn_sites >= all_bins_syn_nonsyn[i]) & (nonsyn_sites < all_bins_syn_nonsyn[i+1])) for i in range(len(all_bins_syn_nonsyn)-1)])


    else:
        all_bins = None
        binned_divergence = None

    # calculate fraction of shared blocks 
    len_fraction_shared_blocks = cumulative_inter_block_len/cumulative_block_len
    len_fraction_shared_blocks_union = cumulative_inter_block_len/(cumulative_union_block_len)

    #if syn_sites_dict != None:
        #total_divergence_nonsyn = cumulative_n_nonsyn/cumulative_block_len_nonsyn
        #total_divergence_syn = cumulative_n_syn/cumulative_block_len_syn
        
    if syn_sites_dict == None:
        #total_divergence_nonsyn = None
        #total_divergence_syn = None
        cumulative_n_nonsyn = None
        cumulative_n_syn = None
        cumulative_block_len_nonsyn = None
        cumulative_block_len_syn = None


    if cumulative_inter_block_len == 0:
        total_divergence = None
    else: 
        total_divergence = len(cumulative_block_position)/cumulative_inter_block_len


    return all_bins, binned_divergence, total_divergence, cumulative_n_nonsyn, cumulative_n_syn, cumulative_block_len_nonsyn, cumulative_block_len_syn, len_fraction_shared_blocks, len_fraction_shared_blocks_union




def calculate_reverse_complement_sequence(dna_sequence):
    
    return "".join(base_table[base] for base in dna_sequence[::-1])






def parse_annotation_table():

    '''
    Parses annotation file provided by Beatriz
    Start and stop refer to positions in the *unaligned* genomes
    
    '''

    #annotation_table_file = gzip.open('%stop40_all_proteins_norev_good.tbl.gz' % config.data_directory, "rt")
    annotation_table_file = gzip.open('%sgutsy25_all_metadata_phold.tsv.gz' % config.data_directory, "rt")
    header = annotation_table_file.readline().strip().split('\t')
    transl_table_idx = header.index('transl_table')
    region_idx = header.index('Region')

    annotation_dict = {}
    for line in annotation_table_file:
        line_split = line.strip().split('\t')

        if len(line_split) < 71:
            continue

        genome = line_split[4]

        if 'CDS' in line_split:
            
            # initialize entry
            if genome not in annotation_dict:

                annotation_dict[genome] = {}
                annotation_dict[genome]['gene_all'] = []
                annotation_dict[genome]['start_all'] = []
                annotation_dict[genome]['stop_all'] = []
                annotation_dict[genome]['frame_all'] = []
                annotation_dict[genome]['transl_table_all'] = []
                annotation_dict[genome]['region_all'] = []

            annotation_dict[genome]['gene_all'].append(line_split[0])
            annotation_dict[genome]['start_all'].append(int(line_split[1]))
            annotation_dict[genome]['stop_all'].append(int(line_split[2]))
            annotation_dict[genome]['frame_all'].append(line_split[3])
            #annotation_dict[genome]['type_all'].append(line_split[2])
            annotation_dict[genome]['transl_table_all'].append(int(line_split[transl_table_idx]))
            annotation_dict[genome]['region_all'].append(line_split[region_idx])

        #if 'product' in line_split:
        #    annotation_dict[genome]['product_all'].append(line_split[1])

        #if 'locus_tag' in line_split:
        #    annotation_dict[genome]['locus_tag_all'].append(line_split[1])

        #if 'transl_table' in line_split:
        #    annotation_dict[genome]['transl_table_all'].append(int(line_split[1]))


    annotation_table_file.close()

    # save dictionary
    sys.stderr.write("Saving dictionary...\n")
    with open(annotation_dict_path, 'wb') as handle:
        pickle.dump(annotation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")






def make_syn_sites_votu_dict_from_pangraph(votu):

    '''
    Creates a fourfold status site dictionary for each genome in an OTU using annotation data.
    Saves dictionary as a pickle file for each vOTU
    '''
    
    # annotation dict is of all genomes
    annotation_dict = pickle.load(open(annotation_dict_path, "rb"))

    # all translation code 11
    #for key in annotation_dict.keys():
    #    if set(annotation_dict[key]['transl_table_all']) != {11}:
    #        print(set(annotation_dict[key]['transl_table_all']))

    sys.stderr.write("Identifying fourfold status of mutations in %s....\n" % votu)
    # get fasta files for the votu
    fasta_all_genomes = classFASTA('%suhgv_mgv_otu_fna/%s.fna' % (config.data_directory, votu)).readFASTA()
    fasta_genome_dict = {x[0]:x[1] for x in fasta_all_genomes}

    # get fasta files for the votu
    pangraph_file_path_all = glob.glob('%sSingle_vOTUs/Pangraphs/*%s*.json' % (config.data_directory, votu))

    if len(pangraph_file_path_all) == 0:
        return None
    else:
        pangraph_file_path = pangraph_file_path_all[0]

    pangraph_votu = load_pangraph_data(pangraph_file_path)
    #pan = pypangraph.Pangraph.from_json(pangraph_file_path)
    pan = pypangraph.Pangraph.load_json(pangraph_file_path)
    loc = pypangraph.Locator(pan)
    genomes_to_examine = list(set(fasta_genome_dict.keys()) & set(annotation_dict.keys()))

    start_stop_genome_dict = {}
    # numpy arrays for ease
    for genome_ in genomes_to_examine:

        start_stop_genome_dict[genome_] = {}
        start_all = numpy.asarray(annotation_dict[genome_]['start_all'])
        stop_all = numpy.asarray(annotation_dict[genome_]['stop_all'])
        
        start_stop_matrix = numpy.vstack((start_all, stop_all))
        max_position_over_genes = numpy.max(start_stop_matrix, axis=0)
        min_position_over_genes = numpy.min(start_stop_matrix, axis=0)
        
        start_stop_genome_dict[genome_]['start_all'] = start_all
        start_stop_genome_dict[genome_]['stop_all'] = stop_all
        start_stop_genome_dict[genome_]['max_position_over_genes'] = max_position_over_genes
        start_stop_genome_dict[genome_]['min_position_over_genes'] = min_position_over_genes


    syn_sites_dict = {}
    # loop through blocks
    for block_i in pangraph_votu['blocks']:

        block_i_id = block_i['id']

        if block_i_id not in syn_sites_dict:   
            syn_sites_dict[block_i_id] = {}
            syn_sites_dict[block_i_id]['data'] = {}
            syn_sites_dict[block_i_id]['keep_block'] = True

        
        block_i_mutate = block_i['mutate']
        bl = pan.blocks[block_i_id]
        aln = bl.alignment
        sequences, occurrences = aln.generate_sequences()
        occurrences_genome_names = [x[0] for x in occurrences]

        # looping through genomes
        syn_sites_dict[block_i_id]['data']['genomes'] = {}
        for j in range(0, len(block_i['positions'])):
            
            positions_i_j = block_i['positions'][j]
            genome_name_i_j = positions_i_j[0]['name']

            # strand refers to the orientation of the fragment of genome j that has block_i
            strand_i_j = positions_i_j[0]['strand']
            # positions refer to the positions of the block in the FASTA file
            min_position_i_j, max_position_i_j = positions_i_j[1]

            start_all_j = start_stop_genome_dict[genome_name_i_j]['start_all']
            stop_all_j = start_stop_genome_dict[genome_name_i_j]['stop_all']
            max_position_over_genes_j = start_stop_genome_dict[genome_name_i_j]['max_position_over_genes']
            min_position_over_genes_j = start_stop_genome_dict[genome_name_i_j]['min_position_over_genes']
            fasta_genome_j = fasta_genome_dict[genome_name_i_j]

            # returns the sequence of the block for the genome with insertions and deletions added
            # all items in sequences are ordered with respect to the block sequence
            # AKA the reverse complement has not been calculated
            block_sequence_j = sequences[occurrences_genome_names.index(genome_name_i_j)]

            # reverse complement
            # this sequence should match its sequence in the fasta file
            if strand_i_j == False:
                block_sequence_j = calculate_reverse_complement_sequence(block_sequence_j)


            if max_position_i_j > min_position_i_j:
                len_position_i_j = max_position_i_j - min_position_i_j + 1
            else:
                len_position_i_j = (len(fasta_genome_j) - min_position_i_j) + max_position_i_j + 1


            if (len_position_i_j !=  len(block_sequence_j)):
                #print('Reconstituted sequence not equal to positions!')
                #sys.stderr.write("Reconstituted sequence not equal to positions!\n")
                # ignore block if we cant examine it in a single genome
                syn_sites_dict[block_i_id]['keep_block'] = False
                continue


            block_i_mutate_genome_j = [x[1] for x in block_i_mutate if x[0]['name'] == genome_name_i_j][0]

            # no mutations
            if len(block_i_mutate_genome_j) == 0:
                syn_sites_dict[block_i_id]['keep_block'] = False
                continue


            # If the block wraps circular we do two searches
            # The one at the end of molecule, and the second at the beginning of molecule
            # when min_position_i_j > max_position_i_j the block goes from the end of the fasta to the beginning
            # we are only looking at **complete** genes within a fragment
            if (min_position_i_j > max_position_i_j):
                
                new_max_position = len(fasta_genome_j)
                # counting starts at one for gene positions
                new_min_position = 1

                # get genes from min_position_i_j to end of genomes
                genes_1_idx = (max_position_over_genes_j <= new_max_position) & (min_position_over_genes_j >= min_position_i_j)
                genes_2_idx = (max_position_over_genes_j <= max_position_i_j) & (min_position_over_genes_j >= new_min_position)
                idx_to_keep = (genes_1_idx | genes_2_idx)


            else:
                # Identify genes are contained by the block
                idx_to_keep = (max_position_over_genes_j <= max_position_i_j) & (min_position_over_genes_j >= min_position_i_j)
            

            min_position_to_keep = start_all_j[idx_to_keep]
            max_position_to_keep = stop_all_j[idx_to_keep]

            # skip if there are no genes in this block for this genome
            if (len(min_position_to_keep) == 0):
                continue
            
            # we have the positions of the genes in this block for this genome.
            # find the positions of the synonymous sites            
            # loop through genes in the chunk
            sites_synon_status_all = []
            sites_position_in_fasta_all = []
            for gene_g_idx in range(len(min_position_to_keep)):

                min_position_g = min_position_to_keep[gene_g_idx]
                max_position_g = max_position_to_keep[gene_g_idx]

                # get sequence from fasta
                # reverse complement
                if min_position_g > max_position_g:
                    # check this with real annotated genomes
                    seq_g = fasta_genome_j[max_position_g-1:min_position_g]
                    # reorient sequence
                    gene_sequence_g = calculate_reverse_complement_sequence(seq_g)
                    # swap order
                    min_position_g, max_position_g = max_position_g, min_position_g


                else:
                    # gene positions start counting at one
                    # index at zero for python
                    gene_sequence_g = fasta_genome_j[min_position_g-1:max_position_g]

                # loop among positions within the gene
                for position_in_gene in list( range(len(gene_sequence_g))): #calculate codon start
                    # start position of codon
                    codon_start = int((position_in_gene)/3)*3
                    codon = gene_sequence_g[codon_start:codon_start+3] 
                    position_in_codon = position_in_gene%3
                    
                    sites_synon_status_all.append(codon_synonymous_opportunity_table[codon][position_in_codon])
                    sites_position_in_fasta_all.append(min_position_g + position_in_gene)


            # the positions of each fasta position in the current block
            sites_position_in_block_all = [loc.find_position(strain=genome_name_i_j, pos=h)[1] for h in sites_position_in_fasta_all]
            # the positions and allele of each mutation in the current block that are in CDS regions

            # no CDS in the block! skip.
            if len(sites_position_in_block_all) == 0:
                continue


            # sites_synon_status has the positions of the sites in the gene with respect to the fasta file
            syn_sites_dict[block_i_id]['data']['n_sites'] = {}
            # sites_synon_status_all
            sites_synon_status_all = numpy.asarray(sites_synon_status_all)
            # count number of sites in block for each fourfold status
            for fourfold_status in range(4):
                syn_sites_dict[block_i_id]['data']['n_sites'][fourfold_status] = sum(sites_synon_status_all == fourfold_status)
            

            syn_sites_dict[block_i_id]['data']['genomes'][genome_name_i_j] = {}
            syn_sites_dict[block_i_id]['data']['genomes'][genome_name_i_j]['site_block_position'] = sites_position_in_block_all
            syn_sites_dict[block_i_id]['data']['genomes'][genome_name_i_j]['site_fasta_position'] = sites_position_in_fasta_all
            syn_sites_dict[block_i_id]['data']['genomes'][genome_name_i_j]['site_syn_status'] = sites_synon_status_all

            # strand of the block on the fasta file
            syn_sites_dict[block_i_id]['data']['genomes'][genome_name_i_j]['strand'] = strand_i_j


    # save dictionary
    sys.stderr.write("Saving dictionary...\n")
    with open(syn_sites_pangraph_dict_path % votu, 'wb') as handle:
        pickle.dump(syn_sites_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")





def make_syn_sites_votu_dict_from_alignment(votu):
    
    '''
    Creates a fourfold status site dictionary for each genome in a vOTU using annotation data.
    Annotations are determined using the unaligned fasta
    Saves dictionary as a pickle file for each vOTU
    '''
    
    # annotation dict is of all genomes
    annotation_dict = pickle.load(open(annotation_dict_path, "rb"))
    
    sys.stderr.write("Identifying fourfold status of mutations in %s....\n" % votu)
    # get fasta files for the votu
    # get unaligned genomes    
    fasta_all_genomes = classFASTA('%suhgv_mgv_otu_fna/%s.fna' % (config.data_directory, votu)).readFASTA()
    fasta_genome_dict = {x[0]:x[1] for x in fasta_all_genomes}

    genomes_to_examine = list(set(fasta_genome_dict.keys()) & set(annotation_dict.keys()))
    
    syn_sites_dict = {}
    
    for genome in genomes_to_examine:
                
        fasta_genome = fasta_genome_dict[genome]
                
        for g_idx in range(len(annotation_dict[genome]['start_all'])):
            
            # check to make sure it's protein coding 
            if annotation_dict[genome]['region_all'][g_idx] != 'CDS':
                continue
            
            # start_g starts counting at ONE
            start_g = annotation_dict[genome]['start_all'][g_idx]
            stop_g = annotation_dict[genome]['stop_all'][g_idx]
            frame_g = annotation_dict[genome]['frame_all'][g_idx]
            #transl_table_g = annotation_dict[genome]['transl_table_all'][g_idx]
            # transl_table_all == 11 for all genes
            
            # reverse complement
            if frame_g == '-':
                seq_g = fasta_genome[stop_g-1:start_g]
                seq_g = calculate_reverse_complement_sequence(seq_g)
                
            else:
                seq_g = fasta_genome[start_g-1:stop_g]
                
                        
            # check if we need the reverse complement
            #print(annotation_dict[genome]['frame_all'])
            # minus reading frame
            #if frame_g == '-':
            #    seq_g = calculate_reverse_complement_sequence(seq_g)
            
            if genome not in syn_sites_dict:
                syn_sites_dict[genome] = {}
                
            #if len(seq_g) == 0:
            #    print(seq_g)
            
            syn_status_g = []
            # loop among positions within the gene
            for position_in_gene in list(range(len(seq_g))): #calculate codon start
                # start position of codon
                   
                codon_start = int((position_in_gene)/3)*3
                codon = seq_g[codon_start:codon_start+3] 
                position_in_codon = position_in_gene%3                
                syn_status_g.append(codon_synonymous_opportunity_table[codon][position_in_codon])
            
            # if the gene is on the '-' strand,
            # then the positions of the fourfold status do not match 
            # the positions of the alleles
            if frame_g == '-':
                # so the positions match for reverse complement
                syn_status_g = syn_status_g[::-1]
                                                
                                                
            for position_in_gene in list(range(len(seq_g))):
                # counting here starts at ONE
                syn_sites_dict[genome][start_g+position_in_gene] = syn_status_g[position_in_gene]


    
    sys.stderr.write("Saving dictionary...\n")
    with open(syn_sites_alignment_dict_path % votu, 'wb') as handle:
        pickle.dump(syn_sites_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")
        
    
    
    
    
    
    




def computed_poisson_thinning(diffs, opportunities):
    # apply this to calculation of all dN/dS
    # specifically when calculating dS
    thinned_diffs_1 = numpy.random.binomial(diffs, 0.5)
    thinned_diffs_2 = diffs - thinned_diffs_1
    d1 = thinned_diffs_1 / (opportunities.astype(float) / 2)
    d2 = thinned_diffs_2 / (opportunities.astype(float) / 2)
    return d1, d2






        
    
    

def make_survival_dist(data, range_, probability=True):

    data = data[numpy.isfinite(data)]
    survival_array = [sum(data>=i) for i in range_]
    #survival_array = [sum(data>=i)/len(data) for i in range_]
    survival_array = numpy.asarray(survival_array)

    if probability == True:
        survival_array = survival_array/len(data)

    return survival_array



def map_sites_core_to_original(votu):
    
    '''
    Function adapted from Jimmy's script to create a dictionary that maps the position of sites in the aligned genomes
    to their corresponding sites in the unaligned genomes
    
    Positions for both aligned and unaligned genomes start at ONE (not zero)
    '''
    
    # get fasta files for the votu
    pangraph_file_path_all = glob.glob('%sSingle_vOTUs/Pangraphs/*%s*.json' % (config.data_directory, votu))

    if len(pangraph_file_path_all) == 0:
        return None
    else:
        pangraph_file_path = pangraph_file_path_all[0]
        
        
    sys.stderr.write("Making map between original and core sites for vOTU %s...\n" % votu)
    # Load the Pangraph
    pan = pypangraph.Pangraph.load_json(pangraph_file_path)
    loc = pypangraph.Locator(pan)

    # Extract paths and block statistics
    pd = pan.to_paths_dict()
    bs = pan.to_blockstats_df()

    # Filter for core blocks
    bs_core = bs[bs['core'] == True]
    core_blocks = bs_core.index

    # Get all strain names
    strains = list(pd.keys())

    reference_strain = strains[0]

    #Create dictionary from core genome block and block position to core alignment position
    core_block_position_to_core_alignment = {}

    # Use the reference strain's path to define core genome block order
    reference_path = pd[reference_strain]
    core_reference_path = [x for x in reference_path if x in core_blocks]

    offset = 0  # running coordinate in the core alignment

    for block_id in core_reference_path:
        bl = pan.blocks[block_id]
        aln = bl.generate_alignment()
        block_len = len(aln[0])

        for pos_in_block in range(0, block_len):  
            key=offset + pos_in_block+1
            value = (block_id, pos_in_block+1)
            core_block_position_to_core_alignment[key] = value

        offset += block_len
    core_alignment_length = offset

    for strain1 in strains:
        
        original_alignment_to_block={}
        
        for i in range(0, 1000000000):
            try:
                bl_id, bl_pos, occ = loc.find_position(strain=strain1, pos=i)
                original_alignment_to_block[(bl_id, bl_pos)] = i
            except:
                break

        core_alignment_to_original_alignment = {}
        count=0
        
        for i in range(1, core_alignment_length+1):
            try:
                value=core_block_position_to_core_alignment[i]
                mapped_position = original_alignment_to_block[value]
                core_alignment_to_original_alignment[i]=mapped_position
                count+=1
            except KeyError:
                continue
        #create txt file and save dictionary
        #output_file = os.path.join(output_dir, f'{strain1}_core_alignment_to_original.pkl')
                
        output_file = map_sites_core_to_original_path % strain1
        with open(output_file, 'wb') as f:
            pickle.dump(core_alignment_to_original_alignment, f)

    


def build_allele_counts_map(votu):
    
    '''
    Function that creates and saves a dictionary that contains the alleles and fourfold annotation status for each 
    site in the *aligned* genomes
    Analogous to Ben's parse_snps() function...
    '''
        
    sys.stderr.write("Loading aligned FASTA for vOTU %s...\n" % votu)
    fasta_file_path_all = glob.glob('%sSingle_vOTUs/Core_Alignments/*%s*.fna' % (config.data_directory, votu))
    
    if len(fasta_file_path_all) == 0:
        return None
    else:
        fasta_file_path = fasta_file_path_all[0]
        
    fasta_all_genomes = classFASTA(fasta_file_path).readFASTA()
    fasta_genome_dict = {x[0]:x[1] for x in fasta_all_genomes}
    
    sys.stderr.write("Loading fourfold site map....\n")
    fourfold_dict = pickle.load(open(syn_sites_alignment_dict_path % votu, "rb"))
    
    # Genomes present in both annotation file and the aligned fasta
    genomes_to_examine = list(set(fasta_genome_dict.keys()) & set(fourfold_dict.keys()))
    
    # site maps
    sys.stderr.write("Loading aligned-to-unaligned site map....\n")
    map_sites_core_to_original_all = {}
    for g in genomes_to_examine:
        
        #map_sites_core_to_original_g = pickle.load(open(map_sites_core_to_original_path % g, "rb"))
        # reverse dictionary so that value becomes key and vice-versa
        #map_sites_original_to_core_g_rev = {v: k for k, v in map_sites_original_to_core_g.items()}
        map_sites_core_to_original_all[g] = pickle.load(open(map_sites_core_to_original_path % g, "rb"))
        
        #print(min(map_sites_core_to_original_all[g].keys()), max(map_sites_core_to_original_all[g].keys()))
    
    

    fasta_genomes = [g[0] for g in fasta_all_genomes if g[0] in genomes_to_examine]
    fasta_seqs = [g[1] for g in fasta_all_genomes if g[0] in genomes_to_examine]
    # zip fasta so that each sub list is a site in the genome
    fasta_seqs_align = [list(x) for x in zip(*fasta_seqs)]
    
    allele_counts_map = {}
    allele_counts_map['genomes'] = fasta_genomes
    allele_counts_map['aligned_sites'] = {}
        
    for s_idx in range(len(fasta_seqs_align)):
        
        alleles_s = fasta_seqs_align[s_idx]
        site_g_aligned = s_idx+1
        
        fourfold_site = []
        #unaligned_position = []
        for g in fasta_genomes:
            
            # positions in map_sites_core_to_original_all start counting at ONE for both aligned and unaligned positions
            # the aligned site is not in the unaligned genome, meaning there is no allele
            if site_g_aligned not in map_sites_core_to_original_all[g]:
                fourfold_g_site = None
                
            else:
                # count starts at one
                site_g_unaligned = map_sites_core_to_original_all[g][s_idx+1]
            
                # get fourfold status 
                if site_g_unaligned not in fourfold_dict[g]:
                    fourfold_g_site = None
                else:
                    fourfold_g_site = fourfold_dict[g][site_g_unaligned]
                    
            
            fourfold_site.append(fourfold_g_site)      
                
        # site_g_aligned position starts counting at ONE 
        # because the aligned (pangraph) and unaligned (annotation) data starts counting at one
        allele_counts_map['aligned_sites'][site_g_aligned] = {}
        allele_counts_map['aligned_sites'][site_g_aligned]['alleles'] = alleles_s
        allele_counts_map['aligned_sites'][site_g_aligned]['fourfold_status'] = fourfold_site
        #allele_counts_map['aligned_sites'][site_g_aligned]['unaligned_position'] = fourfold_site
            
    
    sys.stderr.write("Saving dictionary...\n")
    allele_counts_map_path_ = allele_counts_map_path % votu
    with open(allele_counts_map_path_, 'wb') as f:
        pickle.dump(allele_counts_map, f)
    sys.stderr.write("Done!\n")
        
        
        

    
    
def filter_allele_counts_map(allele_counts_map, max_fraction_nan=0, min_sample_size=0, only_biallelic=True):
    
    '''
    only_biallelic: boolean variable asking whether to only return sites with two alleles (True) or one and two alleles (False) 
    
    returns the dictionary
    no_nan_bool_idx = bool for whether that that genome has (True) or does not have (False) a nucleotide at that site
    allele_bool_idx = bool for whether that that genome has the minor (True) or major allele (False)
    
    no_nan_bool_idx * allele_bool_idx: index of genomes where an allele is present and where it is *minor*
    '''
    
    sites =  list(allele_counts_map['aligned_sites'].keys())
    
    genomes = allele_counts_map['genomes']
    #genome_pairs_idx = list(combinations(range(len(genomes)), 2))
    n_genomes = len(genomes)
    
    if n_genomes < min_sample_size:
        return {}
    
    else:
        
        allele_counts_map_new = {}
        allele_counts_map_new['genomes'] = genomes
        allele_counts_map_new['aligned_sites'] = {}

        for s in sites:
            
            alleles_s = allele_counts_map['aligned_sites'][s]['alleles']
            fraction_nan = alleles_s.count('-')/n_genomes
            
            # insufficient number of informative sites, or
            # no nucleotides in any genome, uninformative
            if (fraction_nan > max_fraction_nan) or (fraction_nan == 1.0):
                continue 
        
            allele_count_dict = dict(Counter(alleles_s))
            nucleotide_intersect = set(allele_count_dict.keys()) & set(nucleotides)
            
            # no nucleotides in any genome, uninformative
            #if len(nucleotide_intersect) == 0:
            #    continue
            
            # If True, we do not care about invariant sites 
            if only_biallelic == True:
            
                if len(nucleotide_intersect) != 2:
                    continue
                    
            # define major allele            
            major_allele = max(list(nucleotide_intersect), key=lambda k: allele_count_dict[k])
        
            allele_counts_map_new['aligned_sites'][s] = {}
            #major_allele = list(nucleotide_intersect - set(minor_allele))[0]
            allele_counts_map_new['aligned_sites'][s]['major_allele'] = major_allele
            allele_counts_map_new['aligned_sites'][s]['alleles'] = alleles_s
            #allele_counts_map_new['aligned_sites'][s]['n_alleles'] = len(nucleotide_intersect)
            allele_counts_map_new['aligned_sites'][s]['n_obs_no_nan'] = sum(allele_count_dict.values())
            
            # make numpy arrays
            no_nan_bool_idx = numpy.asarray([x != '-' for x in  alleles_s])
            # True if = minor allele or '-'
            allele_bool_idx = numpy.asarray([x != major_allele for x in  alleles_s])
            
            allele_counts_map_new['aligned_sites'][s]['no_nan_bool_idx'] = no_nan_bool_idx
            allele_counts_map_new['aligned_sites'][s]['allele_bool_idx'] = allele_bool_idx
            
            fourfold_status = allele_counts_map['aligned_sites'][s]['fourfold_status']
            allele_counts_map_new['aligned_sites'][s]['fourfold_status'] = numpy.asarray(fourfold_status)
        
    
        return allele_counts_map_new

    




if __name__ == "__main__":

    votu = 'vOTU-000001'

    parse_annotation_table()
    
    #make_syn_sites_votu_dict_from_alignment(votu)
    #build_allele_counts_map(votu)

    #make_syn_sites_votu_dict_from_alignment(votu)
    
    votu_all = get_single_votus()

    for votu in votu_all:

        continue
                
        #print(votu)
        
        ##map_sites_original_to_core(votu)
        make_syn_sites_votu_dict_from_alignment(votu)
        build_allele_counts_map(votu)
    
    
    
        
    
    
            
        
    
    
    
    
    
