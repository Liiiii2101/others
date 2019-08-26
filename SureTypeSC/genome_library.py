import sys
import os
import argparse
import logging
from math import isnan
from itertools import izip
from numpy import percentile
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd

from IlluminaBeadArrayFiles import LocusAggregate, BeadPoolManifest, GenotypeCalls, ClusterFile, RefStrand

nan = float("nan")

def map_name(data):
    frame  = pd.read_csv(data)
    value = frame.values
    for i in range(value.shape[0]):
        if value[i][0] == 'Sample_ID' :
            start = i+1
    value = value[start:]
    value = np.asarray(value)
    id_n = list(value[:,0])
    bar = value[:,1]
    pos = value[:,2]

    file_a = list(map(lambda x,y:x+'_'+y,bar,pos))
    #map_dict = dictionary = dict(zip(file, id_n))
    
    return id_n,file_a

def map_genotype(a):
    if a == 0:
        return 'NC'
    if a == 1:
        return 'AA'
    if a == 2:
        return 'AB'
    if a == 3:
        return 'BB'


class LocusSummary(object):
    def __init__(self, genotype_counts, score_stats,x_raw,y_raw):
        self.genotype_counts = genotype_counts
        self.score_stats = score_stats
        #self.genotype = genotype
        self.x_raw = x_raw
        self.y_raw = y_raw
        #sef.x_norm = x_norm
        #self.y_norm = y_norm

class GenotypeCounts(object):
    """
    Summarize information about genotype counts for diploid genotyping counting
    """

    def __init__(self, genotypes):
        self.no_calls = 0
        self.aa_count = 0
        self.ab_count = 0
        self.bb_count = 0

        for genotype in genotypes:
            if genotype == 0:
                self.no_calls += 1
            elif genotype == 1:
                self.aa_count += 1
            elif genotype == 2:
                self.ab_count += 1
            elif genotype == 3:
                self.bb_count += 1

    def get_num_calls(self):
        """
        Get the number of calls (i.e., not no-calls)

        Returns:
            int: The number of calls
        """
        return self.aa_count + self.ab_count + self.bb_count

    def get_call_frequency(self):
        """
        Get the call rate

        Returns:
            float: The frequency of calls
        """
        num_calls = self.get_num_calls()
        return num_calls / float(num_calls + self.no_calls) if num_calls + self.no_calls > 0 else nan

    def get_aa_frequency(self):
        """
        Frequency of AA genotype (as fraction of all calls)

        Returns:
            float: AA genotype frequency
        """
        return self.aa_count / float(self.get_num_calls()) if self.get_num_calls() > 0 else nan

    def get_ab_frequency(self):
        """
        Frequency of AB genotype (as fraction of all calls)

        Returns:
            float: AB genotype frequency
        """
        return self.ab_count / float(self.get_num_calls()) if self.get_num_calls() > 0 else nan

    def get_bb_frequency(self):
        """
        Frequency of BB genotype (as fraction of all calls)

        Returns:
            float: BB genotype frequency
        """
        return self.bb_count / float(self.get_num_calls()) if self.get_num_calls() > 0 else nan

    def get_minor_frequency(self):
        """
        Comoputes and return the minor allele frequency. If no calls, will be NaN

        Returns:
            float
        """
        a_allele_count = self.aa_count * 2 + self.ab_count
        a_frequency = a_allele_count / \
            float(2 * self.get_num_calls()) if self.get_num_calls() > 0 else nan
        return min(a_frequency, 1.0 - a_frequency) if not isnan(a_frequency) else nan

    def compute_hardy_weinberg(self):
        """
        Computes and returns statistics related to HW equilibrium

        Returns:
            (float, float): Het excess and ChiSq 100 statistics, respectively
        """
        num_calls = self.get_num_calls()
        if num_calls == 0:
            return (0.0, 0.0)

        if self.aa_count + self.ab_count == 0 or self.ab_count + self.bb_count == 0:
            return (1.0, 0.0)

        num_calls = float(num_calls)

        q = self.get_minor_frequency()
        p = 1 - q

        temp = 0.013 / q
        k = temp * temp * temp * temp
        dh = ((self.ab_count / num_calls + k) / (2 * p * q + k)) - 1
        if dh < 0:
            hw = (2 * norm.cdf(dh, 0, 1 / 10.0))
        else:
            hw = (2 * (1 - norm.cdf(dh, 0, 1 / 10.0)))

        return (hw, dh)


class ScoreStatistics(object):
    """
    Capture statistics related to the gencall score distribution

    Attributes:
        gc_10 : 10th percentile of Gencall score distribution
        gc_50 : 50th percentile of Gencall score distribution
    """

    def __init__(self, scores, genotypes):
        """
        Create new ScoreStatistics object

        Args:
            score (list(float)): A list of gencall scores
            genotypes (list(int)): A list of genotypes

        Returns:
            ScoreStatistics
        """
        #called_scores = sorted([score for (score, genotype) in zip(scores, genotypes) if genotype != 0])
        called_scores = sorted([score for (score, genotype) in zip(scores, genotypes)])
        self.gc_10 = ScoreStatistics.percentile(called_scores, 10)
        self.gc_50 = ScoreStatistics.percentile(called_scores, 50)
        self.called_scores = called_scores
        self.scores = scores
        self.genotypes = genotypes

    @staticmethod
    def percentile(scores, percentile):
        """
        Percentile as calculated in GenomeStudio

        Args:
            scores (list(float)): list of scores (typically for called genotypes)
            percentile (int): percentile to calculate
        
        Returns:
            float
        """
        num_scores = len(scores)
        if num_scores == 0:
            return nan

        idx = int(num_scores*percentile/100)
        fractional_index = num_scores*percentile/100.0 - idx
        if fractional_index < 0.5 :
            idx -= 1

        if idx < 0:
            return scores[0]

        if idx >= num_scores - 1:
            return scores[-1]

        x1 = 100 * (idx + 0.5)/float(num_scores)
        x2 = 100 * (idx + 1 + 0.5)/float(num_scores)
        y1 = float(scores[idx])
        y2 = float(scores[idx+1])

        return y1 + (y2 - y1) / (x2 - x1) * (percentile - x1)




def summarize_locus(locus_aggregate):
    """
    Generate a locus summary based on aggregated locus information

    Args:
        LocusAggregate : Aggregated information for a locus
    
    Returns
        LocusSummary
    """
    genotype_counts = GenotypeCounts(locus_aggregate.genotypes)
    score_stats = ScoreStatistics(locus_aggregate.scores, locus_aggregate.genotypes)
    return LocusSummary(genotype_counts, score_stats)

def basic_locus(locus_aggregate):
    """
    Generate a locus summary based on aggregated locus information

    Args:
        LocusAggregate : Aggregated information for a locus
    
    Returns
        LocusSummary
    """
    genotype_counts = GenotypeCounts(locus_aggregate.genotypes)
    score_stats = ScoreStatistics(locus_aggregate.scores, locus_aggregate.genotypes)
    #genotype = locus_aggregate.genotypes
    #score_stats = locus_aggregate.scores
    x_raw = locus_aggregate.x_intensities
    y_raw = locus_aggregate.y_intensities
    norm = locus_aggregate.transforms
    #print(x_raw[:20])
    #print(type(norm))
    #print(len(norm))
    #print(norm[:10])
    #(x_norm,y_norm) = 
    
    return LocusSummary(genotype_counts, score_stats,x_raw,y_raw)


def get_logger():
    # set up log file
    # create logger
    logger = logging.getLogger('Locus Summary Report')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger



def basic(gtc_dir, manifest_filename, cluster_filename, samplesheet, delim):
    print "Reading cluster file" 
    #with open(cluster_filename, "rb") as cluster_handle:
        #egt = ClusterFile.read_cluster_file(cluster_handle)

    print 'Reading sample file' 
    id_n,code = map_name(samplesheet)
    print 'Number of samples:',len(id_n)

    cate = []

    for i in range(len(id_n)):
        cate.append(id_n[i]+'.GType')
        cate.append(id_n[i]+'.Score')
        cate.append(id_n[i]+'.X')
        cate.append(id_n[i]+'.Y')
        cate.append(id_n[i]+'.X Raw')
        cate.append(id_n[i]+'.Y Raw')
    


    print "Reading manifest file" 
    bpm = BeadPoolManifest(manifest_filename)
    samples = []

    print  "Initializing genotype data" 

    count = 0
    gtc_files = []
    for gtc_file in code:
        gtc_file = gtc_file + '.gtc'
        gtc_files.append(os.path.join(gtc_dir, gtc_file))
        count +=1
        if count >=2:
            break

    samples = map(GenotypeCalls, gtc_files)

    print "Generating" 


    title = ["Name","Chr","Position"]

    locus_name = bpm.names

    #address = bpm.addresses

    chroms = bpm.chroms

    position = bpm.map_infos

    map_dict  = dict(zip(title, [locus_name,chroms,position]))

    df = pd.DataFrame(map_dict)

    #test_count =0

    for j in range(len(samples)):
        print code[j]
        sam = samples[j]

        i = 6*j
        gene = sam.get_genotypes()
        gene = map(map_genotype,gene)
        df[cate[i]] = gene

        scores = sam.get_genotype_scores()
        df[cate[i+1]] = scores

        norm = sam.get_normalized_intensities(bpm.normalization_lookups)
        norm = np.asarray(norm)
        x_norm = list(norm[:,0])
        y_norm = list(norm[:,1])
        df[cate[i+2]] = x_norm
        df[cate[i+3]] = y_norm

        x_raw = sam.get_raw_x_intensities()
        df[cate[i+4]] = x_raw
        y_raw = sam.get_raw_y_intensities()
        df[cate[i+5]] = y_raw

        #test_count +=1

        #if test_count >=2:
            #break




    
    print  "Finish parsing"
    #df.to_csv(output_filename, sep=delim,encoding='utf-8')

    return df








def statistic(gtc_dir, manifest_filename, cluster_filename, delim, output_filename):
    print "Reading cluster file" 
    with open(cluster_filename, "rb") as cluster_handle:
        egt = ClusterFile.read_cluster_file(cluster_handle)

    print "Reading manifest file"
    bpm = BeadPoolManifest(manifest_filename)
    samples = []

    print "Initializing genotype data"
    gtc_files = []
    for gtc_file in os.listdir(gtc_dir):
        if gtc_file.endswith(".gtc"):
            gtc_files.append(os.path.join(gtc_dir, gtc_file))
 

    samples = map(GenotypeCalls, gtc_files)

    #logger.info("Generating report")
    loci = range(len(bpm.normalization_lookups))
    print 'length of loci',len(loci)

    title = 'Row,Locus_Name,Illumicode_Name,#No_Calls,#Calls,Call_Freq,A/A_Freq,A/B_Freq,B/B_Freq,Minor_Freq,Gentrain_Score,50%_GC_Score,10%_GC_Score,Het_Excess_Freq,ChiTest_P100,Cluster_Sep,AA_T_Mean,AA_T_Std,AB_T_Mean,AB_T_Std,BB_T_Mean,BB_T_Std,AA_R_Mean,AA_R_Std,AB_R_Mean,AB_R_Std,BB_R_Mean,BB_R_Std,Plus/Minus Strand'
    
    title = title.split(',')

    df = pd.DataFrame(columns = title)
    #print LocusAggregate.aggregate_samples(samples, loci, basic_locus, bpm.normalization_lookups)

    #print (LocusAggregate.aggregate_samples(samples, loci, basic_locus, bpm.normalization_lookups)).genotype_counts.get_num_calls()

        #output_handle.write(delim.join("Row,Locus_Name,Illumicode_Name,#No_Calls,#Calls,Call_Freq,A/A_Freq,A/B_Freq,B/B_Freq,Minor_Freq,Gentrain_Score,50%_GC_Score,10%_GC_Score,Het_Excess_Freq,ChiTest_P100,Cluster_Sep,AA_T_Mean,AA_T_Std,AB_T_Mean,AB_T_Std,BB_T_Mean,BB_T_Std,AA_R_Mean,AA_R_Std,AB_R_Mean,AB_R_Std,BB_R_Mean,BB_R_Std,Plus/Minus Strand".split(",")) + "\n")
    for (locus, locus_summary) in izip(loci, LocusAggregate.aggregate_samples(samples, loci, basic_locus, bpm.normalization_lookups)):
        locus_name = bpm.names[locus]
        cluster_record = egt.get_record(locus_name)
        row_data = []
        row_data.append(locus + 1)
        row_data.append(locus_name)
        row_data.append(cluster_record.address)
        row_data.append(locus_summary.genotype_counts.no_calls)
        row_data.append(locus_summary.genotype_counts.get_num_calls())
        row_data.append(locus_summary.genotype_counts.get_call_frequency())
        row_data.append(locus_summary.genotype_counts.get_aa_frequency())
        row_data.append(locus_summary.genotype_counts.get_ab_frequency())
        row_data.append(locus_summary.genotype_counts.get_bb_frequency())
        row_data.append(
            locus_summary.genotype_counts.get_minor_frequency())
        row_data.append(cluster_record.cluster_score.total_score)
        row_data.append(locus_summary.score_stats.gc_50)
        row_data.append(locus_summary.score_stats.gc_10)

        (hw_equilibrium, het_excess) = locus_summary.genotype_counts.compute_hardy_weinberg()
        row_data.append(het_excess)
        row_data.append(hw_equilibrium)

        row_data.append(cluster_record.cluster_score.cluster_separation)

        for cluster_stats in (cluster_record.aa_cluster_stats, cluster_record.ab_cluster_stats, cluster_record.bb_cluster_stats):
            row_data.append(cluster_stats.theta_mean)
            row_data.append(cluster_stats.theta_dev)

        for cluster_stats in (cluster_record.aa_cluster_stats, cluster_record.ab_cluster_stats, cluster_record.bb_cluster_stats):
            row_data.append(cluster_stats.r_mean)
            row_data.append(cluster_stats.r_dev)

        if len(bpm.ref_strands) > 0:
            row_data.append(RefStrand.to_string(bpm.ref_strands[locus]))
        else:
            row_data.append("U")

        df.loc[locus] = row_data


        #print locus
    df.to_csv(output_filename, delim, encoding='utf-8')
    return df

            #output_handle.write(delim.join(map(str, row_data)) + "\n")
        #print "Report generation complete"






