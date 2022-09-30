import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from scipy.stats import entropy
from typing import List, Union
import os
import sys
import gzip
from Bio import SeqIO
import binascii


__author__ = 'Przemyslaw Decewicz; George Bouras'

# Przemyslaw Decewicz
def get_features_of_type(seqiorec: SeqRecord, ftype: str) -> List[SeqFeature]:
    """
    Get features of a given type from SeqRecord
    :param seqiorec: a SeqRecord object
    :param ftype: type of a feature
    :return:
    """

    flist = []
    for feature in seqiorec.features:
        if feature.type == ftype:
            flist.append(feature)
    
    return flist

def get_gc_content(seq: Union[str, Seq, SeqRecord]) -> float:
    """
    Calculate GC content of a nucleotide sequence
    :param seq: a nucleotide sequence
    :return:
    """

    gc = 0
    for i in seq:
        if i == 'G' or i == 'C':
            gc += 1
    return gc / len(seq)

def get_features_lengths(seqiorec: SeqRecord, ftype: str) -> List[float]:
    """
    Get average length of SeqFeatures of a given type
    :param seqiorec: a SeqRecord object
    :param ftype: type of a feature
    :return:
    """

    lengths = []
    for feature in seqiorec.features:
        if feature.type == ftype:
            lengths.append(float(len(feature.location.extract(seqiorec).seq)))

    if ftype == 'CDS':
        return [x / 3 for x in lengths]
    else:
        return lengths

def get_coding_density(seqiorec: SeqRecord, ftypes: List[str] = ['CDS', 'tRNA', 'rRNA']) -> float:
    """
    Get coding density for a SeqRecord considering given features types
    :param seqiorec: SeqRecord object
    :param ftypes: a list of feature types
    :return:
    """

    cdcov = np.zeros(len(seqiorec.seq))
    for feature in seqiorec.features:
        if feature.type in ftypes:
            start, stop = map(int, sorted([feature.location.start, feature.location.end]))
            cdcov[start:stop] += 1
    return sum([1 if x > 0 else 0 for x in cdcov]) / len(seqiorec.seq)

def get_distribution_of_stops(seqiorec: SeqRecord, window: int = 210, step: int = 1) -> pd.DataFrame:
    """
    Get distribution of STOP codons in a sequence
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """

    stops = ['TAA', 'TAG', 'TGA']

    stops_distr = {
        'x': range(1, len(seqiorec.seq) + 1),
        'TAA': [np.NAN]*int(step/2),
        'TAG': [np.NAN]*int(step/2),
        'TGA': [np.NAN]*int(step/2)
    }
    
    i = 0
    while i < len(seqiorec.seq):
        window_seq = seqiorec.seq[i : i + window]
        taa = window_seq.count('TAA')
        tag = window_seq.count('TAG')
        tga = window_seq.count('TGA')
        stops_distr['TAA'].extend([taa]*(step))
        stops_distr['TAG'].extend([tag]*(step))
        stops_distr['TGA'].extend([tga]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(stops_distr['TAA'])
    if left > 0:   
        stops_distr['TAA'].extend([np.NAN]*left)
        stops_distr['TAG'].extend([np.NAN]*left)
        stops_distr['TGA'].extend([np.NAN]*left)

    return pd.DataFrame(stops_distr)

def get_codon_num_in_frame(seq: Union[str, SeqRecord, Seq], codon: str) -> int:
    """
    Count the number of codons in a sequence in the first frame.
    :param: seq: nucleotide sequence
    :param: codon: codon to count
    :return: number of codons
    """

    codons_num = 0
    for i in range(0, len(seq), 3):
        if seq[i:i+3] == codon:
            codons_num += 1
    return codons_num

def get_distribution_of_stops_per_frame(seqiorec : Union[str, SeqRecord, Seq], strand : int = 1, window : int = 210, step : int = 30):
    """
    Get distribution of stops
    :param seqiorec: nucleotide sequence to be searched
    :param strand: strand to be searched
    :param window: window size to consider
    :param step: step size to consider
    :return:
    """

    # change strand if needed
    if strand == -1:
        seqiorec = seqiorec.reverse_complement()

    stops = ['TAA', 'TAG', 'TGA']

    # the array starts and ends with NaNs because the distribution of certain stop codon is plotted in the middle of the window
    stops_frame_distr = {
        'x': range(1, len(seqiorec.seq) + 1),

        f'{strand}-TAA': [np.NAN]*int(window/2),
        f'{strand}-TAG': [np.NAN]*int(window/2),
        f'{strand}-TGA': [np.NAN]*int(window/2),

        f'{strand * 2}-TAA': [np.NAN]*int(window/2),
        f'{strand * 2}-TAG': [np.NAN]*int(window/2),
        f'{strand * 2}-TGA': [np.NAN]*int(window/2),

        f'{strand * 3}-TAA': [np.NAN]*int(window/2),
        f'{strand * 3}-TAG': [np.NAN]*int(window/2),
        f'{strand * 3}-TGA': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 3 <= len(seqiorec.seq) - window/2:
        for frame in range(3):
            taa = get_codon_num_in_frame(seqiorec.seq[i + frame : i + window + frame], 'TAA')
            tag = get_codon_num_in_frame(seqiorec.seq[i + frame : i + window + frame], 'TAG')
            tga = get_codon_num_in_frame(seqiorec.seq[i + frame : i + window + frame], 'TGA')
            stops_frame_distr[f'{strand * (frame + 1)}-TAA'].extend([taa]*(step))
            stops_frame_distr[f'{strand * (frame + 1)}-TAG'].extend([tag]*(step))
            stops_frame_distr[f'{strand * (frame + 1)}-TGA'].extend([tga]*(step))
            # print(frame, taa)
        i += step
        
    i -= step
    left = len(stops_frame_distr['x']) - len(stops_frame_distr[f'{strand}-TAA'])
    if left > 0:
        for frame in range(3):
            stops_frame_distr[f'{strand * (frame + 1)}-TAA'].extend([np.NAN]*left)
            stops_frame_distr[f'{strand * (frame + 1)}-TAG'].extend([np.NAN]*left)
            stops_frame_distr[f'{strand * (frame + 1)}-TGA'].extend([np.NAN]*left)

    return pd.DataFrame(stops_frame_distr)

def get_distribution_of_stops_for_all_strands(seqiorec : Union[str, SeqRecord, Seq], window : int = 210, step : int = 30):
    """
    Get distribution of stops for all strands
    :param seqiorec: nucleotide sequence to be searched
    :param window: window size to consider
    :param step: step size to consider
    :return:
    """

    # positive strand
    dfp = get_distribution_of_stops_per_frame(seqiorec, 1, window, step)
    # negative strand
    dfn = get_distribution_of_stops_per_frame(seqiorec, -1, window, step)
    dfn.sort_values(by="x", ascending=False, ignore_index=True, inplace=True)
    dfn.drop('x', axis=1, inplace = True)
    df = pd.concat([dfp, dfn], axis=1)

    return df

def filter_codons(seqiorec : Union[str, SeqRecord, Seq], codons : List[str] = ['TAA', 'TAG', 'TGA']):
    """
    Get only stops from a sequence
    :param seqiorec: nucleotide sequence to be searched
    :return:
    """

    if isinstance(seqiorec, SeqRecord):
        seq = str(seqiorec.seq)
    elif isinstance(seqiorec, Seq):
        seq = str(seqiorec)
        
    selected_codons = []
    for codon in [seq[i:i+3] for i in range(0, len(seq), 3)]:
        if codon in codons:
            selected_codons.append(codon)

    return selected_codons

def probability_of_codons(codons : List[str]):
    """
    Get frequency of codons
    :param codons: list of codons
    :return:
    """
    
    probability = []
    for codon in set(codons):
        probability.append(1/codons.count(codon))

    return probability

def get_entropy_of_stops_per_frame(seqiorec : Union[str, SeqRecord, Seq], strand : int = 1, window : int = 210, step : int = 30):
    """
    Get entropy of STOPs
    :param seqiorec: nucleotide sequence to be searched
    :param strand: strand to be searched
    :param window: window size to consider
    :param step: step size to consider
    :return:
    """

    # change strand if needed
    if strand == -1:
        seqiorec = seqiorec.reverse_complement()

    stops = ['TAA', 'TAG', 'TGA']

    # the array starts and ends with NaNs because the distribution of certain stop codon is plotted in the middle of the window
    stops_frame_entropy = {
        'x': range(1, len(seqiorec.seq) + 1),

        f'{strand}-H1': [np.NAN]*int(window/2),
        f'{strand}-H2': [np.NAN]*int(window/2),
        f'{strand}-H3': [np.NAN]*int(window/2),
    }
    
    i = 0
    while i + window/2 + 3 <= len(seqiorec.seq) - window/2:

        for frame in range(3):
            # get the stops that occur in sequence window
            frame_stops = filter_codons(seqiorec.seq[i + frame : i + window + frame], stops)
            # calculate the entropy of the these stops
            if len(frame_stops) > 1:
                frame_H = entropy(probability_of_codons(frame_stops))
            else:
                frame_H = 0
            # save result
            stops_frame_entropy[f'{strand}-H{(frame + 1)}'].extend([frame_H]*(step))
        i += step
    
    i -= step
    left = len(stops_frame_entropy['x']) - len(stops_frame_entropy[f'{strand}-H1'])
    if left > 0:
        for frame in range(3):
            stops_frame_entropy[f'{strand}-H{(frame + 1)}'].extend([np.NAN]*left)

    return pd.DataFrame(stops_frame_entropy)

def get_entropy_of_stops_for_all_strands(seqiorec : Union[str, SeqRecord, Seq], window : int = 210, step : int = 30):
    """
    Get entropy of stops for all strands
    :param seqiorec: nucleotide sequence to be searched
    :param window: window size to consider
    :param step: step size to consider
    :return:
    """

    # positive strand
    dfp = get_entropy_of_stops_per_frame(seqiorec, 1, window, step)
    # negative strand
    dfn = get_entropy_of_stops_per_frame(seqiorec, -1, window, step)
    dfn.sort_values(by="x", ascending=False, ignore_index=True, inplace=True)
    dfn.drop('x', axis=1, inplace = True)
    df = pd.concat([dfp, dfn], axis=1)

    return df

# George Bouras
def get_mean_cds_length_rec_window(seqiorec : SeqRecord, window_begin : int, window_end : int) -> float:
    """
    Get mean CDS length
    :param seqiorec: SeqRecord object
    :param window_begin: integer
    :param window_end: integer
    :return:
    """

    cds_length = []
    for feature in seqiorec.features:
        if feature.type == 'CDS':
            if feature.location.start > window_begin and feature.location.start < window_end and feature.location.end > window_begin and feature.location.end < window_end:
                cds_length.append(len(feature.location.extract(seqiorec).seq)/3)
    if len(cds_length) == 0:
        mean = (window_end - window_begin)/3
    else:
        mean = np.mean(cds_length)
    return mean


def get_cds_count_length_rec_window(seqiorec : SeqRecord, window_begin : int, window_end : int) -> float:
    """
    Get median CDS length
    :param seqiorec: SeqRecord object
    :param window_begin: integer
    :param window_end: integer
    :return:
    """

    cds_length = []
    count = 0 
    for feature in seqiorec.features:
        if feature.type == 'CDS':
            if feature.location.start > window_begin and feature.location.start < window_end and feature.location.end > window_begin and feature.location.end < window_end:
                count +=1
    return count


def get_rolling_gc(seqiorec : SeqRecord, window : int = 1000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """

    gcs = ['G', 'C']

    gcs_distr = {
        'x': range(1, len(seqiorec.seq) + 1),
        'G': [np.NAN]*int(window/2),
        'C': [np.NAN]*int(window/2),
        'GC': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        window_seq = seqiorec.seq[i : i + window]
        g = window_seq.count('G')
        c = window_seq.count('C')
        gcs_distr['G'].extend([g]*(step))
        gcs_distr['C'].extend([c]*(step))
        gcs_distr['GC'].extend([g+c]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(gcs_distr['G'])
    if left > 0:   
        gcs_distr['G'].extend([np.NAN]*left)
        gcs_distr['C'].extend([np.NAN]*left)
        gcs_distr['GC'].extend([np.NAN]*left)

    return pd.DataFrame(gcs_distr)

def get_rolling_mean_cds(seqiorec : SeqRecord, window : int = 1000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """
    cds_average = {
        'x': range(1, len(seqiorec.seq) + 1),
        'Mean_CDS': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        cds_mean = get_mean_cds_length_rec_window(seqiorec,i, i + window )
        cds_average['Mean_CDS'].extend([cds_mean]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(cds_average['Mean_CDS'])
    if left > 0:   
        cds_average['Mean_CDS'].extend([np.NAN]*left)


    return pd.DataFrame(cds_average)


def get_rolling_count_cds(seqiorec : SeqRecord, window : int = 1000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param seqiorec: SeqRecord object
    :param window: window size
    :param step: step size
    :return:
    """
    cds_count = {
        'x': range(1, len(seqiorec.seq) + 1),
        'Count_CDS': [np.NAN]*int(window/2)
    }
    
    i = 0
    while i + window/2 + 1 <= len(seqiorec.seq) - window/2:
        count = get_cds_count_length_rec_window(seqiorec,i, i + window )
        cds_count['Count_CDS'].extend([count]*(step))
        i += step
        
    i -= step
    left = len(seqiorec.seq) - len(cds_count['Count_CDS'])
    if left > 0:   
        cds_count['Count_CDS'].extend([np.NAN]*left)


    return pd.DataFrame(cds_count)





def is_gzip_file(f):
    """
    This is an elegant solution to test whether a file is gzipped by reading the first two characters.
    I also use a version of this in fastq_pair if you want a C version :)
    See https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed for inspiration
    :param f: the file to test
    :return: True if the file is gzip compressed else false
    """
    with open(f, 'rb') as i:
        return binascii.hexlify(i.read(2)) == b'1f8b'



def parse_genbank(filename, verbose=False):
    """
    Parse a genbank file and return a Bio::Seq object
    """


    try:
        if is_gzip_file(filename):
            handle = gzip.open(filename, 'rt')
        else:
            handle = open(filename, 'r')
    except IOError as e:
        print(f"There was an error opening {filename}", file=sys.stderr)
        sys.exit(20)

    return SeqIO.parse(handle, "genbank")

def get_rolling_deltas(genbank_path_all : str, genbank_path_tag : str, genbank_path_tga : str, genbank_path_taa : str, window : int = 2000, step : int = 1) -> pd.DataFrame:
    """
    Get distribution of stops
    :param window: window size
    :param step: step size
    :return:
    """

    for record in parse_genbank(genbank_path_all):
        df_all = get_rolling_mean_cds(record, window=2000, step=30)
        count_all = get_rolling_count_cds(record, window=2000, step=30)
        #stops_all = get_distribution_of_stops(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_tag):
        df_tag = get_rolling_mean_cds(record, window=2000, step=30)
        count_tag = get_rolling_count_cds(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_tga):
        df_tga = get_rolling_mean_cds(record, window=2000, step=30)
        count_tga = get_rolling_count_cds(record, window=2000, step=30)
    for record in parse_genbank(genbank_path_taa):
        df_taa = get_rolling_mean_cds(record, window=2000, step=30)
        count_taa = get_rolling_count_cds(record, window=2000, step=30)


    df_all['Mean_CDS_all'] = df_all['Mean_CDS'] 
    df_all['Mean_CDS_tag'] = df_tag['Mean_CDS'] 
    df_all['Mean_CDS_tga'] = df_tga['Mean_CDS'] 
    df_all['Mean_CDS_taa'] = df_taa['Mean_CDS'] 
    df_all['Count_CDS_all'] = count_all['Count_CDS'] 
    df_all['Count_CDS_tag'] = count_tag['Count_CDS'] 
    df_all['Count_CDS_tga'] = count_tga['Count_CDS'] 
    df_all['Count_CDS_taa'] = count_taa['Count_CDS'] 
    # df_all['TAG_all'] = stops_all['TAG'] 
    # df_all['TGA_all'] = stops_all['TGA'] 
    # df_all['TAA_all'] = stops_all['TAA'] 

    



    df_all['tag_minus_all_mean_cds'] = df_all['Mean_CDS_tag'] - df_all['Mean_CDS_all']
    df_all['tga_minus_all_mean_cds'] = df_all['Mean_CDS_tga'] - df_all['Mean_CDS_all']
    df_all['taa_minus_all_mean_cds'] = df_all['Mean_CDS_taa'] - df_all['Mean_CDS_all']
    df_all['tag_minus_all_count_cds'] = df_all['Count_CDS_tag'] - df_all['Count_CDS_all']
    df_all['tga_minus_all_count_cds'] = df_all['Count_CDS_tga'] - df_all['Count_CDS_all']
    df_all['taa_minus_all_count_cds'] = df_all['Count_CDS_taa'] - df_all['Count_CDS_all']

    df_all = df_all.drop(['Mean_CDS', 'Mean_CDS_all', 'Mean_CDS_tag', 'Mean_CDS_tga', 'Mean_CDS_taa', 'Count_CDS_all', 'Count_CDS_tag', 'Count_CDS_tga', 'Count_CDS_taa'], axis=1)

    return pd.DataFrame(df_all)

