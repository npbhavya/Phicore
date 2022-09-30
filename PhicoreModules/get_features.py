import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from typing import List, Union

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


# George Bouras
def get_mean_cds_length_rec_window(seqiorec : SeqRecord, window_begin : int, window_end : int) -> float:
    """
    Get median CDS length
    :param seqiorec: SeqRecord object
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