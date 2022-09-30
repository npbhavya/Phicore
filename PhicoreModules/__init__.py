"""
the modules
"""

from .read_genbank import parse_genbank
from .get_features import get_features_of_type, get_rolling_deltas,get_rolling_count_cds, get_cds_count_length_rec_window, get_gc_content, get_features_lengths, get_coding_density, get_distribution_of_stops, get_mean_cds_length_rec_window, get_rolling_gc,  get_rolling_mean_cds
from .stats import mean, median, mode, stdev
from .visualise import visualise_in_subplots, visualise_dataframe
from .write import write_df_to_artemis
from .dna import rc, shannon, kmers, non_overlapping_kmers
from .sequences import read_fasta


__all__ = [
    'parse_genbank', 'mean', 'median', 'mode', 'stdev',
    'get_features_of_type', 'get_gc_content', 'get_features_lengths', 'get_coding_density', 'get_distribution_of_stops', 'get_mean_cds_length_rec_window', 'get_rolling_gc', 'get_rolling_mean_cds', 'get_rolling_mean_cds_delta',
    'visualise_in_subplots', 'visualise_dataframe',
    'write_df_to_artemis', 'rc', 'shannon', 'kmers', 'read_fasta', 'non_overlapping_kmers'
]

