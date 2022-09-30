"""
Plotting functions
"""

__author__ = 'Vijini Mallawaarachchi'


def visualise_in_subplots(df, length, labels, size1, size2, style=""):

    """
    Plot different dataframe columns (labels) in subplots

    df: dataframe
    length: length of genomes
    lables: lables of stats
    size1: figure width
    size2: figure height
    style: plot style (line and marker)
    """

    # Get axes
    axes = df.plot.line(subplots=True, figsize=(size1, size2), legend=False, style=style)

    # Set for each subplot axis
    for i in range(len(axes)):
        axes[i].set_xlim(0, length)
        axes[i].yaxis.grid()
        axes[i].set_axisbelow(True)
        axes[i].set(ylabel=labels[i])


def visualise_dataframe(df, df_x, df_y, size1, size2):

    """
    Plot dataframe in one plot

    df: dataframe
    x: column name to use in x axis
    y: list of column names to use for y axis
    size1: figure width
    size2: figure height
    """

    # Plot dataframe
    df.plot(x=df_x, y=df_y, figsize=(size1, size2))

