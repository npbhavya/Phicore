# Phicore
Testing for codon reassignment in bacteriophages

### Installation 
**Pre-requisities**
- anaconda or miniconda already installed 
- python3
- Jupyter notebook or jupyter lab
  - To setup Jupyter notebook or lab on the cluster: https://fame.flinders.edu.au/blog/2022/07/16/jupyter-deepthought2
- Artemis genome browser: http://sanger-pathogens.github.io/Artemis/Artemis/

**Setting up the environment**

    ```
    conda create -n phicore 
    conda activate phicore
    #install trnascan-se
    conda install -c bioconda trnascan-se
    mamba install -c anaconda seaborn
    mamba install -c anaconda numpy
    mamba install -c conda-forge matplotlib
    mamba install -c conda-forge biopython
    mamba install -c bioconda dna_features_viewer 
    mamba install -y ipykernel
    #install phanotate dev version
    git clone -b develop https://github.com/deprekate/PHANOTATE.git
    cd PHANOTATE/
    python3 -m venv env
    source env/bin/activate
    python setup.py install
    ```

**Installing phicore**

    ```
    git clone https://github.com/npbhavya/Phicore.git
    ```

**Deactivating environment**

    ```
    #deactivate the virtual environment 
    deactivate 
    #deactivate the conda environment 
    conda deactivate
    ```

**Activating environment if already installed**

    ```
    conda activate phicore
    cd PHANOTATE
    source env/bin/activate
    ```

### Test data 
- Fasta files saved to "test-data"
- Genbank files saved to "genbank"

README.md added to both directories