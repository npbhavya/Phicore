# Test genomes 

True positive: 
- GCA_002135175.2, Candidatus Spiroplasma holothuricola isolate MT37 subsample
- OFRY01000050.fasta, human gut metagenome genome assembly
- UAG-readthrough_crAss_clade_sp._strain_cr150_1.fasta

True negatives:
- Bc01.fasta 
- Bc02.fasta 
- Bc11.fasta
  
## Generating the genbank files
Phanotate run on all the 6 genomes for the below combination of stop codons
    - TAG, TGA, TAA (standard genetic code, 11)
    - TAG, TGA (genetic code 101)
    - TGA, TAA (genetic code 15)
    - TAG, TAA (genetic code 4 or 25)
    - TAA 
    - TGA
    - TAG

The commands, 

    ```
    phanotate.py "$f" -e TAG,TAA,TGA -f genbank -o "$f"-TAG-TGA-TAA.gbk 
    phanotate.py "$f" -e TAG,TGA -f genbank -o "$f"-TAG-TGA.gbk
    phanotate.py "$f" -e TGA,TAA -f genbank -o "$f"-TGA-TAA.gbk
    phanotate.py "$f" -e TAG,TAA -f genbank -o "$f"-TAG-TAA.gbk
    phanotate.py "$f" -e TAG -f genbank -o "$f"-TAG.gbk
    phanotate.py "$f" -e TGA -f genbank -o "$f"-TGA.gbk
    phanotate.py "$f" -e TAA -f genbank -o "$f"-TAA.gbk
    ```

