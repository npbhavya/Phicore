"""
Make GC FramePlot for artemis with ability to change the window size and step size 

Exampe Usage: 
GCFramePlot('test-data/Bc01.fasta', 215, 15, 'test.txt').generateTable()
""" 

#imports 
import numpy as np 
from Bio import SeqIO

__author__ = 'Susie Grigson'

class GCFramePlot(): 
    """
    Generates a GCFramePlot which can be read into aretmis. 
    To read the output into artemis select 'Graph' -> 'User Plot' on artemis. 
    """ 
    
    def __init__(self, seq, window_size, step_size, outfile):
        """
        seq - fasta file of viral genome 
        window_size - window size for GCFramePlot
        step_size - step size used for GCFramePlot
        outfile - text file to save the GCFramePlot table 
        """ 
        
        #check parameters are multiples of 3 
        if window_size % 3 != 0: 
            raise ValueError("Window size is not a multiple of 3!")
            
        if step_size % 3 != 0: 
            raise ValueError("Step size is not a multiple of 3!")
        
        #convert genome to a string
        this_seq = str(SeqIO.read(seq, 'fasta').seq)
        #adjust string to have length which is a multiple of 3 
        self.seq = self.adjustLength(this_seq)

        self.window_size = window_size
        self.step_size = step_size
        self.num_steps = int((len(self.seq)-window_size+step_size)/step_size)
        self.outfile = outfile
        
        self.calculateGCFrame()
        
    def calculateGCFrame(self): 
        """
        Calculate the GC content of each codon base for the given window size and step size 
        """ 
        
        #generate arrays to store values 
        self.gc1, self.gc2, self.gc3 = np.zeros((self.num_steps)), np.zeros((self.num_steps)), np.zeros((self.num_steps)) 
    
        #populate arrays 
        for i in range(self.num_steps): 
    
            #get subsequence for this window
            a = self.seq[i*self.step_size:i*self.step_size+self.window_size]

            #calculate the GC content for each codon base 
            self.gc1[i] = self.GC_content([a[i*3] for i in range(int(len(a)/3))])
            self.gc2[i] = self.GC_content([a[i*3+1] for i in range(int(len(a)/3))])
            self.gc3[i] = self.GC_content([a[i*3+2] for i in range(int(len(a)/3))])
        
    def GC_content(self,baselist): 
        """
        Get the GC content of a list of base pairs 
        """ 
        
        return (baselist.count('G')+baselist.count('C'))*100/len(baselist)

    def adjustLength(self, this_seq): 
        """
        Add Ns to adjust the length of the sequence
        """ 
        
        if len(this_seq) % 3 == 1:
            this_seq= this_seq + 'N'

        elif len(this_seq) % 3 ==2: 
            this_seq = this_seq + 'NN'
        
        return this_seq
    
    def generateTable(self): 
        """
        GC plot table to store which can be read into aretmis. 
        Colours are currently hard-coded but could be changed. 
        """ 

        #value to centre each value within the step 
        centre = int(np.median([i for i in range(self.step_size)]))

        #generate the position of each step 
        bases = [i*self.step_size + centre for i in range(self.num_steps)]
    
        #write to file
        with open(self.outfile, 'w') as f: 
            f.write('# BASE VAL1 VAL2' ) 
            f.write('\n')
            f.write('# color 0:255:0 0:0:255 100:100:100')

            #write each line to file
            for i in range(self.num_steps): 
                f.write('\n')
                line = str(bases[i]) + '\t' + str(self.gc1[i]) + '\t' + str(self.gc2[i]) + '\t' + str(self.gc3[i])
                f.write(line)