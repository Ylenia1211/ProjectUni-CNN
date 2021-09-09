from Bio import SeqIO
import itertools
import numpy as np
import re
import random

sequence_array = []

def sequence_500bp():

    with open("dati/16S.fas", "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            #print(record.id)  #contiente id della sequenceuenza es.S001014081
            #print(record.seq) #contiene la sequenceuenza es. agtcgggta..
            sequence = record.seq
            
            #manipolare la sequenza per renderla corta 500 bp
            #print(len(sequence))
            #print(len(sequence)//2)  # prendo il floor della divisione
            n = random.randint(0,len(sequence)//2)
            #print("Punto di partenza random della seq",n)
            manipulate_seq = sequence[n: n+500] #prendo 500 basi consecutive a partire da N
            #print("Lunghezza sequenza manipolata", len(manipulate_seq))
            #print(manipulate_seq)

            #inserirla alla lista delle sequenze
            sequence_array.append(manipulate_seq)

        return sequence_array
