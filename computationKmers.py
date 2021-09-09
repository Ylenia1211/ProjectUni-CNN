from Bio import SeqIO
import itertools
import numpy as np
import re

#from neupy import algorithms
'''
STRUTTURE DATI UTILI
'''

sequence_array = []
id_array = []

taxonomy = dict()
alphabet=  ['a','c','g','t']

label_list = []   #contiene tutte le label (di taxonomy.csv) assegnate a una sequenza del file 16S

'''
ESTRAZIONE DATI 
'''
#############  ESTRAZIONE ETICHETTE PHYLUM ######################
#col=1 #perchè la colonna 0 ha l'id, le tassonomie iniziano da col=1
with open("dati/taxonomy.csv", 'r') as fin:
     for line in fin:
        tokens = line.strip().split(',')
                               #phylum      #class    #order    #family     #genus 
        taxonomy[tokens[0]] = [tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]] #creo coppie (id, value(tassonomie))
        #print(tokens[0])
        #print(tokens)




with open("dati/16S.fas", "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        #print(record.id)  #contiente id della sequenceuenza es.S001014081
        #print(record.sequence) #contiene la sequenceuenza es. agtcgggta..
        id_array.append(record.id)
        sequence_array.append(record.seq)
        label_list.append(taxonomy[record.id])

'''
CONTEGGIO 5-MERS PER OGNI sequenceIN 16S.fas
'''
k_len = 5 # LUNGHEZZA KMERS CONSIDERATI

#“bag–of–words” model
kmers = list(itertools.product(['a', 'g', 'c', 't'], repeat=k_len)) #tutti i possibili k-mers con l = 5


#SALVO IL CONTEGGIO DEI KMERS (4*5) PER OGNI SEQUENZA (tot.3000) 

kmers_counter = np.zeros((3000, 4**k_len), dtype=float) #int se non normalizzo

#per ogni sequence (3000) 
#incremento il contatore specifico per un kmer
#tra tutti i possibili 4*5 =1024----> INPUT del rete convolutiva su 

c = 0# tiene conto delle sequence che sto considerando nel for
for sequence in sequence_array:
    sequence = str(sequence) #prendo la sequenza dalla lista di obj Seq()
    for i in range(len(sequence)-k_len+1): #sliding window
        
        kmer_i = sequence[i:i+k_len] #k-mer iesimo
        #print(kmer_i) 


        #su ogni sequence applico il filtro per eliminare i chr non consentiti (N, S, Y ecc) 
        if(len(re.compile('[agct]').findall(kmer_i)) == k_len):
            find_idK = kmers.index(tuple(kmer_i)) #Exception se sequence[i:i+k_len] contiene un kmers non valido (no in bag of words)
            kmers_counter[c, find_idK] += 1  #incremento il contatore per la sequenceuenza c e il kmers con #id=find_idK          

    kmers_counter[c, :] = np.divide(kmers_counter[c, :], len(sequence)) #normalizzazione
    c += 1

np.savetxt('kmersCount.txt', kmers_counter)
np.save('_kmersCount', kmers_counter)