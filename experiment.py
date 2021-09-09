from Bio import SeqIO
import itertools
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from LeNet import conv_net
from model_grnn import GRNN
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from neupy import algorithms    #Grnn ma senza possibilità di cambiare la dist

'''
STRUTTURE DATI UTILI
'''
sequence_array = []
id_array = []
taxonomy = dict()
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


#Carico il conteggio per ogni k-mers (4*5) PER OGNI SEQUENZA (tot.3000) 
kmers_counter = np.load('_kmersCount.npy', allow_pickle=True)

# uso data structure per immagazzianre i dati parziali
test_predicted = np.zeros((np.array(label_list).shape[0],))
test_predicted_svc = np.zeros((np.array(label_list).shape[0],))
test_predicted_bayes = np.zeros((np.array(label_list).shape[0],))
test_predicted_clf =  np.zeros((np.array(label_list).shape[0],))
test_predicted_grnn =  np.zeros((np.array(label_list).shape[0],))
scoresAcc_CNN = []
scoresF1_CNN = []


dict_model = {"cnn": {"accuracy": [], "f1": [] },
              "svm": {"accuracy": [], "f1": [] },
              "nb": {"accuracy": [], "f1": [] },
              "rf": {"accuracy": [], "f1": [] }, 
              "grnn": {"accuracy": [], "f1": [] }, 
              # GRNN eucl, GRNN jaccard, GRNN city
            }

data_obj = pd.DataFrame(label_list, columns = ['phylum' , 'class', 'order', 'family', 'genus'])
columns = ['phylum' ,'class', 'order', 'family', 'genus']
#mi serve per passare alla rete convolutiva il giusto numero di unità output quando cambio la label( phylum, class, genus ...ect)
#dict_unit = {'phylum': 3, 'class': 6, 'order':19, 'family': 64, 'genus':393} 

#for col in columns: 
    #lab_col = list(set(data_obj[col]))
    #y = np.array([lab_col.index(x) for x in data_obj[col]]) 
    #dict_unit[col] = len(y)


#labels_not_rip = list(set(data_obj['phylum'])) #prendo le etichette in modo unico, set elimina i duplicati
#y = np.array([labels_not_rip.index(x) for x in data_obj['phylum']]) #assegno ad ogni etichetta il suo valore corrisp. es. Actinobacteria=1 
for col in columns: 
    labels_not_rip = list(set(data_obj[col])) 
    y = np.array([labels_not_rip.index(x) for x in data_obj[col]]) 

    y_cat = to_categorical(y)

    i=0
    fold_obj = StratifiedKFold(10, True)  #10
 
    for train_idx, test_idx in fold_obj.split(kmers_counter, y):
                
                i += 1
                print("fold " + str(i))
               
            
                X_train, X_test = kmers_counter[train_idx, :], kmers_counter[test_idx, :]
                y_train_cat, y_test_cat = y_cat[train_idx], y_cat[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                #BUILD CNN 
                #convolutive network LeNet modified
                X_train_reshaped = np.reshape(X_train, (len(X_train),1024,1))
                X_test_reshaped = np.reshape(X_test, (len(X_test),1024,1))

                #y_train_cat = to_categorical(y_train)
                #y_test_cat =  to_categorical(y_test)
                
                #print("shape y_cat;", type(y_test_cat.shape[1]))
                print(y_test_cat.shape) # print(y_test_cat.shape)
                model = conv_net(y_train_cat.shape[1]) #  y_train_cat.shape[1] passo alla rete convolutiva il numero di unità output dict_unit[col]
                print(model.summary())
                #settare a 200 epoche                y_train_cat
                history = model.fit(X_train_reshaped,y_train_cat, epochs=20, batch_size=32, shuffle=True)
                
                # evaluate the model                     y_test_cat
                scores =  model.evaluate(X_test_reshaped, y_test_cat) #  loss, accuracy, f1_score  verbose=0
                
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))   #acc
                #print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))   #f1_M

                scoresAcc_CNN.append(scores[1])
                scoresF1_CNN.append(scores[2])
                #test_predicted = np.reshape(test_predicted, (len(test_predicted),1024,1))
                #test_predicted[test_idx] = model.predict(X_test_reshaped)
                #print(accuracy_score(y_test, test_predicted[test_idx]))
             
                #BUILD SVC 
                model_svc = SVC(kernel='rbf')
                model_svc.fit(X_train, y_train) 
                test_predicted_svc[test_idx] = model_svc.predict(X_test)
                print(accuracy_score(y_test, test_predicted_svc[test_idx]))

                #BUILD NAIVE BAYESIAN 
                model_naiveBay = GaussianNB()
                model_naiveBay.fit(X_train, y_train) 
                test_predicted_bayes[test_idx] = model_naiveBay.predict(X_test)
                print(accuracy_score(y_test, test_predicted_bayes[test_idx]))
               

                #BUILD RANDOM FOREST 
                clf = RandomForestClassifier(max_depth=None, random_state=0)
                clf.fit(X_train, y_train)
                test_predicted_clf[test_idx] = clf.predict(X_test)
                print(accuracy_score(y_test, test_predicted_clf[test_idx]))    
                
                
                #model GRNN
        
                model = GRNN(X_train, y_train, X_test, y_test)
                test_predicted_grnn[test_idx] = model.predict()
                print(accuracy_score(y_test, test_predicted_grnn[test_idx]))

    
    print("End of kfold") 

    #CNN salvo per ogni label (class, phylum, ecc) i valori di accuracy dopo  il computo su 10 fold
    print("%f" % (np.mean(scoresAcc_CNN)))
    print("%f" % (np.mean(scoresF1_CNN)))
    #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


    #CNN salvo per ogni label (class, phylum, ecc) i valori di accuracy dopo  il computo su 10 fold
    dict_model['cnn']['accuracy'].append(np.mean(scoresAcc_CNN))
    np.save('data_accuracy/accuracy_cnn_full'+str(col), list(dict_model['cnn']['accuracy']))
    dict_model['cnn']['f1'].append(np.mean(scoresF1_CNN))
    np.save('data_f1/f1_cnn_full'+str(col), list(dict_model['cnn']['f1']))

    #SVM
    dict_model['svm']['accuracy'].append(accuracy_score(y, test_predicted_svc))
    dict_model['svm']['f1'].append(f1_score(y, test_predicted_svc, average="micro")) #average micro for multiclass
    
    #NB
    dict_model['nb']['accuracy'].append(accuracy_score(y, test_predicted_bayes))
    dict_model['nb']['f1'].append(f1_score(y, test_predicted_bayes, average="micro")) #average micro for multiclass

    #RF
    dict_model['rf']['accuracy'].append(accuracy_score(y, test_predicted_clf))
    dict_model['rf']['f1'].append(f1_score(y, test_predicted_clf, average="micro")) #average micro for multiclass

    #GRNN
    dict_model['grnn']['accuracy'].append(accuracy_score(y, test_predicted_clf))
    dict_model['grnn']['f1'].append(f1_score(y, test_predicted_clf, average="micro")) #average micro for multiclass
     


#CNN, SVM, RF, NB salvo i dati (accuracy e f1 di ogni classe(phylum, class, genus etc)) su file
for key in dict_model.keys():
    np.save('data_accuracy/accuracy_'+ str(key)+ '_full', list(dict_model[key]['accuracy']))
    np.savetxt('data_accuracy/accuracy_'+ str(key)+ '_full.txt', list(dict_model[key]['accuracy'])) 
    np.save('data_f1/f1_'+ str(key) +'_full', list(dict_model[key]['f1']))
    np.savetxt('data_f1/f1_'+str(key) +'_full.txt', list(dict_model[key]['f1']))

'''
np.save('data_accuracy/accuracy_cnn_full', list(dict_model['cnn']['accuracy']))
np.savetxt('data_accuracy/accuracy_cnn_full.txt', list(dict_model['cnn']['accuracy']))
np.save('data_f1/f1_cnn_full', list(dict_model['cnn']['f1']))
np.savetxt('data_f1/f1_cnn_full.txt', list(dict_model['cnn']['f1']))

#SVM salvo dati
np.save('data_accuracy/accuracy_svm_full', list(dict_model['svm']['accuracy']))
np.savetxt('data_accuracy/accuracy_svm_full.txt', list(dict_model['svm']['accuracy']))
np.save('data_f1/f1_svm_full', list(dict_model['svm']['f1']))
np.savetxt('data_f1/f1_svm_full.txt', list(dict_model['svm']['f1']))

#Naive Bayesian (NB) salvo dati
np.save('data_accuracy/accuracy_nb_full', list(dict_model['nb']['accuracy']))
np.savetxt('data_accuracy/accuracy_nb_full.txt', list(dict_model['nb']['accuracy']))
np.save('data_f1/f1_nb_full', list(dict_model['nb']['f1']))
np.savetxt('data_f1/f1_nb_full.txt', list(dict_model['nb']['f1']))

#Random Forest (RF) salvo dati
np.save('data_accuracy/accuracy_rf_full', list(dict_model['rf']['accuracy']))
np.savetxt('data_accuracy/accuracy_rf_full.txt', list(dict_model['rf']['accuracy']))
np.save('data_f1/f1_rf_full', list(dict_model['rf']['f1']))
np.savetxt('data_f1/f1_rf_full.txt', list(dict_model['rf']['f1']))
'''

print("End totale")

# ********************* Stampa accuracy per ogni modello *************************************
#print ("Accuracy Score full length(CNN): ", accuracy_score(y, test_predicted))
#print("Accuracy Score full length(GRNN): ", accuracy_score(y, test_predicted))

#test_predicted_svc = np.load('predictions_svc.npy', allow_pickle=True)
#print("Accuracy Score full length(SVC): ", accuracy_score(y, test_predicted_svc))


#test_predicted_bayes = np.load('predictions_Nbayesian.npy', allow_pickle=True)
#print("Accuracy Score full length(NB): ", accuracy_score(y, test_predicted_bayes))