import matplotlib.pyplot as plt
import numpy as np


columns = ['phylum' , 'class', 'order', 'family', 'genus']
label = ['svm', 'cnn', 'nb', 'rf']
color =['g--', 'r-', 'b-', 'y--']
# PLOT ACCURACY FULL LENGTH   (CNN, SVC, NB, RF)
i=0
for item in label:
    lista = np.load('data_accuracy/accuracy_'+ item +'_full.npy', allow_pickle=True)
    item_l = []
    for v in lista:
        item_l.append(v)
    plt.plot(columns, item_l, color[i], label = str(item))
    i += 1

plt.title("Accuracy full length - CNN vs other classifiers")
plt.legend()
plt.show()

# PLOT f1 FULL LENGTH   (CNN, SVC, NB, RF)
i=0
for item in label:
    lista = np.load('data_f1/f1_'+ item +'_full.npy', allow_pickle=True)
    item_l = []
    for v in lista:
        item_l.append(v)
    plt.plot(columns, item_l, color[i], label = str(item))
    i += 1

# show a legend on the plot
plt.title("F1 full length - CNN vs other classifiers")
plt.legend()

# Show the plot
plt.show()

'''
#SVC 
list_SVC = np.load('data_accuracy/accuracy_svm_full.npy', allow_pickle=True)
# Data for  plot
x_svm = []
for v in list_SVC: #prediction_SVC.values()
    x_svm.append(v)
# Create the plot
plt.plot(columns, x_svm, 'g--', label="SVM" )


#CNN
list_CNN = np.load('data_accuracy/accuracy_cnn_full.npy', allow_pickle=True)
x_cnn = [] 
for v in list_CNN: # prediction values cnn
    x_cnn.append(v)
plt.plot(columns, x_cnn, 'r-', label="CNN" )


#NB
list_NB = np.load('data_accuracy/accuracy_nb_full.npy', allow_pickle=True)
x_nb = [] 
for v in list_NB: # prediction values cnn
    x_nb.append(v)
plt.plot(columns, x_nb, 'b-', label="NB" )

#rf
list_RF = np.load('data_accuracy/accuracy_rf_full.npy', allow_pickle=True)
x_rf = [] 
for v in list_RF: # prediction values cnn
    x_rf.append(v)
plt.plot(columns, x_rf, 'y--', label="RF" )

plt.title("Accuracy full length - CNN vs other classifiers")
plt.legend()

# Show the plot
plt.show()

#PLOT F1_ FULL  _full

list_SVC = np.load('data_accuracy/f1_svm_full.npy', allow_pickle=True)
# Data for  plot
x_svm = []
for v in list_SVC: #prediction_SVC.values()
    x_svm.append(v)
# Create the plot
plt.plot(columns, x_svm, 'g--', label="SVM" )
# r- is a style code meaning red solid line

list_CNN = np.load('data_accuracy/f1_cnn_full.npy', allow_pickle=True)
x_cnn = [] 
for v in list_CNN: # prediction values cnn
    x_cnn.append(v)
# Create the plot
plt.plot(columns, x_cnn, 'r-', label="CNN" )

#NB
list_NB = np.load('data_accuracy/f1_nb_full.npy', allow_pickle=True)
x_nb = [] 
for v in list_NB: # prediction values cnn
    x_nb.append(v)
plt.plot(columns, x_nb, 'b-', label="NB" )

#rf
list_RF = np.load('data_accuracy/f1_rf_full.npy', allow_pickle=True)
x_rf = [] 
for v in list_RF: # prediction values cnn
    x_rf.append(v)
plt.plot(columns, x_rf, 'y--', label="RF" )

# show a legend on the plot
plt.title("F1 full length - CNN vs other classifiers")
plt.legend()

# Show the plot
plt.show()
'''