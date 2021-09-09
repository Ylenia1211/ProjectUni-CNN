import matplotlib.pyplot as plt
import numpy as np


columns = ['phylum' , 'class', 'order', 'family', 'genus']
label = ['svm', 'cnn', 'nb', 'rf']  #cnn
color =['g--', 'r-', 'b-', 'y--']
# PLOT ACCURACY short LENGTH   (CNN, SVC, NB, RF)
i=0
for item in label:
    lista = np.load('data_accuracy/accuracy_'+ item +'_short.npy', allow_pickle=True)
    item_l = []
    for v in lista:
        item_l.append(v)
    plt.plot(columns, item_l, color[i], label = str(item))
    i += 1

plt.title("Accuracy short length - CNN vs other classifiers")
plt.legend()
plt.show()

# PLOT f1 short LENGTH   (CNN, SVC, NB, RF)
i=0
for item in label:
    lista = np.load('data_f1/f1_'+ item +'_short.npy', allow_pickle=True)
    item_l = []
    for v in lista:
        item_l.append(v)
    plt.plot(columns, item_l, color[i], label = str(item))
    i += 1

# show a legend on the plot
plt.title("F1 short length - CNN vs other classifiers")
plt.legend()

# Show the plot
plt.show()
