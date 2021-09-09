from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv1D
from  tensorflow.keras.layers import MaxPooling1D
from  tensorflow.keras.layers import Flatten
from  tensorflow.keras.layers import Dense
from  tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def conv_net(u):
    model = Sequential()
    #Layer 1
    #Conv Layer 1
    model.add(Conv1D(filters = 10, 
                    kernel_size = 5, #grandezza kernel
                    strides = 1, 
                    activation = 'tanh', 
                    input_shape = (1024,1))) #width, height, channels ??
    #Pooling layer 1
    model.add(MaxPooling1D(pool_size= 2, strides = 1))
    #Layer 2  
    #Conv Layer 2
    model.add(Conv1D(filters = 20, 
                    kernel_size = 5,
                    strides = 1,
                    activation = 'tanh',
                    )) 
    #Pooling Layer 2
    model.add(MaxPooling1D(pool_size= 2, strides = 1))
    #Flatten
    model.add(Flatten())
    #Layer 3
    #Fully connected layer 1
    model.add(Dense(units = 500, activation = 'relu'))
    #Layer 4
    #Fully connected layer 2
    #model.add(Dense(units = 84, activation = 'relu'))
    #Layer 5
    #Output Layer
    model.add(Dense(units = u, activation = 'softmax')) #5 classi
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', f1_m])
    return model