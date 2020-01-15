import tensorflow as tf
import numpy as np


def load_data(traintestscore):
    if(traintestscore==0):
        path = 'train_data\\train_'
    elif(traintestscore==1):
        path = 'test_data\\test_'
    elif(traintestscore==2):
        path='Tox21_AR/score_data/score_'
    graphsize=np.loadtxt(path+'graph_size.csv',dtype=np.int,delimiter=',',skiprows=1)
    labels=np.loadtxt(path+'labels.csv',dtype=np.int,delimiter=',',skiprows=1)
    graphsize=graphsize[:,1:]
    datasetnum=graphsize.shape[0]
    graphsize=np.array(graphsize)
    graphsize=graphsize.reshape((datasetnum))
    labels=labels[:,1:]
    labels=np.array(labels)
    labels=labels.reshape((datasetnum))
    adjgraph=np.loadtxt(path+'graphs.csv',dtype=np.int,delimiter=',')
    adjgraph=np.reshape(adjgraph,(datasetnum,132,132,1))
    adjgraph_toxic=[]
    for i in range(datasetnum):
        if(labels[i]==1):
            adjgraph_toxic.append(adjgraph[i])
    adjgraph_toxic=np.array(adjgraph_toxic)
    toxic_label=[]
    for i in range(adjgraph_toxic.shape[0]):
        toxic_label.append(1)
    toxic_label=np.array(toxic_label)
    return datasetnum,graphsize,labels,adjgraph,adjgraph_toxic,toxic_label


def createmodel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(132, (3, 3), activation='relu', input_shape=(132, 132, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(66, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(33, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))   
    model.summary()
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


datasetnum,graphsize,labels,adjgraph,adjgraph_toxic,toxic_label=load_data(0)
smallset=np.concatenate((adjgraph[0:500], adjgraph_toxic), axis=0)
smalllabel=np.concatenate((labels[0:500], toxic_label), axis=0)
#for i in range(5):
    #adjgraph=np.concatenate((adjgraph, adjgraph_toxic), axis=0)
    #labels=np.concatenate((labels, toxic_label), axis=0)
#print(adjgraph.shape)
#print(labels.shape)
datasetnum_test,graphsize_test,labels_test,adjgraph_test,adjgraph_toxic_test,toxic_label_test=load_data(1)        
model=createmodel()
model.fit(smallset, 
          smalllabel,  
          epochs=10,
          validation_data=(adjgraph_test,labels_test)
          )
test=adjgraph_test[0:114]
labelstest=labels_test[0:114]
results = model.evaluate(test, labelstest, verbose=2)
print('test loss, test acc:', results)
test=adjgraph_test[114:228]
labelstest=labels_test[114:228]
results = model.evaluate(test, labelstest,verbose=2)
print('test loss, test acc:', results)
sess=tf.keras.backend.get_session()     
saver = tf.train.Saver()
saver.save(sess, './my_model', global_step = 1)
