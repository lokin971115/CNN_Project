import tensorflow as tf
import numpy as np


def load_data_score():
    graphsize=np.loadtxt('../Tox21_AR/score_graph_size.csv',dtype=np.int,delimiter=',',skiprows=1)
    graphsize=graphsize[:,1:]
    datasetnum=graphsize.shape[0]
    graphsize=np.array(graphsize)
    graphsize=graphsize.reshape((datasetnum))
    adjgraph=np.loadtxt('../Tox21_AR/score_graphs.csv',dtype=np.float32,delimiter=',')
    adjgraph=np.reshape(adjgraph,(datasetnum,132,132,1))
    return datasetnum,graphsize,adjgraph


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
    

datasetnum,graphsize,adjgraph=load_data_score()        
model=createmodel()
sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('.')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
tf.keras.backend.set_session(sess)
adjgraph=adjgraph.astype(np.float32)
predictions=model.predict(adjgraph)
result=[]
for i, probi in enumerate(predictions):
    class_idx = np.argmax(probi)
    result.append(class_idx)
result=np.array(result)
np.savetxt('labels.txt',result,fmt='%i')

