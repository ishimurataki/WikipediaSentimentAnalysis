# Setting working directory
import os
path = "/Users/takanaoishimura/Desktop/Kaggle/WikiSentiment" 
os.chdir(path)
print(os.getcwd())
del(path)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from text_functions import sentences_to_popularity_array

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Toxic Comment analysis
# Subsetting dataset to have equal number of toxic and nontoxic comments
nsamples = sum(dataset.iloc[:,2])
toxic_indexes = random.sample(dataset.index[dataset['toxic'] == 1].tolist(), 
                              nsamples)
nontoxic_indexes = random.sample(dataset.index[dataset['toxic'] == 0].tolist(), 
                                 int(nsamples*2))
keep_rows_toxic = toxic_indexes + nontoxic_indexes
random.shuffle(keep_rows_toxic)

dataset_toxic = dataset.iloc[keep_rows_toxic, [1,2]]
del(keep_rows_toxic, toxic_indexes, nontoxic_indexes, nsamples)

# Create indexed_sentences and describer for toxic comment dataset
indexed_sentences, describer = sentences_to_popularity_array(dataset_toxic)
describer.to_csv('WordPopularity.csv')

# Create training and testing sets
X_train = indexed_sentences[:30000, ]
Y_train = np.asarray(dataset_toxic.iloc[:30000, 1])
X_test = indexed_sentences[30000:len(indexed_sentences), ]
Y_test = np.asarray(dataset_toxic.iloc[30000:len(indexed_sentences), 1])

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Pad the sequence to the same length
max_review_length = 1600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Using embedding from Keras
embedding_vector_length = 300
model = Sequential()
model.add(Embedding(10001, embedding_vector_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Callback for logging every batch rather than epochs
class LossAccuracyHistory(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_losses = []
        self.val_accuracy = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
      
        x, y = self.test_data
        index = random.sample(range(0, len(x)), 50)
        x = x[index]
        y = y[index]
        test_loss_batch, test_acc_batch = self.model.evaluate(x, y, verbose=0)
        self.val_losses.append(test_loss_batch)
        self.val_accuracy.append(test_acc_batch)
        
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

learning_history = LossAccuracyHistory((X_test, Y_test))
model.fit(X_train, Y_train, epochs=3,
          batch_size=64, callbacks = [learning_history])

# Evaluation on the test set
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Prediction array
Y_pred = model.predict(X_test)
Y_pred = np.round(Y_pred)

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Save model object to machine
model.save("my_model.h5")

###############################################
# Visualizing Training Learning History 
       
acc_list, loss_list, val_acc_list, val_loss_list = [], [], [], []

z = 5
for i in range(0, int(len(learning_history.losses)/z)):
    acc_mean = sum(learning_history.accuracy[z*i:z*(i+1)])/z
    loss_mean = sum(learning_history.losses[z*i:z*(i+1)])/z
    val_acc_mean = sum(learning_history.val_accuracy[z*i:z*(i+1)])/z
    val_loss_mean = sum(learning_history.val_losses[z*i:z*(i+1)])/z
    
    acc_list.append(acc_mean)
    loss_list.append(loss_mean)
    val_acc_list.append(val_acc_mean)
    val_loss_list.append(val_loss_mean)
    

df = pd.DataFrame({'acc':np.asarray(acc_list),
                   'val_acc':np.asarray(val_acc_list),
                   'loss':np.asarray(loss_list),
                   'val_loss':np.asarray(val_loss_list)})
df.to_csv('AccLoss_Improv.csv')

# summarize history for accuracy
plt.plot(df['acc'][:150])
plt.plot(df['val_acc'][:150])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('batch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid('True')
plt.show()

# summarize history for loss
plt.plot(df['loss'][:400])
plt.plot(df['val_loss'][:400])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid('True')
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='model.png')

##################################################################
# Classifying custom comments as toxic or non-toxic

# Import libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords

while True:
    custom = input("Enter your own sentence:\n")
    
    # Clean up the comment
    custom = re.sub(pattern =  '[^a-zA-Z]', 
                    repl = ' ',
                    string = custom)
    custom = custom.lower()
    custom = custom.split()
    remove_words = []
    for word in custom:
        if word in set(stopwords.words('english')) or len(word) == 1:
            remove_words.append(word)
    for word in remove_words:
            custom.remove(word)
    
    remove_words = []
    for i in range(0, len(custom)):
        word = custom[i]
        if word in describer['word'].unique():
            rank = int(describer.iloc[:,1][describer['word'].isin([word])])
            custom[i] = rank
        else:
            remove_words.append(word)
    for word in remove_words:
        custom.remove(word)
        
    custom = [0]*(max_review_length  - len(custom)) + custom
    custom = np.asarray(custom).reshape((1,len(custom)))
    
    prob_of_censor = float(model.predict(custom))
    censor = int(np.round(model.predict(custom)))

    if censor == 1:
        print("******* YOUR COMMENT IS TOXIC!! IT WILL BE CENSORED ********")
    else:
        print("Your comment is just fine!")
    print("Censorship Probability: " + str(round(prob_of_censor, 4)))
    
    cont = input("Continue? [Y/n]")
    if cont and cont.lower() in ('N', 'n'):
      break
        
