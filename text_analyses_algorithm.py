# Setting working directory
import os
path = "/Users/takanaoishimura/Desktop/Kaggle/WikiSentiment" 
os.chdir(path)
print(os.getcwd())
del(path)

import numpy as np
import pandas as pd
import random
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

# Create training and testing sets
X_train = indexed_sentences[:30000, ]
Y_train = np.asarray(dataset_toxic.iloc[:30000, 1])
X_test = indexed_sentences[30000:len(indexed_sentences), ]
Y_test = np.asarray(dataset_toxic.iloc[30000:len(indexed_sentences), 1])

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard

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

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

# Evaluation on the test set
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

Y_pred = model.predict(X_test)
Y_pred = np.round(Y_pred)

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
        
