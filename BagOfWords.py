# Wiki Sentiments Bag of Words Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Setting working directory
import os
path = "/Users/takanaoishimura/Desktop/Kaggle/WikiSentiment" 
os.chdir(path)
print(os.getcwd())
del(path)

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Subsetting dataset to have equal number of toxic and nontoxic comments
nsamples = sum(dataset.iloc[:,2])
toxic_indexes = random.sample(dataset.index[dataset['toxic'] == 1].tolist(), 
                              nsamples)
nontoxic_indexes = random.sample(dataset.index[dataset['toxic'] == 0].tolist(), 
                                 int(nsamples*2))
keep_rows_toxic = toxic_indexes + nontoxic_indexes
random.shuffle(keep_rows_toxic)

dataset_toxic = dataset.iloc[keep_rows_toxic, [1,2]]
dataset_toxic.index = range(0, len(dataset_toxic.index))
del(keep_rows_toxic, toxic_indexes, nontoxic_indexes, nsamples)


# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

corpus = []
for i in range(0, len(dataset_toxic)):
    print(i)
    review = re.sub(pattern =  '[^a-zA-Z]', 
                    repl = ' ',
                    string = dataset_toxic["comment_text"][i])
    review = review.lower()
    review = review.split()
    remove_words = []
    for word in review:
        if word in set(stopwords.words('english')) or len(word) == 1:
            remove_words.append(word)
    for word in remove_words:
        review.remove(word)
    review = ' '.join(review)
    corpus.append(review)

del(word)
del(review)
del(i)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(raw_documents = corpus).toarray()
Y = dataset_toxic.iloc[:,1]

# Splitting the dataset into Training set and Testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test Set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = Y_test, y_pred = Y_pred)
cm

