# Wiki Sentiments Bag of Words Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setting working directory
import os
path = "/Users/takanaoishimura/Desktop/Kaggle/WikiSentiment" 
os.chdir(path)
print(os.getcwd())
del(path)

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_toxic = dataset.iloc[:,[1,2]]

# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 2000):
    review = re.sub(pattern =  '[^a-zA-Z]', 
                    repl = ' ',
                    string = dataset_toxic["comment_text"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    for word in review:
        if word in set(stopwords.words('english')):
            review.remove(word)
        else:
            review[review.index(word)] = ps.stem(word)
    review = ' '.join(review)
    corpus.append(review)

del(word)
del(review)
del(i)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(raw_documents = corpus).toarray()
Y = dataset_toxic.iloc[0:3000, 1]

# Splitting the dataset into Training set and Testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Fitting Classifer to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test Set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = Y_test, y_pred = Y_pred)
cm

