# FUNCTIONS FOR CONVERTING TEXT TO VECTOR OF POPULARITY RANKINGS

# Import libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords

# Function to convert comments in dataset to vector of indexes
def get_sentences(data_in):
    corpus = []
    print(len(data_in))
    for i in range(0, len(data_in)):
        if i % 100 == 0:
            print(i)
        review = re.sub(pattern =  '[^a-zA-Z]', 
                        repl = ' ',
                        string = data_in.iloc[i, 0])
        review = review.lower()
        review = review.split()
        remove_words = []
        for word in review:
            if word in set(stopwords.words('english')) or len(word) == 1:
                remove_words.append(word)
        for word in remove_words:
            review.remove(word)
        corpus.append(review)
    del(word)
    del(review)
    del(i)
    return corpus

# Function to convert comments in dataset to vector of indexes and 
# restrict the size of vocabulary to a specified value
def sentences_to_popularity_array(data_in, n_vocab=10000):
  sentences = get_sentences(data_in)
  indexed_sentences = []

  i = 2
  word2idx = {'START': 0, 'END': 1}

  word_idx_count = {
    0: float('inf'),
    1: float('inf'),
  }

  for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
      if token not in word2idx:
        word2idx[token] = i
        i += 1

      # keep track of counts for later sorting
      idx = word2idx[token]
      word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

      indexed_sentence.append(idx)
    indexed_sentences.append(indexed_sentence)

  # restrict vocab size
  import operator
  sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
  sorted_word_idx_count = sorted_word_idx_count[:n_vocab+2]
  word_count_dictionary = {}
  for i in range(0,len(sorted_word_idx_count)):
      word_count_dictionary[sorted_word_idx_count[i][0]] = sorted_word_idx_count[i][1]
      
  idx_to_rank = {}
  for i in range(2, len(sorted_word_idx_count)):
      idx = sorted_word_idx_count[i][0]
      idx_to_rank[idx] = i - 1

  for i in range(0, len(indexed_sentences)):
      if i % 100 == 0:
          print(i)
      sentence = indexed_sentences[i]
      remove_indexes = []
      for z in range(0, len(sentence)):
          wordkey = sentence[z]
          if wordkey not in word_count_dictionary:
              remove_indexes.append(z)
          else:
              indexed_sentences[i][z] = idx_to_rank[wordkey]
      remove_indexes = sorted(remove_indexes, reverse = True)
      for idx in remove_indexes:
          indexed_sentences[i].pop(idx)
          
  indexed_sentences = np.asarray(indexed_sentences)
  
  # Creating dataset with words, idx, count, rank
  idx2word = {}
  for word, key in word2idx.items():
      idx2word[key] = word
      
  top_words, count_list, rank_list = [], [], []
  
  for key,rank in idx_to_rank.items():
      word = idx2word[key]
      count = word_count_dictionary[key]      
      top_words.append(word)
      count_list.append(count)
      rank_list.append(rank)
      
  df = pd.DataFrame({'word':np.asarray(top_words), 
                     'count':np.asarray(count_list), 
                     'rank':np.asarray(rank_list)})
  
  # Return the indexed_sentences and the word describer dataframe 
  return indexed_sentences, df
