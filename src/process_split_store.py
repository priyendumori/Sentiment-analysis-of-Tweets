
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("../input_data/data.csv", names=["A","B","C","D","E","F"])
df = df.drop(["B","C","D","E"], axis=1)
print df.head()


# In[2]:


from porter_stemmer import PorterStemmer
import re
import string
from sklearn.feature_extraction import stop_words

def processTweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    #remove @username
    tweet = re.sub('@[^\s]+','',tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = re.sub(' +', ' ',tweet)
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ')  
    # Removing Stopwords from tweet using sklearn.feature_extraction
    split_list = tweet.split(" ")
    tweet = [ word for word in split_list if word not in stop_words.ENGLISH_STOP_WORDS ]
    # Stemming the 
    ps = PorterStemmer()
#     print tweet
#     t = []
#     for word in tweet:
#         print word
#         t.append(ps.stem(word))
    tweet = [ ps.stem(word) for word in tweet ] 
#     tweet = t
    tweet = ' '.join(tweet)
    return tweet




processed_data = list()

for index, row in df.iterrows():
    processed_data.append(processTweet(row['F']))


# In[3]:


df['processed'] = processed_data
df.head()


# In[4]:


from sklearn.model_selection import train_test_split
X = df['processed']
Y = df['A']
X_train_val, X_test , Y_train_val, Y_test = train_test_split(X,Y,test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val,Y_train_val,test_size=0.25)


# In[14]:


train_df = pd.concat([X_train, Y_train],axis='columns').reset_index(drop=True)
train_df.head()
train_df.to_csv("../input_data/train_df.csv", sep=',',index=False)


# In[6]:


validation_df = pd.concat([X_val, Y_val],axis='columns').reset_index(drop=True)
validation_df.to_csv("../input_data/validation_df.csv", sep=',',index=False)


# In[7]:


test_df = pd.concat([X_test, Y_test],axis='columns').reset_index(drop=True)
test_df.to_csv("../input_data/test_df.csv", sep=',',index=False)

