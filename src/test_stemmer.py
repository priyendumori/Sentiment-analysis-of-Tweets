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
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ')  
    # Removing Stopwords from tweet using sklearn.feature_extraction
    split_list = tweet.split(" ")
    tweet = [ word for word in split_list if word not in stop_words.ENGLISH_STOP_WORDS ]
    # Stemming the 
    ps = PorterStemmer()
    tweet = [ ps.stem(word) for word in tweet ] 
    tweet = ' '.join(tweet)
    return tweet


import pandas as pd

df = pd.read_csv("../input_data/data.csv")
print df.head()

processed_data = list()


for _, row in df.iterrows():
    processed_data.append(processTweet(row))

df.processed = processed_data


# sample_tweet = '''@sonamakapoor  its because of people like you who dont use public transport or less fuel consumption vehicles You Know that your luxury car gives 3 or 4 km per litre mileage and  10 20 ACs in your house are equally responsible for global warming First control your pollution'''

# print processTweet(sample_tweet)