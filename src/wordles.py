
# coding: utf-8

# In[1]:


import pandas as pd
from wordcloud import WordCloud 
from matplotlib import pyplot as plt
df = pd.read_csv("../input_data/train_df.csv")


# In[2]:


def wordle(df):
    all_words = []
    for line in df['processed']:
        try:
            all_words.extend(line.split())
        except:
            pass
    text = " ".join(all_words)
    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[3]:


negative_df= df[df['A']==0].reset_index(drop=True)


# In[4]:


positive_df = df[df['A']==4].reset_index(drop=True)


# In[5]:


wordle(df)


# In[6]:


wordle(positive_df)


# In[7]:


wordle(negative_df)

