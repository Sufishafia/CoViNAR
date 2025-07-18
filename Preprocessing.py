import nltk
from nltk.corpus import stopwords
import re

#Preprocess the tweets

def preprocess_tweet(tweet):
    # Remove links, special characters, emojis, and URLs
    tweet = re.sub(r'http\S+|www\S+|[^\w\s#@]', '', tweet)
    
    # Remove hashtag sign, but keep hashtag text
    tweet = re.sub(r'#', '', tweet)
    
    # Remove mention sign, but keep mentioned text
    tweet = re.sub(r'@', '', tweet)
    
    # Remove emojis
    tweet = emoji.demojize(tweet, delimiters=("", ""))
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase
    tweet = tweet.lower()
       
    return tweet
# Remove duplicate tweets
df = df.drop_duplicates(subset='tweet_column_name')




#Remove stopwords

def remove_stopwords(text):
    new_text=[]

    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)

    x=new_text[:]
    new_text.clear()
    return " ".join(x)


df['Text']=df['Text'].apply(remove_stopwords)

df.to_csv("path to preprocessed data.csv")
