#Preprocess the tweets
import re
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
    
    # Remove duplicate tweets
    tweet = ' '.join(sorted(set(tweet.split()), key=tweet.split().index))
    
    return tweet
#Remove html tags
# import re
# def remove_html_tags(text):
#     pattern = re.compile('<.*?>')
#     return pattern.sub(r'',text)


# new_df['Text']=new_df['Text'].apply(remove_html_tags)


# #remove URLs

# def remove_urls(text):
#     pattern = re.compile(r'https?://\S+|www\.\S+')
#     return pattern.sub(r'',text)


# new_df['Text']=new_df['Text'].apply(remove_urls)


# #Remove mentions and hashtags from tweets
# new_df['Text'] = new_df['Text'].apply(lambda x: re.sub(r'@[^\s]+','',x))
# new_df['Text'] = new_df['Text'].apply(lambda x: re.sub(r'#[^\s]+','',x))


# #removing punctuations
# import string,time

# exclude=string.punctuation
# def remove_punct(text):
#     for char in exclude:
#         text=text.replace(char,'')
#     return text


# new_df['Text']=new_df['Text'].apply(remove_punct)


# #lowercasing
# new_df = new_df.apply(lambda x:x.str.lower() if x.dtype=='object' else x)


# #Remove stopwords----not to be done if pos tagging have to be done
# import nltk
# from nltk.corpus import stopwords
# def remove_stopwords(text):
#     new_text=[]

#     for word in text.split():
#         if word in stopwords.words('english'):
#             new_text.append('')
#         else:
#             new_text.append(word)

#     x=new_text[:]
#     new_text.clear()
#     return " ".join(x)


# new_df['Text']=new_df['Text'].apply(remove_stopwords)


new_df.to_csv("path to preprocessed data.csv")
