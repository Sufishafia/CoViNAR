#Scrape the tweets
import snscrape.modules.twitter as sntwitter
import pandas as pd

# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('(#coronovirus OR #covid OR #covid19 OR #pandemic OR #lockdown OR #virus OR #outbreak OR # china OR #deaths) since:2023-09-10 until:2023-09-11 lang:en').get_items()):
    if i>200:
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.place])

# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text','Location'])

print(tweets_df2.shape)
print(tweets_df2.head())





