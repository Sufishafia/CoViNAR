#BERTopic

#pip install hdbscan
#pip install sentence_transformers
#pip install umap-learn==0.5.1
#pip install umap-learn
#pip install bertopic

# Import necessary libraries
# Data processing
import pandas as pd
import numpy as np
# Dimension reduction
# from umap import umap_ as UMAP
import umap.umap_ as umap
from sklearn.decomposition import PCA
# Clustering
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
# Count vectorization
from sklearn.feature_extraction.text import CountVectorizer
# # Sentence transformer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

data = pd.read_csv('2021_4.csv', encoding='utf8') #, lineterminator='\n' , encoding = 'latin-1')
df1 = data.iloc[:100000, :]
arr = df1['ProcessedText'].tolist()

#UMAP + HDBSCAN

# Initiate UMAP. Set the random state of the UMAP model to prevent stochastic behavious
umap_model = umap.UMAP(n_neighbors=20,
                  n_components=5,
                  min_dist=0.0,
                  metric='cosine',
                  random_state=100) # metric can be euclidean, manhattan, etc

hdbscan_model= HDBSCAN(min_cluster_size=21, min_samples = 2, metric='euclidean', prediction_data=True)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2") #384-dimensional sentence embeddings

from bertopic.representation import KeyBERTInspired
# Train model     # instead run in BATCHES
# Create BERTopic model
topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    n_gram_range=(1, 3),
    verbose=True
)

# Fit the model and get topics
topics, probs = topic_model.fit_transform(arr)

# Display topic information
print(topic_model.get_topic_info().to_string())


# topics, probs = topic_model.fit_transform(arr)
# print(topic_model.get_topic_info().to_string())
# topic_model.get_topic_info()

#Check Coherence
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# Preprocess documents
cleaned_docs = topic_model._preprocess_text(arr)

# Extract vectorizer and tokenizer from BERTopic
vectorizer = topic_model.vectorizer_model
analyzer = vectorizer.build_analyzer()
# tokenizer = vectorizer.build_tokenizer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names_out()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in topic_model.get_topic(topic)]
            for topic in range(len(set(topics))-1)]

# Evaluate
coherence_model = CoherenceModel(topics=topic_words,
                              texts=tokens,
                              corpus=corpus,
                              dictionary=dictionary,
                              coherence='c_v')
coherence = coherence_model.get_coherence()

print(coherence)
# Define the mapping of topic numbers to labels
topic_label_mapping = {-1:0, ...}

# Apply the mapping to create a new 'label' column
df1['label'] = df1['topics'].map(topic_label_mapping)_


# Keyword Filtering
import pandas as pd
import re

# Define your keywords for filtering
keywords = [ 'oxygen',
    'bed', 
    'hospital',
    'urgent',
    'need',
    'available',
    'help',
    'plasma',
    'remdesivir',
    'icu',
    'ventilator',
    'tocilizumab',
    'ambulance',
    
    'emergency',
    
    'oximeter',
    'vaccine',
    'shortage' ]

# Get topic information
topic_info = topic_model.get_topic_info()

# Method 1: Filter topics based on topic names/representations
def filter_topics_by_keywords(topic_model, keywords):
    """
    Filter topics that contain any of the specified keywords in their representation
    """
    filtered_topic_ids = []
    
    # Get all topics (excluding -1 which is outlier topic)
    for topic_id in topic_model.get_topics().keys():
        if topic_id == -1:  # Skip outlier topic
            continue
            
        # Get topic representation (top words)
        topic_words = topic_model.get_topic(topic_id)
        topic_words_list = [word for word, score in topic_words]
        
        # Check if any keyword is in the topic words
        for keyword in keywords:
            if any(keyword.lower() in word.lower() for word in topic_words_list):
                filtered_topic_ids.append(topic_id)
                break
    
    return filtered_topic_ids

# Get filtered topic IDs
filtered_topic_ids = filter_topics_by_keywords(topic_model, keywords)
print(f"Filtered Topic IDs: {filtered_topic_ids}")

# Method 2: Create dataframe with tweets from filtered topics
def create_filtered_dataframe(df_original, topics, filtered_topic_ids):
    """
    Create a dataframe containing only tweets from filtered topics
    """
    # Create a dataframe with original data, topics, and probabilities
    result_df = df_original.copy()
    result_df['Topic'] = topics
    result_df['Probability'] = probs if 'probs' in globals() else None
    
    # Filter for selected topics only
    filtered_df = result_df[result_df['Topic'].isin(filtered_topic_ids)]
    
    return filtered_df

# Create the filtered dataframe
# Assuming your original dataframe is df1 (first 100k rows)
filtered_tweets_df = create_filtered_dataframe(df, topics, filtered_topic_ids)

print(f"\nTotal tweets in filtered topics: {len(filtered_tweets_df)}")
print(f"Distribution across filtered topics:")
print(filtered_tweets_df['Topic'].value_counts().sort_index())

# Method 3: Get detailed topic information for filtered topics
def get_filtered_topic_details(topic_model, filtered_topic_ids):
    """
    Get detailed information about filtered topics
    """
    filtered_topic_details = []
    
    for topic_id in filtered_topic_ids:
        topic_words = topic_model.get_topic(topic_id)
        top_words = [word for word, score in topic_words[:10]]  # Top 10 words
        
        filtered_topic_details.append({
            'Topic_ID': topic_id,
            'Top_Words': ', '.join(top_words),
            'Count': len([t for t in topics if t == topic_id])
        })
    
    return pd.DataFrame(filtered_topic_details)

# Get details of filtered topics
topic_details_df = get_filtered_topic_details(topic_model, filtered_topic_ids)
print(f"\nFiltered Topic Details:")
print(topic_details_df.to_string(index=False))

# Method 4: Advanced filtering - search within tweet text as well
def advanced_keyword_filtering(df_with_topics, keywords, text_column='Clean_Text'):
    """
    Filter tweets that either belong to keyword-related topics OR contain keywords in text
    """
    # Filter by topic (already done above)
    topic_filtered = df_with_topics[df_with_topics['Topic'].isin(filtered_topic_ids)]
    
    # Additionally filter by text content
    keyword_pattern = '|'.join([re.escape(keyword.lower()) for keyword in keywords])
    text_filtered = df_with_topics[
        df_with_topics[text_column].str.lower().str.contains(keyword_pattern, na=False)
    ]
    
    # Combine both filters (union)
    combined_filtered = pd.concat([topic_filtered, text_filtered]).drop_duplicates()
    
    return combined_filtered, topic_filtered, text_filtered

# Apply advanced filtering
combined_df, topic_only_df, text_only_df = advanced_keyword_filtering(filtered_tweets_df, keywords)

print(f"\nAdvanced Filtering Results:")
print(f"Tweets filtered by topic only: {len(topic_only_df)}")
print(f"Tweets filtered by text content only: {len(text_only_df)}")
print(f"Combined (topic OR text): {len(combined_df)}")

filtered_tweets_df.shape

import re

# Define phrase-based patterns
need_patterns = [
    r'\bneed(ed)?\b', r'\burgent(ly)?\b', r'\brequire(d|ment)?\b', 
    r'looking for', r'please help', r'any leads', r'sos', r'immediate requirement'
]
available_patterns = [
    r'\bavailable\b', r'\bproviding\b', r'\bverified\b', r'can help', 
    r'has oxygen', r'beds open', r'plasma donor'
]

# Combine patterns into regex
need_regex = re.compile('|'.join(need_patterns), re.IGNORECASE)
available_regex = re.compile('|'.join(available_patterns), re.IGNORECASE)

# Apply to filtered tweets
def classify_by_regex(text):
    if pd.isnull(text):
        return 'Not Relevant'
    if need_regex.search(text):
        return 'Need'
    elif available_regex.search(text):
        return 'Available'
    else:
        return 'Not Relevant'

filtered_tweets_df['Rule_Label'] = filtered_tweets_df['Clean_Text'].apply(classify_by_regex)
