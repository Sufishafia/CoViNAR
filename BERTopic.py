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
topic_model = BERTopic(embedding_model=sentence_model, umap_model = umap_model, hdbscan_model= hdbscan_model, representation_model=KeyBERTInspired,n_gram_range=(1,3),verbose=True)
# print(arr[:10])
topics, probs = topic_model.fit_transform(arr)
print(topic_model.get_topic_info().to_string())
topic_model.get_topic_info()

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




















