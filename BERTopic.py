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

